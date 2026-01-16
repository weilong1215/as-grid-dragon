# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Bitget Adapter (Zeabur Optimized Version)
==============
修正持倉不同步、平倉後記憶不歸零問題
"""

import json
import logging
import time
import hmac
import base64
import hashlib
from typing import Optional, Dict, List

import ccxt

from .base import (
    ExchangeAdapter,
    TickerUpdate,
    OrderUpdate,
    PositionUpdate,
    BalanceUpdate,
    AccountUpdate,
    PrecisionInfo,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger("as_grid_max")

# Bitget WebSocket URLs
BITGET_WS_MAINNET = "wss://ws.bitget.com/v2/ws/private"
BITGET_WS_PUBLIC_MAINNET = "wss://ws.bitget.com/v2/ws/public"
BITGET_WS_TESTNET = "wss://ws.bitget.com/v2/ws/private"

# Bitget 心跳間隔 (30 秒)
BITGET_PING_INTERVAL = 30

class BitgetAdapter(ExchangeAdapter):
    def __init__(self):
        super().__init__()
        self._testnet = False
        self._api_key: str = ""
        self._api_secret: str = ""
        self._password: str = ""

    def get_exchange_name(self) -> str: return "bitget"
    def get_display_name(self) -> str: return "Bitget"
    def needs_rest_ticker(self) -> bool: return True

    def init_exchange(self, api_key: str, api_secret: str, testnet: bool = False, password: str = "") -> None:
        self._testnet = testnet
        self._api_key = api_key
        self._api_secret = api_secret
        self._password = password
        options = {
            "apiKey": api_key,
            "secret": api_secret,
            "password": password,
            "options": {"defaultType": "swap"}
        }
        if testnet: options["sandbox"] = True
        self.exchange = ccxt.bitget(options)
        self.exchange.options["defaultType"] = "swap"
        logger.info(f"[Bitget] 交易所初始化完成 (testnet={testnet})")

    def load_markets(self) -> None:
        if not self.exchange: raise RuntimeError("請先呼叫 init_exchange()")
        self.exchange.load_markets(reload=False)
        self._markets_loaded = True

    def get_precision(self, symbol: str) -> PrecisionInfo:
        import math
        if not self._markets_loaded: self.load_markets()
        def _to_decimal_places(value):
            if isinstance(value, float) and 0 < value < 1: return int(abs(math.log10(value)))
            return int(value) if value else 0
        try:
            market = self.exchange.market(symbol)
            precision = market.get("precision", {})
            limits = market.get("limits", {})
            price_prec = _to_decimal_places(precision.get("price", 4))
            amount_prec = _to_decimal_places(precision.get("amount", 0))
            return PrecisionInfo(price_precision=price_prec, amount_precision=amount_prec, min_quantity=float(limits.get("amount", {}).get("min", 0) or 0), min_notional=5.0, tick_size=price_prec, step_size=amount_prec)
        except: return PrecisionInfo(4, 0, 1, 5.0, 4, 0)

    def convert_symbol_to_ccxt(self, raw_symbol: str) -> str:
        raw = raw_symbol.upper().replace("/", "").replace(":", "")
        for quote in ["USDC", "USDT"]:
            if raw.endswith(quote): return f"{raw[:-len(quote)]}/{quote}:{quote}"
        return raw_symbol

    def convert_symbol_to_ws(self, raw_symbol: str) -> str:
        if ":" in raw_symbol: raw_symbol = raw_symbol.split(":")[0]
        return raw_symbol.replace("/", "").replace(":", "").upper()

    def fetch_balance(self) -> Dict[str, BalanceUpdate]:
        result = {}
        try:
            balance = self.exchange.fetch_balance({"type": "swap"})
            for currency in ["USDC", "USDT"]:
                if currency in balance:
                    info = balance[currency]
                    result[currency] = BalanceUpdate(currency, float(info.get("total", 0) or 0), float(info.get("free", 0) or 0))
        except: pass
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # 核心修正：fetch_positions
    # ═══════════════════════════════════════════════════════════════════════════
    def fetch_positions(self) -> List[PositionUpdate]:
        """修正：不跳過 0 倉位，確保主程式能接收到歸零訊號"""
        result = []
        try:
            positions = self.exchange.fetch_positions()
            if not positions:
                return []
            for pos in positions:
                # 取得持倉數量，不論正負或是否為 0
                contracts = float(pos.get("contracts", 0) or pos.get("size", 0) or 0)
                
                # 注意：這裡不再執行 if contracts == 0: continue
                
                side = pos.get("side", "").upper()
                if side not in ["LONG", "SHORT"]: continue

                result.append(PositionUpdate(
                    symbol=pos.get("symbol", ""),
                    position_side=side,
                    quantity=abs(contracts),
                    entry_price=float(pos.get("entryPrice", 0) or 0),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0) or 0),
                    leverage=int(pos.get("leverage", 1) or 1),
                ))
        except Exception as e:
            logger.error(f"[Bitget] 獲取持倉失敗: {e}")
        return result

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, position_side: str = "BOTH", reduce_only: bool = False) -> Dict:
        params = {'hedged': True}
        if reduce_only: params['reduceOnly'] = True
        if position_side == "LONG": params["holdSide"] = "long"
        elif position_side == "SHORT": params["holdSide"] = "short"
        return self.exchange.create_order(symbol, "limit", side.lower(), amount, price, params)

    def create_market_order(self, symbol: str, side: str, amount: float, position_side: str = "BOTH", reduce_only: bool = False) -> Dict:
        params = {'hedged': True}
        if reduce_only: params['reduceOnly'] = True
        if position_side == "LONG": params["holdSide"] = "long"
        elif position_side == "SHORT": params["holdSide"] = "short"
        return self.exchange.create_order(symbol, "market", side.lower(), amount, None, params)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try: self.exchange.cancel_order(order_id, symbol); return True
        except: return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        try: return self.exchange.fetch_open_orders(symbol)
        except: return []

    def fetch_funding_rate(self, symbol: str) -> float:
        try: return float(self.exchange.fetch_funding_rate(symbol).get("fundingRate", 0) or 0)
        except: return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # WebSocket 修正
    # ═══════════════════════════════════════════════════════════════════════════
    def get_websocket_url(self) -> str: return BITGET_WS_MAINNET
    def get_public_websocket_url(self) -> str: return BITGET_WS_PUBLIC_MAINNET

    async def start_user_stream(self) -> Optional[str]:
        if not self._api_key: return None
        ts = str(int(time.time()))
        sign = base64.b64encode(hmac.new(self._api_secret.encode(), f"{ts}GET/user/verify".encode(), hashlib.sha256).digest()).decode()
        return json.dumps({"op": "login", "args": [{"apiKey": self._api_key, "passphrase": self._password, "timestamp": ts, "sign": sign}]})

    def get_keepalive_interval(self) -> int: return BITGET_PING_INTERVAL

    def get_subscription_message(self, symbols: List[str]) -> str:
        args = [{"instType": "USDT-FUTURES", "channel": "orders", "instId": "default"},
                {"instType": "USDT-FUTURES", "channel": "positions", "instId": "default"},
                {"instType": "USDT-FUTURES", "channel": "account", "coin": "default"}]
        return json.dumps({"op": "subscribe", "args": args})

    def parse_ws_message(self, raw_message: str) -> Optional[WSMessage]:
        try:
            data = json.loads(raw_message)
            if "event" in data or data.get("op") == "pong": return None
            arg = data.get("arg", {}); channel = arg.get("channel", ""); payload = data.get("data", [])
            if not payload: return None
            if channel == "orders":
                order = self._parse_order_update(payload[0])
                if order: return WSMessage(WSMessageType.ORDER_UPDATE, order.symbol, order)
            elif channel == "positions":
                account = self._parse_position_update(payload)
                if account: return WSMessage(WSMessageType.ACCOUNT_UPDATE, data=account)
            elif channel == "account":
                account = self._parse_account_data(payload)
                if account: return WSMessage(WSMessageType.ACCOUNT_UPDATE, data=account)
            return None
        except: return None

    def _parse_ticker(self, data: dict) -> Optional[TickerUpdate]:
        try:
            raw = data.get("instId", "")
            if not raw: return None
            return TickerUpdate(self.convert_symbol_to_ccxt(raw), float(data.get("last", 0)), float(data.get("bestBid", 0)), float(data.get("bestAsk", 0)), float(data.get("ts", 0))/1000)
        except: return None

    def _parse_order_update(self, order_data: dict) -> Optional[OrderUpdate]:
        try:
            raw = order_data.get("instId", "")
            if not raw: return None
            status_map = {"live": "NEW", "new": "NEW", "partially_filled": "PARTIALLY_FILLED", "filled": "FILLED", "cancelled": "CANCELED"}
            return OrderUpdate(
                symbol=self.convert_symbol_to_ccxt(raw),
                order_id=str(order_data.get("ordId", "")),
                side=order_data.get("side", "").upper(),
                position_side=order_data.get("posSide", "BOTH").upper(),
                status=status_map.get(order_data.get("status", "").lower(), "UNKNOWN"),
                order_type=order_data.get("ordType", "").upper(),
                quantity=float(order_data.get("sz", 0)),
                filled_quantity=float(order_data.get("fillSz", 0)),
                price=float(order_data.get("px", 0)),
                avg_price=float(order_data.get("avgPx", 0)),
                realized_pnl=float(order_data.get("pnl", 0)),
                commission=abs(float(order_data.get("fee", 0))),
                is_reduce_only=str(order_data.get("reduceOnly", "false")).lower() == "true",
                timestamp=float(order_data.get("uTime", time.time()*1000))/1000,
            )
        except: return None

    # ═══════════════════════════════════════════════════════════════════════════
    # 核心修正：_parse_position_update
    # ═══════════════════════════════════════════════════════════════════════════
    def _parse_position_update(self, positions: list) -> Optional[AccountUpdate]:
        """修正：確保 WebSocket 傳回 0 時也能成功更新"""
        try:
            result = []
            for pos in positions:
                total = float(pos.get("total", 0) or pos.get("available", 0) or 0)
                
                # --- 重要：移除 if total == 0: continue ---
                
                raw = pos.get("instId", "")
                hold_side = pos.get("holdSide", "").upper()
                if hold_side not in ["LONG", "SHORT"]: continue

                result.append(PositionUpdate(
                    symbol=self.convert_symbol_to_ccxt(raw),
                    position_side=hold_side,
                    quantity=abs(total),
                    entry_price=float(pos.get("avgPx", 0) or 0),
                    unrealized_pnl=float(pos.get("upl", 0) or 0),
                    leverage=int(pos.get("lever", 1) or 1),
                ))
            return AccountUpdate(positions=result, balances=[], timestamp=time.time())
        except: return None

    def _parse_account_data(self, account_data: list) -> Optional[AccountUpdate]:
        try:
            balances = []
            for acc in account_data:
                coin = acc.get("coin", "").upper()
                if coin in ["USDC", "USDT"]:
                    balances.append(BalanceUpdate(coin, float(acc.get("equity", 0) or 0), float(acc.get("available", 0) or 0)))
            return AccountUpdate(positions=[], balances=balances, timestamp=time.time())
        except: return None
