# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Bitget Adapter (Final Robust Version)
==============
1. 補齊所有抽象方法，解決啟動失敗問題。
2. 優化持倉同步邏輯，確保交易所 0 倉位能正確回報。
3. 修正 set_leverage 參數簽名，符合主程式規範。
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
BITGET_PING_INTERVAL = 30

class BitgetAdapter(ExchangeAdapter):
    def __init__(self):
        super().__init__()
        self._testnet = False
        self._api_key: str = ""
        self._api_secret: str = ""
        self._password: str = ""

    def get_exchange_name(self) -> str:
        return "bitget"

    def get_display_name(self) -> str:
        return "Bitget"
    
    def needs_rest_ticker(self) -> bool:
        return True

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
        if testnet:
            options["sandbox"] = True
        self.exchange = ccxt.bitget(options)
        self.exchange.options["defaultType"] = "swap"
        logger.info(f"[Bitget] 交易所初始化完成")

    def load_markets(self) -> None:
        if not self.exchange:
            return
        self.exchange.load_markets(reload=False)
        self._markets_loaded = True

    def get_precision(self, symbol: str) -> PrecisionInfo:
        import math
        if not self._markets_loaded:
            self.load_markets()
        def _to_decimal_places(value):
            if isinstance(value, float) and 0 < value < 1:
                return int(abs(math.log10(value)))
            return int(value) if value else 0
        try:
            market = self.exchange.market(symbol)
            precision = market.get("precision", {})
            limits = market.get("limits", {})
            p_prec = _to_decimal_places(precision.get("price", 4))
            a_prec = _to_decimal_places(precision.get("amount", 0))
            return PrecisionInfo(p_prec, a_prec, float(limits.get("amount", {}).get("min", 0) or 0), 5.0, p_prec, a_prec)
        except:
            return PrecisionInfo(4, 0, 1, 5.0, 4, 0)

    def convert_symbol_to_ccxt(self, raw_symbol: str) -> str:
        raw = raw_symbol.upper().replace("/", "").replace(":", "")
        for quote in ["USDC", "USDT"]:
            if raw.endswith(quote):
                return f"{raw[:-len(quote)]}/{quote}:{quote}"
        return raw_symbol

    def convert_symbol_to_ws(self, raw_symbol: str) -> str:
        if ":" in raw_symbol:
            raw_symbol = raw_symbol.split(":")[0]
        return raw_symbol.replace("/", "").replace(":", "").upper()

    def fetch_balance(self) -> Dict[str, BalanceUpdate]:
        result = {}
        try:
            balance = self.exchange.fetch_balance({"type": "swap"})
            for currency in ["USDC", "USDT"]:
                if currency in balance:
                    info = balance[currency]
                    result[currency] = BalanceUpdate(currency, float(info.get("total", 0) or 0), float(info.get("free", 0) or 0))
        except:
            pass
        return result

    def fetch_positions(self) -> List[PositionUpdate]:
        """自動修復版：確保消失的倉位也能被正確感知"""
        result = []
        try:
            positions = self.exchange.fetch_positions()
            if not positions:
                return []
            for pos in positions:
                contracts = float(pos.get("contracts", 0) or pos.get("size", 0) or 0)
                side = pos.get("side", "").upper()
                if side not in ["LONG", "SHORT"]:
                    continue
                result.append(PositionUpdate(
                    symbol=pos.get("symbol", ""),
                    position_side=side,
                    quantity=abs(contracts),
                    entry_price=float(pos.get("entryPrice", 0) or 0),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0) or 0),
                    leverage=int(pos.get("leverage", 1) or 1),
                ))
            return result
        except Exception as e:
            logger.error(f"[Bitget] 獲取持倉失敗: {e}")
            return []

    def set_leverage(self, symbol: str, leverage: int, params: dict = {}) -> bool:
        """實作 set_leverage 並符合主程式參數規範"""
        try:
            self.exchange.set_leverage(leverage, symbol, params)
            return True
        except:
            return False

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
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except:
            return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            return self.exchange.fetch_open_orders(symbol)
        except:
            return []

    def fetch_funding_rate(self, symbol: str) -> float:
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return float(funding.get("fundingRate", 0) or 0)
        except:
            return 0.0

    def get_websocket_url(self) -> str:
        return BITGET_WS_MAINNET

    def get_public_websocket_url(self) -> str:
        return BITGET_WS_PUBLIC_MAINNET

    def build_stream_url(self, symbols: List[str], user_stream_key: Optional[str] = None) -> str:
        return self.get_websocket_url()

    async def keepalive_user_stream(self) -> None:
        """保持 WebSocket 連線活躍"""
        pass

    async def start_user_stream(self) -> Optional[str]:
        if not self._api_key: return None
        ts = str(int(time.time()))
        message = f"{ts}GET/user/verify"
        signature = base64.b64encode(hmac.new(self._api_secret.encode(), message.encode(), hashlib.sha256).digest()).decode()
        return json.dumps({"op": "login", "args": [{"apiKey": self._api_key, "passphrase": self._password, "timestamp": ts, "sign": signature}]})

    def get_keepalive_interval(self) -> int:
        return BITGET_PING_INTERVAL

    def get_subscription_message(self, symbols: List[str]) -> str:
        args = [
            {"instType": "USDT-FUTURES", "channel": "orders", "instId": "default"},
            {"instType": "USDT-FUTURES", "channel": "positions", "instId": "default"},
            {"instType": "USDT-FUTURES", "channel": "account", "coin": "default"}
        ]
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
        except:
            return None

    def _parse_order_update(self, o: dict) -> Optional[OrderUpdate]:
        try:
            raw = o.get("instId", "")
            sm = {"live": "NEW", "new": "NEW", "partially_filled": "PARTIALLY_FILLED", "filled": "FILLED", "cancelled": "CANCELED"}
            return OrderUpdate(
                symbol=self.convert_symbol_to_ccxt(raw),
                order_id=str(o.get("ordId", "")),
                side=o.get("side", "").upper(),
                position_side=o.get("posSide", "BOTH").upper(),
                status=sm.get(o.get("status", "").lower(), "UNKNOWN"),
                order_type=o.get("ordType", "").upper(),
                quantity=float(o.get("sz", 0)),
                filled_quantity=float(o.get("fillSz", 0)),
                price=float(o.get("px", 0)),
                avg_price=float(o.get("avgPx", 0)),
                realized_pnl=float(o.get("pnl", 0)),
                commission=abs(float(o.get("fee", 0))),
                is_reduce_only=str(o.get("reduceOnly", "false")).lower() == "true",
                timestamp=float(o.get("uTime", time.time()*1000))/1000,
            )
        except:
            return None

    def _parse_position_update(self, positions: list) -> Optional[AccountUpdate]:
        try:
            result = []
            for p in positions:
                total = float(p.get("total", 0) or p.get("available", 0) or 0)
                raw = p.get("instId", "")
                hold_side = p.get("holdSide", "").upper()
                if hold_side not in ["LONG", "SHORT"]: continue
                result.append(PositionUpdate(self.convert_symbol_to_ccxt(raw), hold_side, abs(total), float(p.get("avgPx", 0)), float(p.get("upl", 0)), int(p.get("lever", 1))))
            return AccountUpdate(result, [], time.time())
        except:
            return None

    def _parse_account_data(self, account_data: list) -> Optional[AccountUpdate]:
        try:
            balances = []
            for a in account_data:
                currency = a.get("coin", "").upper()
                if currency in ["USDC", "USDT"]:
                    balances.append(BalanceUpdate(currency, float(a.get("equity", 0)), float(a.get("available", 0))))
            return AccountUpdate([], balances, time.time())
        except:
            return None
