# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Bitget Adapter (Final Robust Version)
==============
1. 補齊所有抽象方法，解決 "Can't instantiate abstract class" 錯誤。
2. 針對 22002 錯誤進行攔截，強迫主程式更新本地數據，解決 BEAT/ALLO 記憶死鎖。
3. 強化 fetch_positions，確保交易所實體持倉優先於本地快取。
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

class BitgetAdapter(ExchangeAdapter):
    def __init__(self):
        super().__init__()
        self._testnet = False
        self._api_key = ""
        self._api_secret = ""
        self._password = ""

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

    def load_markets(self) -> None:
        if self.exchange:
            self.exchange.load_markets(reload=False)
            self._markets_loaded = True

    def get_precision(self, symbol: str) -> PrecisionInfo:
        import math
        if not self._markets_loaded: self.load_markets()
        def _to_dp(v): return int(abs(math.log10(v))) if isinstance(v, float) and 0 < v < 1 else int(v) if v else 0
        try:
            m = self.exchange.market(symbol)
            p, l = m.get("precision", {}), m.get("limits", {})
            pp, ap = _to_dp(p.get("price", 4)), _to_dp(p.get("amount", 0))
            return PrecisionInfo(pp, ap, float(l.get("amount", {}).get("min", 0)), 5.0, pp, ap)
        except: return PrecisionInfo(4, 0, 1, 5.0, 4, 0)

    def convert_symbol_to_ccxt(self, s: str) -> str:
        s = s.upper().replace("/", "").replace(":", "")
        for q in ["USDC", "USDT"]:
            if s.endswith(q): return f"{s[:-len(q)]}/{q}:{q}"
        return s

    def convert_symbol_to_ws(self, s: str) -> str:
        return s.split(":")[0].replace("/", "").upper() if ":" in s else s.replace("/", "").upper()

    def fetch_balance(self) -> Dict[str, BalanceUpdate]:
        res = {}
        try:
            b = self.exchange.fetch_balance({"type": "swap"})
            for c in ["USDC", "USDT"]:
                if c in b: res[c] = BalanceUpdate(c, float(b[c].get("total", 0)), float(b[c].get("free", 0)))
        except: pass
        return res

    def fetch_positions(self) -> List[PositionUpdate]:
        """強制同步：只申報交易所真正存在的持倉"""
        res = []
        try:
            ps = self.exchange.fetch_positions()
            if not ps: return []
            for p in ps:
                qty = abs(float(p.get("contracts", 0) or p.get("size", 0) or 0))
                if qty > 0.000001:
                    res.append(PositionUpdate(
                        p.get("symbol", ""), p.get("side", "").upper(), qty,
                        float(p.get("entryPrice", 0)), float(p.get("unrealizedPnl", 0)), int(p.get("leverage", 1))
                    ))
            return res
        except: return []

    def set_leverage(self, symbol: str, leverage: int, params: dict = {}) -> bool:
        """補齊抽象方法：設定槓桿"""
        try: self.exchange.set_leverage(leverage, symbol, params); return True
        except: return False

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, position_side: str = "BOTH", reduce_only: bool = False) -> Dict:
        p = {'hedged': True}
        if reduce_only: p['reduceOnly'] = True
        if position_side == "LONG": p["holdSide"] = "long"
        elif position_side == "SHORT": p["holdSide"] = "short"
        try:
            return self.exchange.create_order(symbol, "limit", side.lower(), amount, price, p)
        except Exception as e:
            if "22002" in str(e): # 處理無倉位可平的錯誤，強制同步
                return {"id": "fake_sync", "status": "closed"}
            raise e

    def create_market_order(self, symbol: str, side: str, amount: float, position_side: str = "BOTH", reduce_only: bool = False) -> Dict:
        p = {'hedged': True}
        if reduce_only: p['reduceOnly'] = True
        if position_side == "LONG": p["holdSide"] = "long"
        elif position_side == "SHORT": p["holdSide"] = "short"
        try:
            return self.exchange.create_order(symbol, "market", side.lower(), amount, None, p)
        except Exception as e:
            if "22002" in str(e):
                return {"id": "fake_sync", "status": "closed"}
            raise e

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try: self.exchange.cancel_order(order_id, symbol); return True
        except: return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        try: return self.exchange.fetch_open_orders(symbol)
        except: return []

    def fetch_funding_rate(self, symbol: str) -> float:
        try: return float(self.exchange.fetch_funding_rate(symbol).get("fundingRate", 0))
        except: return 0.0

    def get_websocket_url(self) -> str: return "wss://ws.bitget.com/v2/ws/private"
    def get_public_websocket_url(self) -> str: return "wss://ws.bitget.com/v2/ws/public"
    
    def build_stream_url(self, symbols: List[str], user_stream_key: Optional[str] = None) -> str:
        """補齊抽象方法"""
        return self.get_websocket_url()

    async def keepalive_user_stream(self) -> None:
        """補齊抽象方法"""
        pass

    def get_keepalive_interval(self) -> int: return 30

    async def start_user_stream(self) -> Optional[str]:
        if not self._api_key: return None
        ts = str(int(time.time()))
        sign = base64.b64encode(hmac.new(self._api_secret.encode(), f"{ts}GET/user/verify".encode(), hashlib.sha256).digest()).decode()
        return json.dumps({"op": "login", "args": [{"apiKey": self._api_key, "passphrase": self._password, "timestamp": ts, "sign": sign}]})

    def get_subscription_message(self, symbols: List[str]) -> str:
        args = [{"instType": "USDT-FUTURES", "channel": "orders", "instId": "default"},
                {"instType": "USDT-FUTURES", "channel": "positions", "instId": "default"},
                {"instType": "USDT-FUTURES", "channel": "account", "coin": "default"}]
        return json.dumps({"op": "subscribe", "args": args})

    def parse_ws_message(self, raw: str) -> Optional[WSMessage]:
        try:
            d = json.loads(raw)
            if "event" in d or d.get("op") == "pong": return None
            arg = d.get("arg", {}); ch = arg.get("channel", ""); py = d.get("data", [])
            if not py: return None
            if ch == "orders":
                o = self._parse_order_update(py[0])
                if o: return WSMessage(WSMessageType.ORDER_UPDATE, o.symbol, o)
            elif ch == "positions":
                a = self._p_pos(py)
                if a: return WSMessage(WSMessageType.ACCOUNT_UPDATE, data=a)
            elif ch == "account":
                a = self._p_acc(py)
                if a: return WSMessage(WSMessageType.ACCOUNT_UPDATE, data=a)
            return None
        except: return None

    def _parse_order_update(self, o: dict) -> Optional[OrderUpdate]:
        try:
            m = {"live": "NEW", "new": "NEW", "filled": "FILLED", "cancelled": "CANCELED"}
            return OrderUpdate(self.convert_symbol_to_ccxt(o.get("instId", "")), str(o.get("ordId", "")), o.get("side", "").upper(), o.get("posSide", "BOTH").upper(), m.get(o.get("status", "").lower(), "UNKNOWN"), o.get("ordType", "").upper(), float(o.get("sz", 0)), float(o.get("fillSz", 0)), float(o.get("px", 0)), float(o.get("avgPx", 0)), float(o.get("pnl", 0)), abs(float(o.get("fee", 0))), str(o.get("reduceOnly", "")).lower() == "true", float(o.get("uTime", time.time()*1000))/1000)
        except: return None

    def _p_pos(self, ps: list) -> Optional[AccountUpdate]:
        try:
            res = []
            for p in ps:
                res.append(PositionUpdate(self.convert_symbol_to_ccxt(p.get("instId", "")), p.get("holdSide", "").upper(), abs(float(p.get("total", 0))), float(p.get("avgPx", 0)), float(p.get("upl", 0)), int(p.get("lever", 1))))
            return AccountUpdate(res, [], time.time())
        except: return None

    def _p_acc(self, ac: list) -> Optional[AccountUpdate]:
        try:
            bl = []
            for a in ac:
                if a.get("coin", "").upper() in ["USDC", "USDT"]:
                    bl.append(BalanceUpdate(a.get("coin", "").upper(), float(a.get("equity", 0)), float(a.get("available", 0))))
            return AccountUpdate([], bl, time.time())
        except: return None
