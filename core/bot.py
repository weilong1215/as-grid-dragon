# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
MaxGridBot - 主交易機器人
========================
WebSocket 連接、交易執行、狀態管理

架構:
    GridBot 使用 ExchangeAdapter 抽象層與交易所互動，
    支援任何 CCXT 兼容的交易所。
"""

import json
import time
import asyncio
import logging
import ssl
import certifi
from datetime import datetime
from typing import Dict, Optional

import websockets

from .strategy import GridStrategy
from config.models import GlobalConfig, GlobalState, SymbolState
from utils import safe_float
from indicators.bandit import UCBBanditOptimizer
from indicators.leading import LeadingIndicatorManager
from indicators.funding import FundingRateManager, GLFTController
from indicators.dgt import DGTBoundaryManager, DynamicGridManager

# 多交易所支援
from exchanges import get_adapter, ExchangeAdapter
from exchanges.base import WSMessageType

logger = logging.getLogger("as_grid_max")


def normalize_symbol(symbol: str) -> str:
    """
    標準化交易對符號，處理各交易所不同格式

    Examples:
        XRPUSDT -> XRPUSDT
        XRP_USDT -> XRPUSDT
        XRP/USDT:USDT -> XRPUSDT
        xrpusdt -> XRPUSDT
    """
    # 移除分隔符並轉大寫
    s = symbol.upper().replace("_", "").replace("/", "").replace(":", "")

    # 處理 CCXT 格式中重複的結算幣種 (如 XRPUSDTUSDT -> XRPUSDT)
    for quote in ["USDT", "USDC", "BUSD"]:
        if s.endswith(quote + quote):
            s = s[:-len(quote)]
            break

    return s


class MaxGridBot:
    """MAX 網格交易機器人"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.state = GlobalState()

        # === 多交易所支援 ===
        exchange_type = getattr(config, 'exchange_type', 'binance')
        self.adapter: ExchangeAdapter = get_adapter(exchange_type)

        # === 舊版兼容: 保留 self.exchange 引用 ===
        self.exchange = None  # 將在 _init_exchange() 中設定

        self.ws = None
        self._stop_event = asyncio.Event()
        self.precision_info: Dict[str, Dict] = {}
        self.bandit_optimizer = UCBBanditOptimizer(config.bandit)
        self.leading_indicator = LeadingIndicatorManager(config.leading_indicator)
        self.funding_manager: Optional[FundingRateManager] = None
        self.glft_controller = GLFTController()
        self.dgt_manager = DGTBoundaryManager(config.dgt)
        self.dynamic_grid = DynamicGridManager()
        self.last_grid_time: Dict[str, float] = {}
        self.grid_interval = 0.5
        # User Data Stream
        self.listen_key: Optional[str] = None
        self.listen_key_time: float = 0

    def _init_exchange(self):
        """初始化交易所 (使用 Adapter 抽象層)"""
        # 使用 Adapter 初始化
        testnet = getattr(self.config, 'testnet', False)
        password = getattr(self.config, 'api_password', '')  # Bitget 需要 passphrase
        self.adapter.init_exchange(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            testnet=testnet,
            password=password  # 傳遞 Bitget passphrase
        )
        self.adapter.load_markets()

        # === 舊版兼容: 保留 self.exchange 引用 ===
        self.exchange = self.adapter.exchange

        # 載入精度資訊
        for sym_cfg in self.config.symbols.values():
            if not sym_cfg.enabled:
                continue
            try:
                precision = self.adapter.get_precision(sym_cfg.ccxt_symbol)
                self.precision_info[sym_cfg.ccxt_symbol] = {
                    "price": int(precision.price_precision),
                    "amount": int(precision.amount_precision),
                    "min_notional": precision.min_notional
                }
            except Exception as e:
                logger.error(f"獲取 {sym_cfg.ccxt_symbol} 精度失敗: {e}")

        # 檢查並啟用對沖模式 (與 Terminal 版一致)
        self._check_hedge_mode()

        self.funding_manager = FundingRateManager(self.exchange)
        logger.info(f"[Bot] {self.adapter.get_display_name()} 初始化完成，{len(self.precision_info)} 個交易對")

    def _check_hedge_mode(self):
        """
        檢查並啟用對沖模式 (與 Terminal 版一致)
        
        Binance 需要啟用 Hedge Mode (雙向持倉) 才能同時持有多空倉位
        """
        for sym_config in self.config.symbols.values():
            if sym_config.enabled:
                try:
                    mode = self.exchange.fetch_position_mode(symbol=sym_config.ccxt_symbol)
                    if not mode.get('hedged', False):
                        print(f"[Bot] {sym_config.symbol} 啟用對沖模式...")
                        self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'true'})
                        print(f"[Bot] 對沖模式已啟用")
                        break
                except Exception as e:
                    logger.warning(f"[Bot] 檢查對沖模式失敗: {e}")

    def _init_state(self):
        for sym_cfg in self.config.symbols.values():
            if sym_cfg.enabled:
                self.state.symbols[sym_cfg.ccxt_symbol] = SymbolState(symbol=sym_cfg.symbol)
        self.state.start_time = datetime.now()
        self.state.running = True

    async def _get_listen_key(self) -> Optional[str]:
        """獲取 User Data Stream Key (使用 Adapter)"""
        self.listen_key = await self.adapter.start_user_stream()
        if self.listen_key:
            self.listen_key_time = time.time()
        return self.listen_key

    async def _keepalive_listen_key(self):
        """定期延長 User Data Stream (使用 Adapter)"""
        interval = self.adapter.get_keepalive_interval()
        if interval <= 0:
            return  # 此交易所不需要 keepalive

        while not self._stop_event.is_set():
            await asyncio.sleep(interval)
            try:
                await self.adapter.keepalive_user_stream()
                self.listen_key_time = time.time()
            except Exception as e:
                logger.error(f"[Bot] Keepalive 失敗: {e}")
                await self._get_listen_key()

    async def _sync_positions(self):
        try:
            # 使用 Adapter 獲取標準化的持倉資料
            positions = self.adapter.fetch_positions()
            for pos in positions:
                # 使用標準化 symbol 匹配
                normalized_sym = normalize_symbol(pos.symbol)
                ccxt_symbol = None
                for cfg in self.config.symbols.values():
                    cfg_normalized = normalize_symbol(cfg.symbol)
                    if cfg_normalized == normalized_sym:
                        ccxt_symbol = cfg.ccxt_symbol
                        break
                if not ccxt_symbol or ccxt_symbol not in self.state.symbols:
                    continue

                side = pos.position_side.lower()  # LONG 或 SHORT
                if side == "long":
                    self.state.symbols[ccxt_symbol].long_position = pos.quantity
                elif side == "short":
                    self.state.symbols[ccxt_symbol].short_position = pos.quantity
                self.state.symbols[ccxt_symbol].unrealized_pnl = pos.unrealized_pnl

            # 使用 Adapter 獲取標準化的餘額資料
            balances = self.adapter.fetch_balance()
            for currency in ["USDC", "USDT"]:
                if currency in balances:
                    bal = balances[currency]
                    acc = self.state.get_account(currency)
                    acc.wallet_balance = bal.wallet_balance
                    acc.available_balance = bal.available_balance
                    # 計算保證金使用
                    acc.margin_used = acc.wallet_balance - acc.available_balance
            self.state.update_totals()
        except Exception as e:
            logger.error(f"同步倉位失敗: {e}")

    async def _sync_orders(self):
        """
        同步掛單狀態 (與 Terminal 版一致)
        
        追蹤每個交易對的掛單數量:
        - buy_long_orders: 多頭補倉單數量
        - sell_long_orders: 多頭止盈單數量
        - buy_short_orders: 空頭止盈單數量
        - sell_short_orders: 空頭補倉單數量
        """
        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue
            ccxt_symbol = sym_config.ccxt_symbol

            try:
                orders = self.adapter.fetch_open_orders(ccxt_symbol)
                state = self.state.symbols.get(ccxt_symbol)
                if not state:
                    continue

                # 重置掛單計數
                state.buy_long_orders = 0
                state.sell_long_orders = 0
                state.buy_short_orders = 0
                state.sell_short_orders = 0

                for order in orders:
                    qty = abs(float(order.get('amount', 0) or order.get('info', {}).get('origQty', 0)))
                    side = order.get('side', '').lower()
                    pos_side = order.get('info', {}).get('positionSide', 'BOTH').upper()

                    if side == 'buy' and pos_side == 'LONG':
                        state.buy_long_orders += qty
                    elif side == 'sell' and pos_side == 'LONG':
                        state.sell_long_orders += qty
                    elif side == 'buy' and pos_side == 'SHORT':
                        state.buy_short_orders += qty
                    elif side == 'sell' and pos_side == 'SHORT':
                        state.sell_short_orders += qty

            except Exception as e:
                logger.error(f"同步 {ccxt_symbol} 掛單失敗: {e}")

    async def run(self):
        try:
            print("[Bot] 開始初始化交易所...")
            self._init_exchange()
            print("[Bot] 交易所初始化完成")

            self._init_state()
            print("[Bot] 狀態初始化完成，state.running = True")

            await self._sync_positions()
            print("[Bot] 倉位同步完成")

            for cfg in self.config.symbols.values():
                if cfg.enabled:
                    try:
                        self.adapter.set_leverage(cfg.ccxt_symbol, cfg.leverage)
                    except Exception as e:
                        print(f"[Bot] 設置 {cfg.symbol} 槓桿失敗: {e}")

            # 掛初始網格
            await self._place_initial_grids()
            print("[Bot] 啟動 WebSocket...")
            await self._websocket_loop()
        except Exception as e:
            print(f"[Bot] run() 執行失敗: {type(e).__name__}: {e}")
            raise  # 重新拋出讓外層捕獲

    async def _place_initial_grids(self):
        """啟動時掛初始網格"""
        for cfg in self.config.symbols.values():
            if cfg.enabled:
                await self._place_grid(cfg)
                logger.info(f"[Bot] {cfg.symbol} 初始網格已掛")
                await asyncio.sleep(0.2)  # 避免 API 限流

    async def stop(self):
        self._stop_event.set()
        self.state.running = False
        if self.ws:
            await self.ws.close()

    async def _websocket_loop(self):
        """WebSocket 主循環 (使用 Adapter 構建 URL)"""
        # 收集啟用的交易對
        symbols = []
        for sym_cfg in self.config.symbols.values():
            if sym_cfg.enabled:
                symbols.append(sym_cfg.symbol)

        if not symbols:
            logger.error("[Bot] 沒有啟用的交易對")
            return

        # 獲取 Listen Key
        await self._get_listen_key()

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        # 啟動 keepalive
        asyncio.create_task(self._keepalive_listen_key())

        while not self._stop_event.is_set():
            try:
                # 使用 Adapter 構建 URL
                url = self.adapter.build_stream_url(symbols, self.listen_key)

                print(f"[Bot] 正在連接 WebSocket: {url[:80]}...")
                async with websockets.connect(url, ssl=ssl_context, ping_interval=30, ping_timeout=10) as ws:
                    self.ws = ws
                    self.state.connected = True
                    print(f"[Bot] WebSocket 已連接 ({self.adapter.get_display_name()})")
                    asyncio.create_task(self._sync_loop())
                    msg_count = 0
                    async for message in ws:
                        if self._stop_event.is_set():
                            break
                        msg_count += 1
                        await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("[Bot] WebSocket 連接關閉，重連中...")
            except Exception as e:
                logger.error(f"[Bot] WebSocket 錯誤: {e}")
            self.state.connected = False
            await asyncio.sleep(5)

    async def _sync_loop(self):
        """定期同步循環 (與 Terminal 版 sync_all 一致)"""
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.sync_interval)
            await self._sync_positions()
            await self._sync_orders()
            # 風控檢查
            await self._risk_monitor_loop()

    async def _handle_message(self, message: str):
        """處理 WebSocket 消息 (使用 Adapter 解析)"""
        try:
            ws_msg = self.adapter.parse_ws_message(message)
            if not ws_msg:
                return

            if ws_msg.msg_type == WSMessageType.TICKER:
                await self._handle_ticker_update(ws_msg.symbol, ws_msg.data)
            elif ws_msg.msg_type == WSMessageType.ORDER_UPDATE:
                await self._handle_order_update(ws_msg.data)
            elif ws_msg.msg_type == WSMessageType.ACCOUNT_UPDATE:
                await self._handle_account_update(ws_msg.data)

        except Exception as e:
            logger.error(f"處理訊息錯誤: {e}")

    async def _handle_order_update(self, order_update):
        """
        處理訂單更新 (標準化 OrderUpdate 格式)

        Args:
            order_update: exchanges.base.OrderUpdate 實例
        """
        # 只處理完全成交
        if order_update.status != "FILLED":
            return

        symbol = order_update.symbol  # 原始交易對 (如 XRPUSDC 或 XRP_USDT)
        side = order_update.side      # BUY 或 SELL
        qty = order_update.filled_quantity
        price = order_update.avg_price
        realized_pnl = order_update.realized_pnl
        is_reduce = order_update.is_reduce_only

        logger.info(f"[Bot] 訂單成交: {symbol} {side} {qty}@{price} PnL={realized_pnl:.4f}")

        # 找到對應的 ccxt_symbol (處理不同交易所的 symbol 格式)
        ccxt_symbol = None
        normalized_symbol = normalize_symbol(symbol)
        for cfg in self.config.symbols.values():
            cfg_normalized = normalize_symbol(cfg.symbol)
            if cfg_normalized == normalized_symbol:
                ccxt_symbol = cfg.ccxt_symbol
                break

        if not ccxt_symbol:
            return

        sym_state = self.state.symbols.get(ccxt_symbol)
        if not sym_state:
            return

        # 更新交易統計
        sym_state.total_trades += 1
        sym_state.total_profit += realized_pnl
        self.state.total_trades += 1
        self.state.total_profit += realized_pnl

        # 記錄最近交易
        sym_state.recent_trades.append({
            "time": datetime.now().isoformat(),
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": realized_pnl
        })

        # === 關鍵: 呼叫 Bandit 記錄交易 ===
        if self.config.bandit.enabled and is_reduce and realized_pnl != 0:
            trade_side = "long" if side == "SELL" else "short"  # 減倉方向相反
            self.bandit_optimizer.record_trade(realized_pnl, trade_side)
            logger.debug(f"[Bandit] 記錄交易: {trade_side} pnl={realized_pnl:.4f}")

        # === 關鍵: 呼叫領先指標記錄 ===
        if self.config.leading_indicator.enabled:
            trade_side = "buy" if side == "BUY" else "sell"
            self.leading_indicator.record_trade(ccxt_symbol, price, qty, trade_side)

    async def _handle_account_update(self, account_update):
        """
        處理帳戶更新 (標準化 AccountUpdate 格式)

        Args:
            account_update: exchanges.base.AccountUpdate 實例
        """
        # 更新持倉
        for pos in account_update.positions:
            # 找到對應的 ccxt_symbol (處理不同交易所的 symbol 格式)
            ccxt_symbol = None
            normalized_symbol = pos.symbol.upper().replace("_", "").replace("/", "").replace(":", "")
            for cfg in self.config.symbols.values():
                cfg_normalized = normalize_symbol(cfg.symbol)
                if cfg_normalized == normalized_symbol:
                    ccxt_symbol = cfg.ccxt_symbol
                    break
            if not ccxt_symbol or ccxt_symbol not in self.state.symbols:
                continue

            sym_state = self.state.symbols[ccxt_symbol]
            side = pos.position_side.lower()  # LONG 或 SHORT

            if side == "long":
                sym_state.long_position = pos.quantity
            else:
                sym_state.short_position = pos.quantity
            sym_state.unrealized_pnl = pos.unrealized_pnl

        # 更新餘額
        for bal in account_update.balances:
            if bal.currency in ["USDC", "USDT"]:
                acc = self.state.get_account(bal.currency)
                acc.wallet_balance = bal.wallet_balance
                acc.available_balance = bal.available_balance
                acc.margin_used = acc.wallet_balance - acc.available_balance

        self.state.update_totals()

    async def _handle_ticker_update(self, raw_symbol: str, ticker_update):
        """
        處理 Ticker 更新 (標準化 TickerUpdate 格式)

        Args:
            raw_symbol: 原始交易對符號 (如 XRPUSDC 或 XRP_USDT)
            ticker_update: exchanges.base.TickerUpdate 實例
        """
        # 標準化 symbol 進行比對
        normalized_raw = normalize_symbol(raw_symbol)
        matched = False
        for cfg in self.config.symbols.values():
            cfg_normalized = normalize_symbol(cfg.symbol)
            
            if cfg_normalized == normalized_raw and cfg.ccxt_symbol in self.state.symbols:
                matched = True
                sym_state = self.state.symbols[cfg.ccxt_symbol]
                
                # 更新價格信息
                old_price = sym_state.latest_price
                sym_state.latest_price = ticker_update.price
                sym_state.best_bid = ticker_update.bid
                sym_state.best_ask = ticker_update.ask


                # 更新領先指標
                self.leading_indicator.update_spread(cfg.ccxt_symbol, sym_state.best_bid, sym_state.best_ask)
                self.dynamic_grid.update_price(cfg.ccxt_symbol, sym_state.latest_price)

                if self.config.bandit.contextual_enabled:
                    self.bandit_optimizer.update_price(sym_state.latest_price)

                # 總是嘗試執行網格交易，而不僅僅是間隔控制
                # 這樣可以確保價格變化時立即響應
                await self._place_grid(cfg)
                
                # 如果價格變動較大，也更新時間戳以避免過於頻繁的網格操作
                now = time.time()
                last = self.last_grid_time.get(cfg.ccxt_symbol, 0)
                if now - last >= self.grid_interval:
                    self.last_grid_time[cfg.ccxt_symbol] = now
                break


    async def _place_grid(self, cfg):
        """
        掛出網格訂單 (與 Terminal 版一致)
        
        邏輯:
        - 持倉為 0 時: 在 best_bid/best_ask 下單開倉
        - 有持倉時: 下止盈單和補倉單
        """
        ccxt_sym = cfg.ccxt_symbol
        sym_state = self.state.symbols.get(ccxt_sym)
        if not sym_state:
            logger.warning(f"[Bot] 無法找到 {ccxt_sym} 的狀態")
            return
        
        # 使用當前價格，如果價格未更新，則跳過本次網格操作
        price = sym_state.latest_price
        if price <= 0:
            logger.debug(f"[Bot] {cfg.symbol} 價格未更新 ({price})，跳過本次網格操作")
            return
        precision = self.precision_info.get(ccxt_sym, {"price": 4, "amount": 0, "min_notional": 5})
        min_notional = precision.get("min_notional", 5) * 1.1  # +10% 緩衝，避免邊界問題
        
        # 動態計算滿足 min_notional 的最小數量
        import math
        min_qty_for_notional = min_notional / price if price > 0 else 1
        # 向上取整到精度
        amount_prec = precision.get("amount", 0)
        if amount_prec > 0:
            factor = 10 ** amount_prec
            base_qty = math.ceil(max(cfg.initial_quantity, min_qty_for_notional) * factor) / factor
        else:
            base_qty = math.ceil(max(cfg.initial_quantity, min_qty_for_notional))
        
        long_pos = sym_state.long_position
        short_pos = sym_state.short_position
        
        # 計算實際名義價值
        notional = base_qty * price
        print(f"[Place] {cfg.symbol} 價格={price:.6f}, 多倉={long_pos}, 空倉={short_pos}, 數量={base_qty}, 名義價值={notional:.2f}U")
        
        tp_spacing, gs_spacing = cfg.take_profit_spacing, cfg.grid_spacing
        if self.config.bandit.enabled:
            params = self.bandit_optimizer.get_current_params()
            tp_spacing = params.take_profit_spacing
            gs_spacing = params.grid_spacing
        if self.config.leading_indicator.enabled:
            adjusted, reason = self.leading_indicator.get_spacing_adjustment(ccxt_sym, gs_spacing)
            gs_spacing = adjusted

        # 計算策略決策
        long_decision = GridStrategy.get_grid_decision(
            price=price, my_position=long_pos, opposite_position=short_pos,
            position_threshold=cfg.position_threshold, position_limit=cfg.position_limit,
            base_qty=base_qty, take_profit_spacing=tp_spacing, grid_spacing=gs_spacing, side='long')
        short_decision = GridStrategy.get_grid_decision(
            price=price, my_position=short_pos, opposite_position=long_pos,
            position_threshold=cfg.position_threshold, position_limit=cfg.position_limit,
            base_qty=base_qty, take_profit_spacing=tp_spacing, grid_spacing=gs_spacing, side='short')
        sym_state.long_dead_mode = long_decision['dead_mode']
        sym_state.short_dead_mode = short_decision['dead_mode']

        try:
            # === 多頭處理 ===
            await self.cancel_orders_for_side(ccxt_sym, 'long')
            
            if long_pos > 0:
                # 有持倉: 下止盈單
                tp_price = round(long_decision['tp_price'], precision['price'])
                tp_qty = round(min(long_decision['tp_qty'], long_pos), precision['amount'])
                if tp_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'sell', tp_qty, tp_price,
                        position_side='LONG', reduce_only=True
                    )
            
            # 補倉單 (或無倉時的開倉單)
            if not long_decision['dead_mode'] and long_decision['entry_price']:
                entry_price = round(long_decision['entry_price'], precision['price'])
                entry_qty = round(long_decision['entry_qty'], precision['amount'])
                # 無倉時使用 best_bid，有倉時使用計算的 entry_price
                if long_pos == 0 and sym_state.best_bid > 0:
                    entry_price = round(sym_state.best_bid, precision['price'])
                if entry_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'buy', entry_qty, entry_price,
                        position_side='LONG', reduce_only=False
                    )

            # === 空頭處理 ===
            await self.cancel_orders_for_side(ccxt_sym, 'short')
            
            if short_pos > 0:
                # 有持倉: 下止盈單
                tp_price = round(short_decision['tp_price'], precision['price'])
                tp_qty = round(min(short_decision['tp_qty'], short_pos), precision['amount'])
                if tp_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'buy', tp_qty, tp_price,
                        position_side='SHORT', reduce_only=True
                    )
            
            # 補倉單 (或無倉時的開倉單)
            if not short_decision['dead_mode'] and short_decision['entry_price']:
                entry_price = round(short_decision['entry_price'], precision['price'])
                entry_qty = round(short_decision['entry_qty'], precision['amount'])
                # 無倉時使用 best_ask，有倉時使用計算的 entry_price
                if short_pos == 0 and sym_state.best_ask > 0:
                    entry_price = round(sym_state.best_ask, precision['price'])
                if entry_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'sell', entry_qty, entry_price,
                        position_side='SHORT', reduce_only=False
                    )
                    
        except Exception as e:
            logger.error(f"[Bot] {cfg.symbol} 下單失敗: {e}")

    async def cancel_orders_for_side(self, symbol: str, position_side: str):
        """
        取消指定方向的掛單 (與 Terminal 版一致)
        
        精確判斷取消邏輯:
        - position_side='long': 取消 BUY LONG (補倉) 和 SELL LONG (止盈) 訂單
        - position_side='short': 取消 SELL SHORT (補倉) 和 BUY SHORT (止盈) 訂單
        
        Args:
            symbol: CCXT 交易對 (如 XRP/USDC:USDC)
            position_side: 'long' 或 'short'
        """
        try:
            orders = self.adapter.fetch_open_orders(symbol)
            for order in orders:
                order_side = order.get('side', '').lower()
                order_pos_side = order.get('info', {}).get('positionSide', 'BOTH').upper()
                reduce_only = order.get('reduceOnly', False) or order.get('info', {}).get('reduceOnly', False)

                should_cancel = False
                if position_side.lower() == 'long':
                    # 多頭補倉單: buy + LONG + 非減倉
                    # 多頭止盈單: sell + LONG + 減倉
                    if (not reduce_only and order_side == 'buy' and order_pos_side == 'LONG') or \
                       (reduce_only and order_side == 'sell' and order_pos_side == 'LONG'):
                        should_cancel = True
                elif position_side.lower() == 'short':
                    # 空頭補倉單: sell + SHORT + 非減倉
                    # 空頭止盈單: buy + SHORT + 減倉
                    if (not reduce_only and order_side == 'sell' and order_pos_side == 'SHORT') or \
                       (reduce_only and order_side == 'buy' and order_pos_side == 'SHORT'):
                        should_cancel = True

                if should_cancel:
                    await asyncio.to_thread(self.adapter.cancel_order, order['id'], symbol)
        except Exception as e:
            logger.error(f"撤單失敗 {symbol}: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 風控邏輯
    # ═══════════════════════════════════════════════════════════════════════════

    async def _risk_monitor_loop(self):
        """風控監控循環 - 在 _sync_loop 中呼叫"""
        if not self.config.risk.enabled:
            return

        # 檢查保證金使用率
        await self._check_margin_threshold()

        # 檢查硬止損 (防爆倉)
        await self._check_hard_stop()

        # 檢查追蹤止盈
        await self._check_trailing_stop()

    async def _check_margin_threshold(self):
        """檢查保證金閾值"""
        risk = self.config.risk
        for currency in ["USDC", "USDT"]:
            acc = self.state.get_account(currency)
            if acc.equity <= 0:
                continue

            margin_ratio = acc.margin_ratio
            if margin_ratio >= risk.margin_threshold:
                logger.warning(f"[Risk] {currency} 保證金使用率 {margin_ratio*100:.1f}% 超過閾值 {risk.margin_threshold*100:.0f}%")
                # 可以在這裡觸發減倉邏輯

    async def _check_hard_stop(self):
        """
        硬止損檢查 - 防止爆倉

        當單方向浮虧超過帳戶權益的 max_loss_pct 時，強制平倉該方向
        例如: max_loss_pct = 0.03 (3%)，帳戶權益 $1000
              多單浮虧 > $30 時，強制平倉所有多單
        """
        risk = self.config.risk
        if not risk.hard_stop_enabled:
            return

        # 計算帳戶總權益
        total_equity = 0
        for currency in ["USDC", "USDT"]:
            acc = self.state.get_account(currency)
            total_equity += acc.equity

        if total_equity <= 0:
            return

        max_loss = total_equity * risk.max_loss_pct

        # 檢查每個交易對的多空方向浮虧
        for cfg in self.config.symbols.values():
            if not cfg.enabled:
                continue

            ccxt_sym = cfg.ccxt_symbol
            sym_state = self.state.symbols.get(ccxt_sym)
            if not sym_state:
                continue

            price = sym_state.latest_price
            if price <= 0:
                continue

            # 計算多單浮虧 (負數表示虧損)
            if sym_state.long_position > 0 and sym_state.long_avg_price > 0:
                long_pnl = (price - sym_state.long_avg_price) * sym_state.long_position
                if long_pnl < -max_loss:
                    logger.warning(f"[HardStop] {cfg.symbol} 多單浮虧 ${abs(long_pnl):.2f} 超過閾值 ${max_loss:.2f}")
                    await self._emergency_close_side(cfg, "long")

            # 計算空單浮虧 (負數表示虧損)
            if sym_state.short_position > 0 and sym_state.short_avg_price > 0:
                short_pnl = (sym_state.short_avg_price - price) * sym_state.short_position
                if short_pnl < -max_loss:
                    logger.warning(f"[HardStop] {cfg.symbol} 空單浮虧 ${abs(short_pnl):.2f} 超過閾值 ${max_loss:.2f}")
                    await self._emergency_close_side(cfg, "short")

    async def _emergency_close_side(self, cfg, side: str):
        """
        緊急平倉單一方向

        Args:
            cfg: SymbolConfig
            side: "long" 或 "short"
        """
        ccxt_sym = cfg.ccxt_symbol
        sym_state = self.state.symbols.get(ccxt_sym)
        if not sym_state:
            return

        try:
            # 取消該方向的掛單
            orders = self.adapter.fetch_open_orders(ccxt_sym)
            for order in orders:
                order_side = order.get('side', '').lower()
                # 多單平倉掛的是 sell 單，空單平倉掛的是 buy 單
                if side == "long" and order_side == "sell":
                    await asyncio.to_thread(self.adapter.cancel_order, order['id'], ccxt_sym)
                elif side == "short" and order_side == "buy":
                    await asyncio.to_thread(self.adapter.cancel_order, order['id'], ccxt_sym)

            # 市價平倉
            if side == "long" and sym_state.long_position > 0:
                await asyncio.to_thread(
                    self.adapter.create_market_order,
                    ccxt_sym, 'sell',
                    sym_state.long_position,
                    position_side=side.upper(),
                    reduce_only=True
                )
                logger.warning(f"[HardStop] {cfg.symbol} 強制平多倉 {sym_state.long_position}")
            
            elif side == "short" and sym_state.short_position > 0:
                await asyncio.to_thread(
                    self.adapter.create_market_order,
                    ccxt_sym, 'buy',
                    sym_state.short_position,
                    position_side=side.upper(),
                    reduce_only=True
                )
                logger.warning(f"[HardStop] {cfg.symbol} 強制平空倉 {sym_state.short_position}")

        except Exception as e:
            logger.error(f"[HardStop] {cfg.symbol} {side} 平倉失敗: {e}")

    async def _check_trailing_stop(self):
        """檢查追蹤止盈"""
        risk = self.config.risk
        total_pnl = self.state.total_unrealized_pnl

        # 更新峰值
        if total_pnl > self.state.peak_equity:
            self.state.peak_equity = total_pnl

        # 檢查是否達到啟動條件
        if total_pnl >= risk.trailing_start_profit:
            # 計算回撤
            drawdown = self.state.peak_equity - total_pnl

            # 動態回撤閾值 (最大 10% 或最小 $2)
            drawdown_threshold = max(
                self.state.peak_equity * risk.trailing_drawdown_pct,
                risk.trailing_min_drawdown
            )

            if drawdown >= drawdown_threshold:
                logger.warning(f"[Risk] 追蹤止盈觸發! 峰值={self.state.peak_equity:.2f}, "
                             f"當前={total_pnl:.2f}, 回撤={drawdown:.2f}")
                # 觸發市價平倉
                await self._emergency_close_all()

    async def _emergency_close_all(self):
        """緊急平倉所有持倉"""
        logger.warning("[Risk] 執行緊急平倉!")

        for cfg in self.config.symbols.values():
            if not cfg.enabled:
                continue

            ccxt_sym = cfg.ccxt_symbol
            sym_state = self.state.symbols.get(ccxt_sym)
            if not sym_state:
                continue

            try:
                # 取消所有掛單
                orders = self.adapter.fetch_open_orders(ccxt_sym)
                for order in orders:
                    await asyncio.to_thread(self.adapter.cancel_order, order['id'], ccxt_sym)

                # 市價平多倉
                if sym_state.long_position > 0:
                    await asyncio.to_thread(
                        self.adapter.create_market_order,
                        ccxt_sym, 'sell',
                        sym_state.long_position,
                        position_side='LONG',
                        reduce_only=True
                    )
                    logger.info(f"[Risk] {cfg.symbol} 平多倉 {sym_state.long_position}")

                # 市價平空倉
                if sym_state.short_position > 0:
                    await asyncio.to_thread(
                        self.adapter.create_market_order,
                        ccxt_sym, 'buy',
                        sym_state.short_position,
                        position_side='SHORT',
                        reduce_only=True
                    )
                    logger.info(f"[Risk] {cfg.symbol} 平空倉 {sym_state.short_position}")

            except Exception as e:
                logger.error(f"[Risk] {cfg.symbol} 緊急平倉失敗: {e}")

        # 停止交易
        self._stop_event.set()
        self.state.running = False
        logger.warning("[Risk] 交易已停止")
