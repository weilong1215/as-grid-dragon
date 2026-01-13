# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
回測管理器
==========
歷史數據載入、回測執行、參數優化
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import pandas as pd
import random

import ccxt

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .strategy import GridStrategy
from config.models import SymbolConfig
from constants import DATA_DIR

console = Console()


class BacktestManager:
    """回測管理器 - 簡化版，直接輸入交易對符號"""

    def __init__(self):
        self.data_dir = DATA_DIR

    def get_data_path(self, symbol_raw: str) -> Path:
        """獲取數據路徑"""
        return self.data_dir / f"futures/um/daily/klines/{symbol_raw}/1m"

    def get_available_dates(self, symbol_raw: str) -> List[str]:
        """獲取可用日期"""
        path = self.get_data_path(symbol_raw)
        if not path.exists():
            return []

        dates = []
        for f in path.glob(f"{symbol_raw}-1m-*.csv"):
            try:
                parts = f.stem.split('-')
                if len(parts) >= 5:
                    date_str = f"{parts[2]}-{parts[3]}-{parts[4]}"
                    dates.append(date_str)
            except Exception:
                pass

        return sorted(dates)

    def load_data(self, symbol_raw: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """載入歷史數據"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            path = self.get_data_path(symbol_raw) / f"{symbol_raw}-1m-{date_str}.csv"

            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if 'open_time' in df.columns:
                        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    all_data.append(df)
                except Exception as e:
                    console.print(f"[yellow]載入 {date_str} 失敗: {e}[/]")

            current += timedelta(days=1)

        if not all_data:
            return None

        full_df = pd.concat(all_data, ignore_index=True)
        return full_df.sort_values('open_time').reset_index(drop=True)

    def _create_exchange(self, exchange_type: str):
        """
        根據交易所類型創建 ccxt exchange 實例

        Args:
            exchange_type: 交易所類型 (binance, bybit, bitget, gate, okx)

        Returns:
            ccxt exchange 實例或 None
        """
        exchange_configs = {
            "binance": {
                "class": ccxt.binance,
                "options": {"defaultType": "future"}
            },
            "bybit": {
                "class": ccxt.bybit,
                "options": {"defaultType": "linear"}  # USDT 永續合約
            },
            "bitget": {
                "class": ccxt.bitget,
                "options": {"defaultType": "swap"}  # 永續合約
            },
            "gate": {
                "class": ccxt.gateio,
                "options": {"defaultType": "swap"}
            },
        }

        config = exchange_configs.get(exchange_type.lower())
        if config is None:
            return None

        try:
            return config["class"]({
                "enableRateLimit": True,
                "options": config["options"]
            })
        except Exception as e:
            console.print(f"[red]創建 {exchange_type} 交易所實例失敗: {e}[/]")
            return None

    def download_data(self, symbol_raw: str, ccxt_symbol: str, start_date: str, end_date: str,
                      exchange_type: str = "binance") -> bool:
        """
        下載歷史數據

        Args:
            symbol_raw: 原始交易對符號 (如 "XRPUSDC")
            ccxt_symbol: CCXT 格式符號 (如 "XRP/USDC:USDC")
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            exchange_type: 交易所類型 (binance, bybit, bitget, gate)
        """
        try:
            # 根據交易所類型創建 exchange 實例
            exchange = self._create_exchange(exchange_type)
            if exchange is None:
                console.print(f"[red]不支援的交易所: {exchange_type}[/]")
                return False

            # 轉換為 ccxt 格式 (不帶 :USDC)
            fetch_symbol = ccxt_symbol.split(":")[0]  # "XRP/USDC"

            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            total_bars = 0
            days = (end - start).days + 1

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"下載 {symbol_raw}...", total=days)
                current = start

                while current <= end:
                    date_str = current.strftime("%Y-%m-%d")
                    output_path = self.get_data_path(symbol_raw) / f"{symbol_raw}-1m-{date_str}.csv"

                    if not output_path.exists():
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        since = int(datetime(current.year, current.month, current.day).timestamp() * 1000)
                        until = since + 24 * 60 * 60 * 1000

                        try:
                            ohlcv = exchange.fetch_ohlcv(fetch_symbol, "1m", since=since, limit=1500)
                            if ohlcv:
                                ohlcv = [bar for bar in ohlcv if bar[0] < until]
                                df = pd.DataFrame(ohlcv, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
                                df.to_csv(output_path, index=False)
                                total_bars += len(df)
                        except Exception as e:
                            console.print(f"[red]{date_str}: {e}[/]")

                    current += timedelta(days=1)
                    progress.update(task, advance=1)

            console.print(f"[green]下載完成: {total_bars:,} 條數據[/]")
            return True

        except Exception as e:
            console.print(f"[red]下載失敗: {e}[/]")
            return False

    def run_backtest(self, config: SymbolConfig, df: pd.DataFrame,
                     hard_stop_pct: float = 0.03,
                     slippage_pct: float = 0.0005,
                     funding_rate: float = 0.0001,
                     funding_interval: int = 480) -> dict:
        """
        執行回測

        同步實盤邏輯:
        1. position_threshold (裝死模式): 持倉超過此值不補倉，只掛特殊止盈
        2. position_limit (止盈加倍): 持倉超過此值或對側超過 threshold，止盈數量加倍
        3. hard_stop_pct (硬止損): 單方向浮虧超過帳戶權益的此比例時強制平倉
        4. slippage_pct (滑點): 開平倉時的價格滑點 (預設 0.05%)
        5. funding_rate (資金費率): 每 8 小時結算的費率 (預設 0.01%)
        6. funding_interval (結算間隔): 資金費率結算間隔 (預設 480 根 1 分鐘 K 線 = 8 小時)

        Args:
            config: 交易對配置
            df: K線數據
            hard_stop_pct: 硬止損閾值 (預設 3%)
            slippage_pct: 滑點比例 (預設 0.05%)
            funding_rate: 資金費率 (預設 0.01%)
            funding_interval: 資金費率結算間隔 (預設 480 分鐘)
        """
        balance = 1000.0
        initial_balance = balance
        max_equity = balance

        long_positions = []
        short_positions = []
        trades = []
        equity_curve = []
        hard_stop_triggered = 0  # 統計硬止損觸發次數

        order_value = config.initial_quantity * df['close'].iloc[0]
        leverage = config.leverage
        fee_pct = 0.0004

        # === 費用追蹤 ===
        total_trading_fees = 0.0      # 交易手續費
        total_slippage_cost = 0.0     # 滑點成本
        total_funding_paid = 0.0      # 資金費率支出
        bar_count = 0                 # K 線計數 (用於資金費率結算)

        # 持倉控制參數
        position_threshold = config.position_threshold
        position_limit = config.position_limit

        # 追蹤上次開倉價格 (用於計算止盈價)
        last_long_entry_price = df['close'].iloc[0]
        last_short_entry_price = df['close'].iloc[0]

        # === 趨勢過濾器 - 計算 MA ===
        trend_filter_enabled = getattr(config, 'trend_filter_enabled', False)
        trend_ma_period = getattr(config, 'trend_ma_period', 200)
        trend_buffer_pct = getattr(config, 'trend_buffer_pct', 0.005)

        if trend_filter_enabled:
            df = df.copy()
            df['ma'] = df['close'].rolling(window=trend_ma_period, min_periods=1).mean()
        else:
            df['ma'] = None

        trend_filtered_longs = 0   # 統計被過濾的多單
        trend_filtered_shorts = 0  # 統計被過濾的空單

        for idx, row in df.iterrows():
            price = row['close']
            high_price = row['high'] if 'high' in row else price
            low_price = row['low'] if 'low' in row else price
            ma_value = row['ma'] if trend_filter_enabled else None
            bar_count += 1

            # === 資金費率結算 (每 8 小時) ===
            if funding_rate > 0 and bar_count % funding_interval == 0:
                long_position_value = sum(p["qty"] * price for p in long_positions)
                short_position_value = sum(p["qty"] * price for p in short_positions)
                # 多單持有者支付資金費率 (正常情況下 funding > 0)
                # 空單持有者收取資金費率
                # 簡化模型：假設資金費率總是正的，多單支付給空單
                funding_cost = (long_position_value - short_position_value) * funding_rate
                balance -= funding_cost
                total_funding_paid += funding_cost

            # 趨勢判斷 (帶緩衝區)
            allow_long = True
            allow_short = True
            if trend_filter_enabled and ma_value is not None:
                upper_band = ma_value * (1 + trend_buffer_pct)
                lower_band = ma_value * (1 - trend_buffer_pct)
                # 價格在 MA 上方 → 多頭趨勢，只做多
                # 價格在 MA 下方 → 空頭趨勢，只做空
                if price > upper_band:
                    allow_short = False  # 不開新空單
                elif price < lower_band:
                    allow_long = False   # 不開新多單

            # 計算當前持倉量
            long_position = sum(p["qty"] for p in long_positions)
            short_position = sum(p["qty"] for p in short_positions)

            # === 同步實盤邏輯：使用當前價格計算網格 ===
            # 多頭網格
            long_decision = GridStrategy.get_grid_decision(
                price=price,  # 使用當前價格（同步實盤）
                my_position=long_position,
                opposite_position=short_position,
                position_threshold=position_threshold,
                position_limit=position_limit,
                base_qty=config.initial_quantity,
                take_profit_spacing=config.take_profit_spacing,
                grid_spacing=config.grid_spacing,
                side='long'
            )

            # 空頭網格
            short_decision = GridStrategy.get_grid_decision(
                price=price,  # 使用當前價格（同步實盤）
                my_position=short_position,
                opposite_position=long_position,
                position_threshold=position_threshold,
                position_limit=position_limit,
                base_qty=config.initial_quantity,
                take_profit_spacing=config.take_profit_spacing,
                grid_spacing=config.grid_spacing,
                side='short'
            )

            long_dead_mode = long_decision['dead_mode']
            short_dead_mode = short_decision['dead_mode']

            # === 多頭開倉邏輯 ===
            if not long_dead_mode and allow_long:  # 加入趨勢過濾
                buy_price = long_decision['entry_price'] if long_decision['entry_price'] else price * (1 - config.grid_spacing)
                # 使用 low_price 檢查是否觸發 (盤中最低價可能觸發買入)
                if low_price <= buy_price:
                    # 實際成交價取 buy_price 或 low_price 中較高者 (模擬限價單成交)
                    fill_price = max(buy_price, low_price)
                    # 計算滑點後的實際成交價 (買入時滑點向上)
                    slippage = random.uniform(0, slippage_pct)
                    actual_entry_price = fill_price * (1 + slippage)
                    slippage_cost = fill_price * slippage

                    qty = order_value / actual_entry_price
                    margin = (qty * actual_entry_price) / leverage
                    fee = qty * actual_entry_price * fee_pct

                    if margin + fee < balance:
                        balance -= (margin + fee)
                        long_positions.append({"price": actual_entry_price, "qty": qty, "margin": margin})
                        last_long_entry_price = actual_entry_price
                        total_trading_fees += fee
                        total_slippage_cost += slippage_cost * qty
            elif not long_dead_mode and not allow_long:
                trend_filtered_longs += 1  # 統計被過濾的多單

            # === 多頭止盈邏輯 ===
            if long_positions:
                # 止盈價格基於最早持倉的開倉價
                oldest_long_price = long_positions[0]["price"]
                sell_price = oldest_long_price * (1 + config.take_profit_spacing)
                long_tp_qty = long_decision['tp_qty']

                # 使用 high_price 檢查是否觸發 (盤中最高價可能觸發止盈)
                if high_price >= sell_price:
                    # 實際成交價取 sell_price 或 high_price 中較低者 (模擬限價單成交)
                    fill_price = min(sell_price, high_price)
                    # 計算滑點後的實際成交價 (賣出時滑點向下)
                    slippage = random.uniform(0, slippage_pct)
                    actual_exit_price = fill_price * (1 - slippage)

                    remaining_tp = long_tp_qty
                    while long_positions and remaining_tp > 0:
                        pos = long_positions[0]
                        if pos["qty"] <= remaining_tp:
                            long_positions.pop(0)
                            gross_pnl = (actual_exit_price - pos["price"]) * pos["qty"]
                            fee = pos["qty"] * actual_exit_price * fee_pct
                            net_pnl = gross_pnl - fee
                            balance += pos["margin"] + net_pnl
                            trades.append({"pnl": net_pnl, "type": "long"})
                            total_trading_fees += fee
                            remaining_tp -= pos["qty"]
                        else:
                            close_ratio = remaining_tp / pos["qty"]
                            close_qty = remaining_tp
                            close_margin = pos["margin"] * close_ratio
                            gross_pnl = (actual_exit_price - pos["price"]) * close_qty
                            fee = close_qty * actual_exit_price * fee_pct
                            net_pnl = gross_pnl - fee
                            balance += close_margin + net_pnl
                            trades.append({"pnl": net_pnl, "type": "long"})
                            total_trading_fees += fee
                            pos["qty"] -= close_qty
                            pos["margin"] -= close_margin
                            remaining_tp = 0

            # === 空頭開倉邏輯 ===
            if not short_dead_mode and allow_short:  # 加入趨勢過濾
                sell_short_price = short_decision['entry_price'] if short_decision['entry_price'] else price * (1 + config.grid_spacing)
                # 使用 high_price 檢查是否觸發 (盤中最高價可能觸發賣空)
                if high_price >= sell_short_price:
                    # 實際成交價取 sell_short_price 或 high_price 中較低者 (模擬限價單成交)
                    fill_price = min(sell_short_price, high_price)
                    # 計算滑點後的實際成交價 (賣空時滑點向下)
                    slippage = random.uniform(0, slippage_pct)
                    actual_entry_price = fill_price * (1 - slippage)
                    slippage_cost = fill_price * slippage

                    qty = order_value / actual_entry_price
                    margin = (qty * actual_entry_price) / leverage
                    fee = qty * actual_entry_price * fee_pct

                    if margin + fee < balance:
                        balance -= (margin + fee)
                        short_positions.append({"price": actual_entry_price, "qty": qty, "margin": margin})
                        last_short_entry_price = actual_entry_price
                        total_trading_fees += fee
                        total_slippage_cost += slippage_cost * qty
            elif not short_dead_mode and not allow_short:
                trend_filtered_shorts += 1  # 統計被過濾的空單

            # === 空頭止盈邏輯 ===
            if short_positions:
                # 止盈價格基於最早持倉的開倉價
                oldest_short_price = short_positions[0]["price"]
                cover_price = oldest_short_price * (1 - config.take_profit_spacing)
                short_tp_qty = short_decision['tp_qty']

                # 使用 low_price 檢查是否觸發 (盤中最低價可能觸發止盈)
                if low_price <= cover_price:
                    # 實際成交價取 cover_price 或 low_price 中較高者 (模擬限價單成交)
                    fill_price = max(cover_price, low_price)
                    # 計算滑點後的實際成交價 (買回平倉時滑點向上)
                    slippage = random.uniform(0, slippage_pct)
                    actual_exit_price = fill_price * (1 + slippage)

                    remaining_tp = short_tp_qty
                    while short_positions and remaining_tp > 0:
                        pos = short_positions[0]
                        if pos["qty"] <= remaining_tp:
                            short_positions.pop(0)
                            gross_pnl = (pos["price"] - actual_exit_price) * pos["qty"]
                            fee = pos["qty"] * actual_exit_price * fee_pct
                            net_pnl = gross_pnl - fee
                            balance += pos["margin"] + net_pnl
                            trades.append({"pnl": net_pnl, "type": "short"})
                            total_trading_fees += fee
                            remaining_tp -= pos["qty"]
                        else:
                            close_ratio = remaining_tp / pos["qty"]
                            close_qty = remaining_tp
                            close_margin = pos["margin"] * close_ratio
                            gross_pnl = (pos["price"] - actual_exit_price) * close_qty
                            fee = close_qty * actual_exit_price * fee_pct
                            net_pnl = gross_pnl - fee
                            balance += close_margin + net_pnl
                            trades.append({"pnl": net_pnl, "type": "short"})
                            total_trading_fees += fee
                            pos["qty"] -= close_qty
                            pos["margin"] -= close_margin
                            remaining_tp = 0

            # 計算淨值
            long_unrealized = sum((price - p["price"]) * p["qty"] for p in long_positions)
            short_unrealized = sum((p["price"] - price) * p["qty"] for p in short_positions)
            unrealized = long_unrealized + short_unrealized
            equity = balance + unrealized
            max_equity = max(max_equity, equity)
            equity_curve.append(equity)

            # === 硬止損檢查 ===
            if hard_stop_pct > 0:
                max_loss = initial_balance * hard_stop_pct

                # 多單浮虧超過閾值，強制平倉
                if long_unrealized < -max_loss and long_positions:
                    for pos in long_positions:
                        gross_pnl = (price - pos["price"]) * pos["qty"]
                        fee = pos["qty"] * price * fee_pct
                        net_pnl = gross_pnl - fee
                        balance += pos["margin"] + net_pnl
                        trades.append({"pnl": net_pnl, "type": "long_stop"})
                    long_positions = []
                    hard_stop_triggered += 1

                # 空單浮虧超過閾值，強制平倉
                if short_unrealized < -max_loss and short_positions:
                    for pos in short_positions:
                        gross_pnl = (pos["price"] - price) * pos["qty"]
                        fee = pos["qty"] * price * fee_pct
                        net_pnl = gross_pnl - fee
                        balance += pos["margin"] + net_pnl
                        trades.append({"pnl": net_pnl, "type": "short_stop"})
                    short_positions = []
                    hard_stop_triggered += 1

        # 計算結果
        final_price = df['close'].iloc[-1]
        unrealized_pnl = sum((final_price - p["price"]) * p["qty"] for p in long_positions)
        unrealized_pnl += sum((p["price"] - final_price) * p["qty"] for p in short_positions)

        realized_pnl = sum(t["pnl"] for t in trades)
        final_equity = balance + unrealized_pnl

        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] < 0]

        return {
            "final_equity": final_equity,
            "initial_balance": 1000.0,
            "return_pct": (final_equity - 1000) / 1000,
            "max_drawdown": 1 - (min(equity_curve) / max_equity) if equity_curve else 0,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "trades_count": len(trades),
            "win_rate": len(winning) / len(trades) if trades else 0,
            "profit_factor": sum(t["pnl"] for t in winning) / abs(sum(t["pnl"] for t in losing)) if losing else float('inf'),
            # 新增詳細統計
            "long_trades": len([t for t in trades if t["type"] == "long"]),
            "short_trades": len([t for t in trades if t["type"] == "short"]),
            "avg_win": sum(t["pnl"] for t in winning) / len(winning) if winning else 0,
            "avg_loss": sum(t["pnl"] for t in losing) / len(losing) if losing else 0,
            "equity_curve": equity_curve,  # 用於繪製收益曲線
            "max_equity": max_equity,
            "min_equity": min(equity_curve) if equity_curve else 1000.0,
            # 硬止損統計
            "hard_stop_triggered": hard_stop_triggered,
            "stop_trades": len([t for t in trades if "stop" in t.get("type", "")]),
            # 趨勢過濾統計
            "trend_filtered_longs": trend_filtered_longs,
            "trend_filtered_shorts": trend_filtered_shorts,
            # 費用統計
            "total_trading_fees": total_trading_fees,
            "total_slippage_cost": total_slippage_cost,
            "total_funding_paid": total_funding_paid,
            "total_costs": total_trading_fees + total_slippage_cost + total_funding_paid,
        }

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7) -> tuple:
        """
        分割數據為訓練集和測試集

        Args:
            df: 完整數據
            train_ratio: 訓練集比例 (預設 70%)

        Returns:
            (train_df, test_df): 訓練集和測試集
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df

    def optimize_params(self, config: SymbolConfig, df: pd.DataFrame,
                        progress_callback=None, use_validation: bool = True,
                        train_ratio: float = 0.7) -> List[dict]:
        """
        優化參數

        Args:
            config: 交易對配置
            df: K線數據
            progress_callback: 進度回調函數
            use_validation: 是否使用 train/test 分割驗證
            train_ratio: 訓練集比例 (預設 70%)

        Returns:
            優化結果列表，包含訓練集和測試集的表現
        """
        results = []

        take_profits = [0.002, 0.003, 0.004, 0.005, 0.006]
        grid_spacings = [0.004, 0.006, 0.008, 0.01, 0.012]

        valid_combos = [(tp, gs) for tp in take_profits for gs in grid_spacings if tp < gs]
        total = len(valid_combos)

        # 分割數據
        if use_validation:
            train_df, test_df = self.split_data(df, train_ratio)
            console.print(f"[cyan]數據分割: 訓練 {len(train_df):,} 筆 ({train_ratio*100:.0f}%), "
                         f"測試 {len(test_df):,} 筆 ({(1-train_ratio)*100:.0f}%)[/]")
        else:
            train_df = df
            test_df = None

        for i, (tp, gs) in enumerate(valid_combos):
            test_config = SymbolConfig(
                symbol=config.symbol,
                ccxt_symbol=config.ccxt_symbol,
                take_profit_spacing=tp,
                grid_spacing=gs,
                initial_quantity=config.initial_quantity,
                leverage=config.leverage
            )

            # 在訓練集上回測
            train_result = self.run_backtest(test_config, train_df)

            result = {
                "take_profit_spacing": tp,
                "grid_spacing": gs,
                "train_return_pct": train_result["return_pct"],
                "train_max_drawdown": train_result["max_drawdown"],
                "train_win_rate": train_result["win_rate"],
                "train_trades": train_result["trades_count"],
            }

            # 在測試集上驗證
            if use_validation and test_df is not None and len(test_df) > 0:
                test_result = self.run_backtest(test_config, test_df)
                result["test_return_pct"] = test_result["return_pct"]
                result["test_max_drawdown"] = test_result["max_drawdown"]
                result["test_win_rate"] = test_result["win_rate"]
                result["test_trades"] = test_result["trades_count"]
                # 計算過擬合指標: 訓練收益 vs 測試收益的差距
                result["overfit_ratio"] = (
                    (train_result["return_pct"] - test_result["return_pct"])
                    / abs(train_result["return_pct"]) if train_result["return_pct"] != 0 else 0
                )
            else:
                result["test_return_pct"] = None
                result["test_max_drawdown"] = None
                result["test_win_rate"] = None
                result["test_trades"] = None
                result["overfit_ratio"] = None

            # 保留完整回測結果用於詳細分析
            result["full_result"] = train_result

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        # 按測試集收益排序 (如果有)，否則按訓練集收益排序
        if use_validation:
            results.sort(key=lambda x: x.get("test_return_pct", 0) or 0, reverse=True)
        else:
            results.sort(key=lambda x: x["train_return_pct"], reverse=True)

        return results
