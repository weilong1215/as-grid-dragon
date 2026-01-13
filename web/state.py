"""
狀態管理模組
============
管理 Streamlit session state 和交易機器人生命週期
"""

import threading
import asyncio
from datetime import datetime
from typing import Optional

import streamlit as st

# 確保可以導入專案模組
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.models import GlobalConfig


def init_session_state():
    """初始化 session state"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.config = GlobalConfig.load()
        st.session_state.bot = None
        st.session_state.bot_thread = None
        st.session_state.bot_loop = None
        st.session_state.trading_active = False
        st.session_state.start_time = None
        st.session_state.last_error = None


def get_config() -> GlobalConfig:
    """取得配置"""
    init_session_state()
    return st.session_state.config


def save_config():
    """儲存配置"""
    st.session_state.config.save()


def reload_config():
    """重新載入配置"""
    st.session_state.config = GlobalConfig.load()


def is_trading_active() -> bool:
    """檢查交易是否運行中"""
    init_session_state()
    return st.session_state.trading_active


def get_bot():
    """取得 bot 實例"""
    init_session_state()
    return st.session_state.bot


def start_trading() -> tuple[bool, str]:
    """
    啟動交易

    Returns:
        (success, message)
    """
    init_session_state()
    config = st.session_state.config

    # 檢查 API 設定
    if not config.api_key or not config.api_secret:
        return False, "請先設定 API Key 和 Secret"

    # 檢查啟用的交易對
    enabled = [s for s in config.symbols.values() if s.enabled]
    if not enabled:
        return False, "沒有啟用的交易對"

    # 如果已經在運行
    if st.session_state.trading_active:
        return False, "交易已在運行中"

    try:
        from core.bot import MaxGridBot
        import traceback

        print(f"[DEBUG] 創建 MaxGridBot，config.api_key: {config.api_key[:8]}...")
        print(f"[DEBUG] exchange_type: {config.exchange_type}")
        print(f"[DEBUG] 啟用的交易對: {[s.symbol for s in config.symbols.values() if s.enabled]}")

        bot = MaxGridBot(config)
        st.session_state.bot = bot
        st.session_state.last_error = None  # 清除上次錯誤
        print("[DEBUG] MaxGridBot 創建成功")

        # 使用閉包捕獲 bot 和共享狀態容器
        # (因為 Streamlit session_state 無法在背景線程中存取)
        thread_state = {"error": None, "loop": None}

        def run_bot_thread(bot_instance, state_container):
            """在獨立執行緒中運行 bot"""
            loop = asyncio.new_event_loop()
            state_container["loop"] = loop
            asyncio.set_event_loop(loop)
            try:
                print("[DEBUG] Bot thread started, calling bot.run()...")
                loop.run_until_complete(bot_instance.run())
            except Exception as e:
                # 保存完整錯誤資訊
                error_detail = f"{type(e).__name__}: {str(e)}"
                full_trace = traceback.format_exc()
                state_container["error"] = error_detail
                # 輸出到控制台（方便 Debug）
                print(f"[ERROR] Bot 執行錯誤: {error_detail}")
                print(f"[ERROR] Traceback:\n{full_trace}")
            finally:
                loop.close()
                print("[DEBUG] Bot thread ended")

        st.session_state.thread_state = thread_state
        st.session_state.bot_thread = threading.Thread(
            target=run_bot_thread,
            args=(bot, thread_state),
            daemon=True
        )
        st.session_state.bot_thread.start()

        # 等待連接建立 (改進: 如果線程死亡提早退出)
        import time
        for i in range(200):  # 最多等 20 秒 (load_markets 可能較慢)
            if st.session_state.bot.state.running:
                break
            # 如果線程已死亡，提早退出
            if not st.session_state.bot_thread.is_alive():
                time.sleep(0.2)  # 等待錯誤訊息寫入
                break
            time.sleep(0.1)

        if st.session_state.bot.state.running:
            st.session_state.trading_active = True
            st.session_state.start_time = datetime.now()
            return True, f"交易已啟動，運行 {len(enabled)} 個交易對"
        else:
            st.session_state.bot = None
            # 從線程狀態容器讀取錯誤
            error_msg = thread_state.get("error", "")
            if error_msg:
                return False, f"Bot 啟動失敗: {error_msg}"
            return False, "Bot 啟動失敗，請檢查 API 設定或網路連接"

    except Exception as e:
        st.session_state.last_error = str(e)
        return False, f"啟動失敗: {str(e)}"


def stop_trading() -> tuple[bool, str]:
    """
    停止交易

    Returns:
        (success, message)
    """
    init_session_state()

    if not st.session_state.trading_active or not st.session_state.bot:
        return False, "交易未運行"

    try:
        bot = st.session_state.bot
        thread_state = st.session_state.get("thread_state", {})
        loop = thread_state.get("loop")

        # 發送停止信號
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(bot.stop(), loop)

        # 等待執行緒結束
        if st.session_state.bot_thread and st.session_state.bot_thread.is_alive():
            st.session_state.bot_thread.join(timeout=5)

        st.session_state.trading_active = False
        st.session_state.bot = None
        st.session_state.start_time = None
        st.session_state.thread_state = None

        return True, "交易已停止"

    except Exception as e:
        return False, f"停止失敗: {str(e)}"


def get_trading_duration() -> Optional[str]:
    """取得交易運行時間"""
    if not st.session_state.get("start_time"):
        return None

    duration = datetime.now() - st.session_state.start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_trading_stats() -> dict:
    """取得交易統計資訊"""
    bot = get_bot()
    if not bot or not st.session_state.trading_active:
        return {}

    state = bot.state
    return {
        "running": state.running,
        "total_unrealized_pnl": state.total_unrealized_pnl,
        "symbols": {
            name: {
                "price": sym.latest_price,
                "long_position": sym.long_position,
                "short_position": sym.short_position,
                "unrealized_pnl": sym.unrealized_pnl,
                "long_avg_price": sym.long_avg_price,
                "short_avg_price": sym.short_avg_price,
            }
            for name, sym in state.symbols.items()
        },
        "accounts": {
            name: {
                "wallet_balance": acc.wallet_balance,
                "available_balance": acc.available_balance,
                "unrealized_pnl": acc.unrealized_pnl,
            }
            for name, acc in state.accounts.items()
        }
    }
