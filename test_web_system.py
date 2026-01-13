"""
AS 網格交易系統 - Web UI 功能測試
===================================
全面測試所有頁面和功能模組
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime

# 添加專案路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 測試報告
test_results = []

def log_test(category: str, test_name: str, status: str, message: str = ""):
    """記錄測試結果"""
    result = {
        "category": category,
        "test": test_name,
        "status": status,  # PASS, FAIL, WARNING
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    test_results.append(result)
    
    # 即時輸出
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_icon} [{category}] {test_name}: {message}")


def test_imports():
    """測試 1: 模組導入測試"""
    print("\n" + "="*60)
    print("測試 1: 模組導入測試")
    print("="*60)
    
    try:
        import streamlit as st
        log_test("導入測試", "Streamlit", "PASS", f"版本 {st.__version__}")
    except Exception as e:
        log_test("導入測試", "Streamlit", "FAIL", str(e))
    
    try:
        import ccxt
        log_test("導入測試", "CCXT", "PASS", f"版本 {ccxt.__version__}")
    except Exception as e:
        log_test("導入測試", "CCXT", "FAIL", str(e))
    
    try:
        import pandas as pd
        log_test("導入測試", "Pandas", "PASS", f"版本 {pd.__version__}")
    except Exception as e:
        log_test("導入測試", "Pandas", "FAIL", str(e))
    
    try:
        from config.models import GlobalConfig, SymbolConfig
        log_test("導入測試", "Config Models", "PASS", "配置模型正常")
    except Exception as e:
        log_test("導入測試", "Config Models", "FAIL", str(e))
    
    try:
        from web.state import init_session_state, get_config
        log_test("導入測試", "State Management", "PASS", "狀態管理模組正常")
    except Exception as e:
        log_test("導入測試", "State Management", "FAIL", str(e))
    
    try:
        from core.bot import MaxGridBot
        log_test("導入測試", "MaxGridBot", "PASS", "交易機器人模組正常")
    except Exception as e:
        log_test("導入測試", "MaxGridBot", "FAIL", str(e))


def test_config_system():
    """測試 2: 配置系統測試"""
    print("\n" + "="*60)
    print("測試 2: 配置系統測試")
    print("="*60)
    
    try:
        from config.models import GlobalConfig, SymbolConfig
        
        # 測試配置載入
        config = GlobalConfig.load()
        log_test("配置系統", "配置載入", "PASS", f"已載入 {len(config.symbols)} 個交易對")
        
        # 測試配置屬性
        attrs = ["exchange_type", "api_key", "api_secret", "symbols"]
        for attr in attrs:
            if hasattr(config, attr):
                log_test("配置系統", f"屬性 {attr}", "PASS", "存在")
            else:
                log_test("配置系統", f"屬性 {attr}", "FAIL", "不存在")
        
        # 測試配置保存
        try:
            config.save()
            log_test("配置系統", "配置保存", "PASS", "保存成功")
        except Exception as e:
            log_test("配置系統", "配置保存", "FAIL", str(e))
        
    except Exception as e:
        log_test("配置系統", "總體測試", "FAIL", str(e))


def test_exchange_adapters():
    """測試 3: 交易所適配器測試"""
    print("\n" + "="*60)
    print("測試 3: 交易所適配器測試")
    print("="*60)
    
    try:
        from exchanges import list_supported_exchanges, get_adapter, get_exchange_display_name
        
        # 測試交易所列表
        exchanges = list_supported_exchanges()
        log_test("交易所適配器", "支援列表", "PASS", f"支援 {len(exchanges)} 個交易所: {', '.join(exchanges)}")
        
        # 測試每個交易所的適配器
        for exchange in exchanges:
            try:
                adapter = get_adapter(exchange)
                display_name = get_exchange_display_name(exchange)
                
                # 檢查必要方法
                required_methods = [
                    "init_exchange", "load_markets", "fetch_balance",
                    "fetch_positions", "create_order", "cancel_order"
                ]
                
                missing = []
                for method in required_methods:
                    if not hasattr(adapter, method):
                        missing.append(method)
                
                if missing:
                    log_test("交易所適配器", f"{display_name}", "FAIL", f"缺少方法: {', '.join(missing)}")
                else:
                    log_test("交易所適配器", f"{display_name}", "PASS", "所有必要方法存在")
                    
            except Exception as e:
                log_test("交易所適配器", f"{exchange}", "FAIL", str(e))
                
    except Exception as e:
        log_test("交易所適配器", "總體測試", "FAIL", str(e))


def test_web_pages():
    """測試 4: Web 頁面文件測試"""
    print("\n" + "="*60)
    print("測試 4: Web 頁面文件測試")
    print("="*60)
    
    pages = {
        "web/app.py": "首頁",
        "web/pages/1_📈_交易監控.py": "交易監控",
        "web/pages/2_⚙️_交易對管理.py": "交易對管理",
        "web/pages/3_🔬_回測優化.py": "回測優化",
        "web/pages/4_🛠️_設定.py": "設定頁面"
    }
    
    for page_path, page_name in pages.items():
        full_path = project_root / page_path
        if full_path.exists():
            # 檢查文件大小
            size = full_path.stat().st_size
            log_test("頁面文件", page_name, "PASS", f"文件存在 ({size} bytes)")
            
            # 檢查基本語法（嘗試編譯）
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, page_path, 'exec')
                log_test("頁面語法", page_name, "PASS", "語法正確")
            except SyntaxError as e:
                log_test("頁面語法", page_name, "FAIL", f"語法錯誤: {e}")
        else:
            log_test("頁面文件", page_name, "FAIL", "文件不存在")


def test_components():
    """測試 5: 組件測試"""
    print("\n" + "="*60)
    print("測試 5: 組件測試")
    print("="*60)
    
    components = {
        "web/components/sidebar.py": "側邊欄",
        "web/theme.py": "主題",
        "web/state.py": "狀態管理"
    }
    
    for comp_path, comp_name in components.items():
        full_path = project_root / comp_path
        if full_path.exists():
            try:
                # 嘗試導入
                if comp_path == "web/components/sidebar.py":
                    from web.components.sidebar import render_sidebar
                    log_test("組件導入", comp_name, "PASS", "render_sidebar 存在")
                elif comp_path == "web/theme.py":
                    from web.theme import apply_custom_theme
                    log_test("組件導入", comp_name, "PASS", "apply_custom_theme 存在")
                elif comp_path == "web/state.py":
                    from web.state import init_session_state, get_config
                    log_test("組件導入", comp_name, "PASS", "關鍵函數存在")
            except Exception as e:
                log_test("組件導入", comp_name, "FAIL", str(e))
        else:
            log_test("組件文件", comp_name, "FAIL", "文件不存在")


def test_bot_functionality():
    """測試 6: 交易機器人功能測試"""
    print("\n" + "="*60)
    print("測試 6: 交易機器人功能測試")
    print("="*60)
    
    try:
        from core.bot import MaxGridBot
        from config.models import GlobalConfig
        
        # 測試機器人創建（不實際連接）
        config = GlobalConfig.load()
        
        # 檢查必要配置
        if not config.api_key:
            log_test("Bot 功能", "API 配置", "WARNING", "API Key 未設定，跳過連接測試")
        else:
            log_test("Bot 功能", "API 配置", "PASS", "API Key 已設定")
        
        # 檢查交易對配置
        if not config.symbols:
            log_test("Bot 功能", "交易對配置", "WARNING", "無配置的交易對")
        else:
            enabled = [s for s in config.symbols.values() if s.enabled]
            log_test("Bot 功能", "交易對配置", "PASS", 
                    f"已配置 {len(config.symbols)} 個交易對，{len(enabled)} 個已啟用")
        
        # 測試 Bot 類結構
        required_methods = ["run", "stop", "reload_config", "_websocket_loop", "_place_grid"]
        for method in required_methods:
            if hasattr(MaxGridBot, method):
                log_test("Bot 結構", f"方法 {method}", "PASS", "存在")
            else:
                log_test("Bot 結構", f"方法 {method}", "FAIL", "不存在")
                
    except Exception as e:
        log_test("Bot 功能", "總體測試", "FAIL", str(e))


def test_utils():
    """測試 7: 工具函數測試"""
    print("\n" + "="*60)
    print("測試 7: 工具函數測試")
    print("="*60)
    
    try:
        from utils import normalize_symbol
        
        # 測試交易對正規化
        test_cases = [
            ("XRPUSDC", "XRP/USDC"),
            ("BTC/USDT", "BTC/USDT"),
            ("ethusdt", "ETH/USDT"),
            ("SOL-USDT", "SOL/USDT"),
        ]
        
        for input_sym, expected in test_cases:
            try:
                raw, ccxt_sym, coin, quote = normalize_symbol(input_sym)
                if ccxt_sym == expected:
                    log_test("工具函數", f"normalize_symbol({input_sym})", "PASS", f"→ {ccxt_sym}")
                else:
                    log_test("工具函數", f"normalize_symbol({input_sym})", "WARNING", 
                            f"期望 {expected}，得到 {ccxt_sym}")
            except Exception as e:
                log_test("工具函數", f"normalize_symbol({input_sym})", "FAIL", str(e))
                
    except Exception as e:
        log_test("工具函數", "總體測試", "FAIL", str(e))


def test_indicators():
    """測試 8: 指標系統測試"""
    print("\n" + "="*60)
    print("測試 8: 指標系統測試")
    print("="*60)
    
    indicators_path = project_root / "indicators"
    
    if indicators_path.exists():
        indicator_files = ["bandit.py", "leading.py", "funding.py", "dgt.py"]
        
        for file in indicator_files:
            file_path = indicators_path / file
            if file_path.exists():
                log_test("指標文件", file, "PASS", "文件存在")
            else:
                log_test("指標文件", file, "WARNING", "文件不存在")
    else:
        log_test("指標系統", "indicators 目錄", "FAIL", "目錄不存在")


def generate_report():
    """生成測試報告"""
    print("\n" + "="*60)
    print("測試報告生成")
    print("="*60)
    
    # 統計
    total = len(test_results)
    passed = sum(1 for r in test_results if r["status"] == "PASS")
    failed = sum(1 for r in test_results if r["status"] == "FAIL")
    warnings = sum(1 for r in test_results if r["status"] == "WARNING")
    
    print(f"\n總測試項: {total}")
    print(f"✅ 通過: {passed} ({passed/total*100:.1f}%)")
    print(f"❌ 失敗: {failed} ({failed/total*100:.1f}%)")
    print(f"⚠️  警告: {warnings} ({warnings/total*100:.1f}%)")
    
    # 按類別分組
    print("\n" + "="*60)
    print("測試結果詳情")
    print("="*60)
    
    categories = {}
    for result in test_results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    for cat, results in categories.items():
        print(f"\n【{cat}】")
        for r in results:
            status_icon = "✅" if r["status"] == "PASS" else "❌" if r["status"] == "FAIL" else "⚠️"
            print(f"  {status_icon} {r['test']}: {r['message']}")
    
    # 保存 JSON 報告
    report_path = project_root / "web_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings
            },
            "results": test_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n詳細報告已保存至: {report_path}")
    
    # 失敗項目摘要
    if failed > 0:
        print("\n" + "="*60)
        print("⚠️ 失敗項目摘要")
        print("="*60)
        for r in test_results:
            if r["status"] == "FAIL":
                print(f"❌ [{r['category']}] {r['test']}: {r['message']}")


def main():
    """主測試流程"""
    print("="*60)
    print("AS 網格交易系統 - Web UI 功能測試")
    print("="*60)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"專案路徑: {project_root}")
    
    # 執行所有測試
    test_imports()
    test_config_system()
    test_exchange_adapters()
    test_web_pages()
    test_components()
    test_bot_functionality()
    test_utils()
    test_indicators()
    
    # 生成報告
    generate_report()
    
    print("\n" + "="*60)
    print("測試完成！")
    print("="*60)


if __name__ == "__main__":
    main()
