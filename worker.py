import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from main import is_live_market_hours, is_nse_open_day, run_autonomous_cycle
from market_data import MarketDataScanner
from sentiment_engine import FinBERTSentimentEngine
from telegram_notifier import TelegramNotifier
from trade_engine import AdaptiveTradeEngine

IST = ZoneInfo("Asia/Kolkata")
CYCLE_SECONDS = 300  # 5-minute heartbeat

STATE_FILES = [
    "portfolio_state.json",
    "trade_history.json",
    "model_weights.json",
    "morning_watchlist.json",
    "shadow_ledger.json",
    "sector_cache.json",
    "ml_feature_log.json",
]


def _has_open_trades(ledger_path: str = "trade_history.json") -> bool:
    import json
    import os

    if not os.path.exists(ledger_path):
        return False
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return any(str(t.get("status", "")).upper() == "OPEN" for t in data) if isinstance(data, list) else False
    except Exception:
        return False


def pull_latest_from_git():
    """Pulls latest remote state before running or committing so local and GitHub stay aligned."""
    try:
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)
    except Exception as e:
        print(f"[WORKER GIT PULL WARN] {e}")


def sync_state_to_git():
    """Pulls, commits, and pushes updated JSON state files back to GitHub so Streamlit Cloud stays synced."""
    try:
        pull_latest_from_git()
        existing_files = [f for f in STATE_FILES if subprocess.run(["git", "ls-files", "--error-unmatch", f], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0 or True]
        subprocess.run(["git", "add", *existing_files], check=False)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        if diff.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", "chore(worker): sync warm-worker state to GitHub [skip ci]"],
                check=False,
            )
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)
            push_res = subprocess.run(["git", "push", "origin", "main"], check=False)
            if push_res.returncode == 0:
                print("[WORKER GIT SYNC] Pushed updated state to origin/main.")
    except Exception as e:
        print(f"[WORKER GIT SYNC WARN] {e}")


def run_warm_worker():
    print("[WARM WORKER] Pre-loading FinBERT model & NSE session cookies in memory...")
    scanner = MarketDataScanner()
    sentiment_ai = FinBERTSentimentEngine()
    notifier = TelegramNotifier()
    print("[WARM WORKER] Ready. Zero-boot-delay loop started.")

    post_close_reconciled_date = ""

    while True:
        now_ist = datetime.now(IST)
        today_str = now_ist.strftime("%Y-%m-%d")
        mins_now = now_ist.hour * 60 + now_ist.minute
        open_day = is_nse_open_day(now_ist)
        live_window = is_live_market_hours(now_ist)

        if open_day and live_window:
            t0 = time.time()
            try:
                pull_latest_from_git()
                engine = AdaptiveTradeEngine()
                run_autonomous_cycle(engine, scanner, sentiment_ai, notifier)
                sync_state_to_git()
                print(f"[WARM WORKER] Cycle completed in {time.time() - t0:.2f}s.")
            except Exception as e:
                print(f"[WARM WORKER ERROR] {e}")
        elif open_day and mins_now >= (15 * 60 + 15) and post_close_reconciled_date != today_str:
            # Ensures open intraday trades are squared off and pushed to GitHub even if worker started after 15:15 IST
            try:
                pull_latest_from_git()
                engine = AdaptiveTradeEngine()
                if _has_open_trades("trade_history.json"):
                    print("[WARM WORKER] Post-15:15 IST open positions found; running square-off & learning...")
                    engine.evaluate_open_trades_and_learn(notify_telegram=True)
                    sync_state_to_git()
                post_close_reconciled_date = today_str
            except Exception as e:
                print(f"[WARM WORKER POST-CLOSE ERROR] {e}")
        else:
            print(f"[WARM WORKER] Market closed ({now_ist.strftime('%I:%M %p IST')}). Sleeping 5m...")

        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    run_warm_worker()