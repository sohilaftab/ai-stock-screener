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


def sync_state_to_git():
    """Commits and pushes updated JSON state files back to GitHub so Streamlit Cloud stays synced."""
    try:
        subprocess.run(
            [
                "git", "add",
                "portfolio_state.json", "trade_history.json", "model_weights.json",
                "morning_watchlist.json", "shadow_ledger.json", "sector_cache.json",
            ],
            check=False,
        )
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        if diff.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", "chore(worker): sync warm-worker state to GitHub"],
                check=False,
            )
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)
            subprocess.run(["git", "push", "origin", "main"], check=False)
    except Exception as e:
        print(f"[WORKER GIT SYNC WARN] {e}")


def run_warm_worker():
    print("[WARM WORKER] Pre-loading FinBERT model & NSE session cookies in memory...")
    scanner = MarketDataScanner()
    sentiment_ai = FinBERTSentimentEngine()
    notifier = TelegramNotifier()
    print("[WARM WORKER] Ready. Zero-boot-delay loop started.")

    while True:
        now_ist = datetime.now(IST)
        if is_nse_open_day(now_ist) and is_live_market_hours(now_ist):
            t0 = time.time()
            try:
                engine = AdaptiveTradeEngine()
                run_autonomous_cycle(engine, scanner, sentiment_ai, notifier)
                sync_state_to_git()
                print(f"[WARM WORKER] Cycle completed in {time.time() - t0:.2f}s.")
            except Exception as e:
                print(f"[WARM WORKER ERROR] {e}")
        else:
            print(f"[WARM WORKER] Market closed ({now_ist.strftime('%I:%M %p IST')}). Sleeping 5m...")

        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    run_warm_worker()