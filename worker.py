"""
PulseAI warm worker - one long-running process (VPS / Docker / systemd) instead of cron'd CI jobs.

Resilience features
-------------------
* Exponential back-off after consecutive failures (instead of hammering a broken data source).
* Fail-closed on a corrupt state file: stops trading, alerts, keeps retrying slowly.
* Rate-limited Telegram alerts for repeated failures, stale cycles and git-sync problems.
* Git sync with timeouts, rebase-abort on conflict, throttling, and ALL state files (incl. ML log/model).
* Runs the end-of-day learning job itself once the market closes (once per day).
* Graceful shutdown on SIGTERM/SIGINT.

Run exactly ONE writer (this worker OR the GitHub Action) to avoid state-file merge conflicts.
"""
import os
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

from health import HealthMonitor
from main import is_live_market_hours, is_nse_open_day, run_autonomous_cycle, run_eod
from market_data import MarketDataScanner
from sentiment_engine import FinBERTSentimentEngine
from state_store import StateCorruptionError
from telegram_notifier import TelegramNotifier
from trade_engine import AdaptiveTradeEngine

IST = ZoneInfo("Asia/Kolkata")
CYCLE_SECONDS = int(os.getenv("PULSEAI_CYCLE_SECONDS", "300"))
GIT_SYNC_ENABLED = os.getenv("PULSEAI_GIT_SYNC", "1") == "1"
GIT_SYNC_MIN_GAP_SEC = 600
STALE_CYCLE_ALERT_SEC = 20 * 60
EOD_START_MIN = 15 * 60 + 35

STATE_FILES = [
    "portfolio_state.json", "trade_history.json", "model_weights.json", "morning_watchlist.json",
    "shadow_ledger.json", "sector_cache.json", "ml_feature_log.json", "ml_context_model.pkl",
]

_stop = False


def _handle_stop(signum, _frame):
    global _stop
    print(f"[WORKER] Signal {signum} received - finishing current step then exiting.")
    _stop = True


def _git(*args, timeout=60):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as e:
        print(f"[WORKER GIT WARN] git {' '.join(args)} failed: {e}")
        return None


def sync_state_to_git(health: HealthMonitor, last_sync: dict, force: bool = False) -> bool:
    """Commit + push state. Never raises; never leaves the repo mid-rebase."""
    if not GIT_SYNC_ENABLED:
        return False
    if not force and time.time() - last_sync.get("ts", 0.0) < GIT_SYNC_MIN_GAP_SEC:
        return False

    existing = [f for f in STATE_FILES if os.path.exists(f)]
    if not existing:
        return False
    _git("add", "--", *existing)
    diff = _git("diff", "--cached", "--quiet")
    if diff is None or diff.returncode == 0:
        return False   # nothing staged (or git unavailable)

    _git("commit", "-m", "chore(worker): sync warm-worker state to GitHub")
    for attempt in range(2):
        pull = _git("pull", "--rebase", "--autostash", "-X", "theirs", "origin", "main", timeout=120)
        if pull is None or pull.returncode != 0:
            _git("rebase", "--abort")
            health.alert("git-pull", f"⚠️ PulseAI: git pull --rebase failed (attempt {attempt + 1}); state NOT pushed.")
            continue
        push = _git("push", "origin", "main", timeout=120)
        if push is not None and push.returncode == 0:
            last_sync["ts"] = time.time()
            health.record_success("git-sync")
            return True
    health.record_failure("git-sync", "push failed after retries")
    return False


def run_warm_worker():
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    notifier = TelegramNotifier()
    health = HealthMonitor(notifier)
    print("[WARM WORKER] Pre-loading FinBERT model & NSE session cookies in memory...")
    scanner = MarketDataScanner()
    sentiment_ai = FinBERTSentimentEngine(eager_load=True)
    last_sync = {"ts": 0.0}
    eod_done_for = None
    heartbeat_sent_for = None
    consecutive_failures = 0
    last_ok_cycle = time.time()
    print("[WARM WORKER] Ready.")

    while not _stop:
        now_ist = datetime.now(IST)
        today = now_ist.strftime("%Y-%m-%d")
        mins = now_ist.hour * 60 + now_ist.minute
        sleep_for = CYCLE_SECONDS

        try:
            if is_nse_open_day(now_ist):
                if is_live_market_hours(now_ist):
                    if heartbeat_sent_for != today:
                        notifier.send_message("🟢 <b>PulseAI worker online</b> - market session started.")
                        heartbeat_sent_for = today
                    t0 = time.time()
                    engine = AdaptiveTradeEngine()           # re-reads state each cycle (fail-closed if corrupt)
                    run_autonomous_cycle(engine, scanner, sentiment_ai, notifier)
                    sync_state_to_git(health, last_sync)
                    consecutive_failures = 0
                    last_ok_cycle = time.time()
                    health.record_success("cycle")
                    print(f"[WARM WORKER] Cycle completed in {time.time() - t0:.2f}s.")
                elif mins >= EOD_START_MIN and eod_done_for != today:
                    engine = AdaptiveTradeEngine()
                    run_eod(engine, notifier)
                    sync_state_to_git(health, last_sync, force=True)
                    eod_done_for = today
                else:
                    print(f"[WARM WORKER] Outside session ({now_ist.strftime('%I:%M %p IST')}).")
            else:
                print("[WARM WORKER] NSE closed today.")

            if is_live_market_hours(now_ist) and time.time() - last_ok_cycle > STALE_CYCLE_ALERT_SEC:
                health.alert("stale", f"⚠️ PulseAI: no successful cycle for {int((time.time() - last_ok_cycle) / 60)} min during market hours.")

        except StateCorruptionError as e:
            # Fail closed: do not trade, do not overwrite. Retry slowly in case the file is repaired.
            health.alert("state-corrupt", f"🛑 <b>PulseAI HALTED</b>: state file corrupt, no valid backup.\n{str(e)[:300]}")
            sleep_for = 600
        except Exception as e:
            consecutive_failures += 1
            health.record_failure("cycle", f"{type(e).__name__}: {e}")
            sleep_for = min(600, CYCLE_SECONDS * (2 ** min(consecutive_failures, 3)) // 2)
            print(f"[WARM WORKER ERROR] {type(e).__name__}: {e} (retry in {sleep_for}s)")

        # sleep in small slices so SIGTERM is honoured quickly
        slept = 0
        while slept < sleep_for and not _stop:
            time.sleep(min(5, sleep_for - slept))
            slept += 5

    print("[WARM WORKER] Stopped cleanly.")


if __name__ == "__main__":
    run_warm_worker()