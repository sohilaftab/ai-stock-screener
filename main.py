import os
import sys
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from market_data import MarketDataScanner
from sentiment_engine import FinBERTSentimentEngine
from telegram_notifier import TelegramNotifier
from trade_engine import AdaptiveTradeEngine

IST = ZoneInfo("Asia/Kolkata")

# Official NSE Equity Trading Holidays (YYYY-MM-DD)
NSE_HOLIDAYS = {
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
    "2027-01-26",  # Republic Day
    "2027-03-08",  # Mahashivratri
    "2027-03-22",  # Holi
    "2027-03-26",  # Good Friday
    "2027-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2027-05-01",  # Maharashtra Day
    "2027-08-15",  # Independence Day
    "2027-10-02",  # Mahatma Gandhi Jayanti
}


def is_nse_open_day(now_ist: datetime) -> bool:
    today_str = now_ist.strftime("%Y-%m-%d")
    if (
        now_ist.weekday() >= 5
        or today_str in NSE_HOLIDAYS
        or not MarketDataScanner.is_nse_trading_day(now_ist.date())
    ):
        print(f"[INFO] NSE is closed today ({today_str} Weekend/Holiday). Exiting cleanly.")
        return False
    return True


def is_live_market_hours(now_ist: datetime) -> bool:
    mins = now_ist.hour * 60 + now_ist.minute
    # NSE continuous trading session: 09:15 AM to 03:25 PM IST
    return (9 * 60 + 15) <= mins <= (15 * 60 + 25)


def scan_and_score_market(
    engine: AdaptiveTradeEngine,
    scanner: MarketDataScanner,
    sentiment_ai: FinBERTSentimentEngine,
):
    """Scans NSE movers, enriches technicals + FinBERT news sentiment, and scores via PulseAI."""
    raw_df = scanner.scan_market_movers() if hasattr(scanner, "scan_market_movers") else scanner.get_top_movers()
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    enriched_df = (
        sentiment_ai.analyze_candidates(raw_df)
        if hasattr(sentiment_ai, "analyze_candidates")
        else sentiment_ai.enrich_with_sentiment(raw_df)
    )
    premarket_df = scanner.fetch_premarket_data() if hasattr(scanner, "fetch_premarket_data") else pd.DataFrame()

    rec_df = engine.generate_intraday_recommendations(enriched_df, premarket_df=premarket_df, top_k=5)
    return enriched_df, rec_df


def run_autonomous_cycle(
    engine: AdaptiveTradeEngine,
    scanner: MarketDataScanner,
    sentiment_ai: FinBERTSentimentEngine,
    notifier: TelegramNotifier,
):
    """
    Unified AI Decision Loop:
    1. Evaluates all OPEN positions against live ticks, news emotion shifts, and profit-giveback rules.
       Any closed position immediately triggers post-trade research and weight updates.
    2. During live market hours, scans for fresh setups and enters ONLY if conviction, risk,
       microstructure, and session rules approve.
    """
    now_ist = datetime.now(IST)
    print(f"[PULSEAI] Running autonomous cycle at {now_ist.strftime('%Y-%m-%d %I:%M:%S %p IST')}")

    actions = engine.manage_live_positions(notify_telegram=True)
    if actions:
        print(f"[PULSEAI] Executed {len(actions)} live position action(s).")

    if is_live_market_hours(now_ist):
        perf = engine.get_performance_metrics()
        if perf["available_cash"] >= 1000 and perf["open_trades"] < 5:
            _, rec_df = scan_and_score_market(engine, scanner, sentiment_ai)
            if rec_df is not None and not rec_df.empty:
                added = engine.record_recommendations(rec_df, notify_telegram=True)
                print(f"[PULSEAI] AI autonomously entered {added} new trade(s).")
            else:
                print("[PULSEAI] AI chose to wait — no setup met conviction/risk criteria.")
    else:
        print("[PULSEAI] Outside live 09:15–15:25 IST trading window; skipping new entries.")


def main():
    now_ist = datetime.now(IST)
    if not is_nse_open_day(now_ist):
        sys.exit(0)

    mode = os.getenv("RUN_MODE", "autonomous_cycle").strip().lower()
    if len(sys.argv) > 1 and sys.argv[1].strip():
        mode = sys.argv[1].strip().lower()

    engine = AdaptiveTradeEngine()
    scanner = MarketDataScanner()
    sentiment_ai = FinBERTSentimentEngine()
    notifier = TelegramNotifier()

    if mode in ("autonomous_cycle", "live_manage", "intraday"):
        run_autonomous_cycle(engine, scanner, sentiment_ai, notifier)

    elif mode in ("morning_prep", "premarket", "premarket_update"):
        _, rec_df = scan_and_score_market(engine, scanner, sentiment_ai)
        if mode == "morning_prep":
            engine.save_morning_watchlist(rec_df)
            if rec_df is not None and not rec_df.empty:
                lines = [
                    f"• <code>{r['symbol']}</code> ({r['action']}) — Conviction {r['conviction']}/100 @ ₹{r['entry_price']}"
                    for _, r in rec_df.iterrows()
                ]
                notifier.send_message(
                    "🌅 <b>PRE-MARKET AI RESEARCH WATCHLIST</b>\n━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(lines)
                )
        else:
            changes = engine.compare_with_morning_watchlist(rec_df)
            engine.save_morning_watchlist(rec_df)
            if changes:
                notifier.send_message("⚡ <b>PRE-OPEN MARKET SHIFTS</b>\n━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(changes))
            if is_live_market_hours(now_ist) and rec_df is not None and not rec_df.empty:
                engine.record_recommendations(rec_df, notify_telegram=True)

    elif mode == "eod":
        result = engine.evaluate_open_trades_and_learn(notify_telegram=True)
        ml_info = engine.ml_model.train_walk_forward(engine.ledger, engine.shadow_ledger)
        perf = engine.get_performance_metrics()
        notifier.send_message(
            f"📊 <b>DAILY PORTFOLIO & AI LEARNING SNAPSHOT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 <b>Total Wallet Equity:</b> <b>₹{perf['current_equity']:,.2f}</b> ({perf['total_pnl_pct']:+.2f}%)\n"
            f"💵 <b>Available Cash:</b> ₹{perf['available_cash']:,.2f} | <b>Open Positions:</b> {perf['open_trades']}\n"
            f"🎯 <b>Win Rate:</b> {perf['win_rate']}% ({perf['settled_trades']} settled)\n"
            f"🤖 <b>Walk-Forward ML:</b> {ml_info.get('samples_count', 0)}/{ml_info.get('min_required', 60)} samples (AUC {ml_info.get('wf_auc', 0.5):.2f})\n"
            f"🧠 <b>AI Evolution:</b> <i>{result['message']}</i>"
        )

    else:
        print(f"[WARN] Unknown mode '{mode}', defaulting to autonomous_cycle.")
        run_autonomous_cycle(engine, scanner, sentiment_ai, notifier)


if __name__ == "__main__":
    main()