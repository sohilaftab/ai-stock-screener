import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from audit_tracker import (
    clean_symbol,
    log_research_cycle,
    send_verified_trade_alert,
    trade_signature,
)
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


def is_premarket_window(now_ist: datetime) -> bool:
    mins = now_ist.hour * 60 + now_ist.minute
    return (8 * 60 + 30) <= mins < (9 * 60 + 15)


def is_entry_window_open(now_ist: datetime) -> bool:
    mins = now_ist.hour * 60 + now_ist.minute
    # Allow new entries strictly from 09:15 AM to 03:05 PM IST (10 min buffer before 15:15 square-off)
    return (9 * 60 + 15) <= mins <= (15 * 60 + 5)


def _today_skip_reasons(engine: AdaptiveTradeEngine, today_str: str) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for row in getattr(engine, "shadow_ledger", []) or []:
        if str(row.get("date", ""))[:10] == today_str:
            sym = clean_symbol(row.get("symbol"))
            if sym:
                reasons[sym] = str(row.get("skip_reason") or row.get("reason") or "Skipped by risk/conviction guard")
    return reasons


def scan_and_score_market(
    engine: AdaptiveTradeEngine,
    scanner: MarketDataScanner,
    sentiment_ai: FinBERTSentimentEngine,
):
    """Scans NSE movers, enriches technicals + FinBERT news sentiment, and scores via PulseAI."""
    raw_df = (
        scanner.scan_market_movers()
        if hasattr(scanner, "scan_market_movers")
        else scanner.get_top_movers()
    )
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if hasattr(sentiment_ai, "analyze_candidates"):
        enriched_df = sentiment_ai.analyze_candidates(raw_df)
    elif hasattr(sentiment_ai, "score_dataframe"):
        enriched_df = sentiment_ai.score_dataframe(raw_df)
    else:
        enriched_rows = []
        for _, row in raw_df.iterrows():
            rec = row.to_dict()
            sym = rec["symbol"]
            raw_sent = "NEUTRAL"
            ai_conf = 50.0
            top_headline = f"{sym} intraday momentum & volume activity"
            try:
                headlines = (
                    sentiment_ai.fetch_google_news(sym)
                    if hasattr(sentiment_ai, "fetch_google_news")
                    else []
                )
                if headlines:
                    top_headline = headlines[0] if isinstance(headlines[0], str) else str(headlines[0])
                if hasattr(sentiment_ai, "analyze_stock_sentiment"):
                    s_res = sentiment_ai.analyze_stock_sentiment(sym, headlines)
                    if isinstance(s_res, dict):
                        raw_sent = str(s_res.get("sentiment", s_res.get("label", "NEUTRAL"))).upper()
                        ai_conf = float(s_res.get("confidence", s_res.get("score", 50.0)))
                        if ai_conf <= 1.0:
                            ai_conf *= 100.0
                        top_headline = str(s_res.get("headline", s_res.get("top_headline", top_headline)))
            except Exception:
                pass
            rec["raw_sentiment"] = raw_sent
            rec["AI Confidence"] = round(max(10.0, min(99.0, ai_conf)), 1)
            rec["Why Is It Moving? (Latest News)"] = top_headline
            enriched_rows.append(rec)
        enriched_df = pd.DataFrame(enriched_rows)

    premarket_df = scanner.fetch_premarket_data() if hasattr(scanner, "fetch_premarket_data") else pd.DataFrame()
    rec_df = engine.generate_intraday_recommendations(enriched_df, premarket_df=premarket_df, top_k=5)
    return enriched_df, rec_df


def _evaluate_positions_and_notify_exits(
    engine: AdaptiveTradeEngine,
    notifier: TelegramNotifier,
) -> dict:
    """
    Evaluate open positions and dispatch Telegram EXIT alerts ONLY for trades
    verified as closed on disk in trade_history.json.
    """
    open_before = {
        trade_signature(t)
        for t in (engine.ledger or [])
        if str(t.get("status", "")).upper() == "OPEN"
    }

    eval_res = engine.evaluate_open_trades_and_learn(notify_telegram=False)

    for trade in (engine.ledger or []):
        sig = trade_signature(trade)
        if sig in open_before and str(trade.get("status", "")).upper() != "OPEN":
            send_verified_trade_alert(notifier, trade, event_type="EXIT")

    return eval_res


def _record_and_notify_entries(
    engine: AdaptiveTradeEngine,
    rec_df: pd.DataFrame,
    notifier: TelegramNotifier,
) -> list[dict]:
    """
    Record qualifying recommendations into trade_history.json and dispatch Telegram
    ENTRY alerts strictly for trades verified on disk.
    """
    before_len = len(engine.ledger or [])
    if rec_df is not None and not rec_df.empty:
        engine.record_recommendations(rec_df, notify_telegram=False)

    new_trades = (engine.ledger or [])[before_len:]
    for trade in new_trades:
        send_verified_trade_alert(notifier, trade, event_type="ENTRY")
    return new_trades


def run_autonomous_cycle(
    engine: AdaptiveTradeEngine,
    scanner: MarketDataScanner,
    sentiment_ai: FinBERTSentimentEngine,
    notifier: TelegramNotifier,
):
    """
    Unified AI Decision Loop:
    1. Evaluates all OPEN positions against targets, stop-loss, giveback rules, and 15:15 IST square-off.
    2. Pre-market (08:30-09:14 IST): updates watchlist + research log silently (no Telegram BUY/SELL spam).
    3. Entry window (09:15-15:05 IST): scans market, logs ALL researched stocks to ai_research_log.json,
       enters only approved trades, and notifies Telegram strictly on executed entries/exits.
    4. Post-15:05 IST: manages/squares off open trades only and writes a clean heartbeat.
    """
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    print(f"[PULSEAI] Running autonomous cycle at {now_ist.strftime('%Y-%m-%d %I:%M:%S %p IST')}")

    eval_res = _evaluate_positions_and_notify_exits(engine, notifier)
    if eval_res.get("message"):
        print(f"[PULSEAI] Learning status: {eval_res['message']}")

    if is_entry_window_open(now_ist):
        perf = engine.get_performance_metrics()
        if perf["available_cash"] >= 1000 and perf["open_trades"] < 5:
            enriched_df, rec_df = scan_and_score_market(engine, scanner, sentiment_ai)
            new_trades = _record_and_notify_entries(engine, rec_df, notifier)
            executed_syms = {clean_symbol(t.get("symbol")) for t in new_trades}
            skip_reasons = _today_skip_reasons(engine, today_str)

            log_research_cycle(
                candidates=enriched_df,
                rec_candidates=rec_df,
                executed_symbols=executed_syms,
                skip_reasons=skip_reasons,
                cycle_mode="autonomous_cycle",
                market_note="Live scan completed inside 09:15-15:05 IST entry window.",
            )
            print(f"[PULSEAI] AI autonomously entered {len(new_trades)} verified trade(s).")
        else:
            note = (
                f"Active cap reached ({perf['open_trades']}/5 open, "
                f"free cash Rs.{perf['available_cash']:,.0f}); monitoring open positions."
            )
            log_research_cycle(candidates=[], cycle_mode="autonomous_cycle", market_note=note)
            print(f"[PULSEAI] {note}")

    elif is_premarket_window(now_ist):
        enriched_df, rec_df = scan_and_score_market(engine, scanner, sentiment_ai)
        engine.save_morning_watchlist(rec_df)
        log_research_cycle(
            candidates=enriched_df,
            rec_candidates=rec_df,
            executed_symbols=set(),
            skip_reasons=_today_skip_reasons(engine, today_str),
            cycle_mode="premarket_research",
            market_note="Pre-market research saved to dashboard watchlist (Telegram stays silent until execution).",
        )
        print("[PULSEAI] Pre-market research updated in webapp (no unexecuted Telegram alerts sent).")

    else:
        note = "Outside 09:15-15:05 IST entry window; managed/squared off open positions only."
        log_research_cycle(candidates=[], cycle_mode="manage_only", market_note=note)
        print(f"[PULSEAI] {note}")


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
    today_str = now_ist.strftime("%Y-%m-%d")

    if mode in ("autonomous_cycle", "live_manage", "intraday"):
        run_autonomous_cycle(engine, scanner, sentiment_ai, notifier)

    elif mode in ("morning_prep", "premarket", "premarket_update"):
        enriched_df, rec_df = scan_and_score_market(engine, scanner, sentiment_ai)
        if mode != "morning_prep":
            engine.compare_with_morning_watchlist(rec_df)
        engine.save_morning_watchlist(rec_df)

        new_trades = []
        if mode != "morning_prep" and is_entry_window_open(now_ist):
            perf = engine.get_performance_metrics()
            if perf["available_cash"] >= 1000 and perf["open_trades"] < 5:
                new_trades = _record_and_notify_entries(engine, rec_df, notifier)

        log_research_cycle(
            candidates=enriched_df,
            rec_candidates=rec_df,
            executed_symbols={clean_symbol(t.get("symbol")) for t in new_trades},
            skip_reasons=_today_skip_reasons(engine, today_str),
            cycle_mode=mode,
            market_note=f"{mode} research logged to dashboard; Telegram alerts restricted to executed trades.",
        )

    elif mode == "eod":
        result = _evaluate_positions_and_notify_exits(engine, notifier)
        ml_info = engine.ml_model.train_walk_forward(engine.ledger, engine.shadow_ledger)
        perf = engine.get_performance_metrics()
        log_research_cycle(
            candidates=[],
            cycle_mode="eod",
            market_note=f"EOD settlement complete. Equity Rs.{perf['current_equity']:,.2f}, Open={perf['open_trades']}.",
        )
        notifier.send_message(
            f"📊 <b>DAILY PORTFOLIO &amp; AI LEARNING SNAPSHOT</b>\n"
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