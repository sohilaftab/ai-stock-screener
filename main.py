import argparse
from datetime import datetime
from dotenv import load_dotenv
from nse_client import NSEClient
from market_data import MarketDataScanner
from sentiment_engine import FinBERTSentimentEngine
from telegram_notifier import TelegramNotifier
from trade_engine import AdaptiveTradeEngine

load_dotenv()


def enrich_dataframe_cli(df, sentiment_ai):
    if df.empty:
        return df
    raw_labels, confs, headlines = [], [], []
    for _, row in df.iterrows():
        res = sentiment_ai.analyze_stock_sentiment(row["symbol"])
        raw_labels.append(res["sentiment"])
        confs.append(int(res["confidence"] * 100))
        headlines.append(res["top_headline"])
    enriched = df.copy()
    enriched["raw_sentiment"] = raw_labels
    enriched["AI Confidence"] = confs
    enriched["Why Is It Moving? (Latest News)"] = headlines
    return enriched


def run_morning_prep_workflow(notifier):
    """08:30 AM IST (Before 9 AM): Scans overnight news + multi-timeframe charts to build morning watchlist."""
    nse = NSEClient()
    tickers = nse.get_nifty_universe_tickers()
    scanner = MarketDataScanner(tickers)
    candidates, _ = scanner.scan_volume_breakouts(vol_multiplier=1.2, min_price_change_pct=0.8, top_n=10)

    if candidates.empty:
        return

    sentiment_ai = FinBERTSentimentEngine()
    enriched = enrich_dataframe_cli(candidates, sentiment_ai)

    engine = AdaptiveTradeEngine()
    recs = engine.generate_intraday_recommendations(enriched, top_k=5)
    engine.save_morning_watchlist(recs)

    perf = engine.get_performance_metrics()
    lines = [
        f"🌅 <b>PRE-9 AM MORNING RESEARCH & WATCHLIST ({datetime.now().strftime('%d %b')})</b>",
        f"💰 <b>Autonomous Wallet:</b> ₹{perf['current_equity']:,.0f} (Available: ₹{perf['available_cash']:,.0f})\n"
    ]
    for _, r in recs.iterrows():
        icon = "🟢 BUY" if r["action"] == "BUY" else "🔴 SHORT"
        lines.append(
            f"{icon} <b>{r['symbol']}</b> | Chart: <code>{r['chart_trend']}</code> (Score: <b>{r['conviction']}/100</b>)\n"
            f"   • Plan: Qty <b>{r['qty']}</b> (₹{r['capital_used']:,.0f}) @ ₹{r['entry_price']} | T1: ₹{r['target_1']} | SL: ₹{r['stop_loss']}\n"
            f"   • News ({r['sentiment']}): {r['headline']}\n"
        )
    lines.append("⏳ <i>Next update at 09:10 AM after NSE Pre-Market order book confirmation.</i>")
    notifier.send_message("\n".join(lines))


def run_premarket_update_workflow(notifier):
    """09:10 AM IST: Checks NSE Pre-Market, detects changes vs Pre-9 AM list, and executes virtual trades."""
    nse = NSEClient()
    pre_df = nse.get_pre_open_movers(key="FO", min_gap_pct=1.5, min_volume=25000, top_n=8)
    tickers = nse.get_nifty_universe_tickers()
    scanner = MarketDataScanner(tickers)
    candidates, _ = scanner.scan_volume_breakouts(vol_multiplier=1.2, min_price_change_pct=0.8, top_n=10)

    sentiment_ai = FinBERTSentimentEngine()
    enriched = enrich_dataframe_cli(candidates, sentiment_ai)

    engine = AdaptiveTradeEngine()
    recs = engine.generate_intraday_recommendations(enriched, premarket_df=pre_df, top_k=5)
    changes = engine.compare_with_morning_watchlist(recs)
    logged_count = engine.record_recommendations(recs)
    perf = engine.get_performance_metrics()

    lines = [f"🚀 <b>9:10 AM PRE-MARKET UPDATE & VIRTUAL EXECUTION</b>\n"]
    if changes:
        lines.append("🔄 <b>Momentum Shifts vs Pre-9 AM List:</b>")
        for c in changes[:4]:
            lines.append(f"• {c}")
        lines.append("")

    lines.append(f"💼 <b>Executed {logged_count} Trades | Allocated: ₹{perf['allocated_capital']:,.0f} / ₹{perf['current_equity']:,.0f}</b>\n")
    for _, r in recs.iterrows():
        icon = "🟢 BUY" if r["action"] == "BUY" else "🔴 SHORT"
        lines.append(
            f"{icon} <b>{r['symbol']}</b> — Qty: <b>{r['qty']}</b> (₹{r['capital_used']:,.0f})\n"
            f"   • Entry: <b>₹{r['entry_price']}</b> | T1: <b>₹{r['target_1']}</b> | T2: <b>₹{r['target_2']}</b> | SL: <b>₹{r['stop_loss']}</b>\n"
            f"   • Max Risk: -₹{r['max_loss_inr']:,.0f} | Reward: +₹{r['expected_profit_inr']:,.0f}\n"
        )
    notifier.send_message("\n".join(lines))


def run_live_manage_workflow(notifier):
    """During Market Hours: Monitors active positions, trails SLs, extends targets, cuts losers early."""
    engine = AdaptiveTradeEngine()
    open_trades = [t for t in engine.ledger if t.get("status") == "OPEN"]

    # If no trades were opened yet today, generate and open them
    if not open_trades:
        run_premarket_update_workflow(notifier)
        return

    actions = engine.manage_live_positions()
    perf = engine.get_performance_metrics()

    now_str = datetime.now().strftime("%I:%M %p")
    lines = [
        f"📡 <b>LIVE AUTONOMOUS TRADE MANAGER ({now_str})</b>",
        f"💰 <b>Live Equity:</b> ₹{perf['current_equity'] + perf['unrealized_pnl_inr']:,.0f} (Unrealized: ₹{perf['unrealized_pnl_inr']:+,.0f})\n"
    ]
    if actions:
        lines.append("⚡ <b>Dynamic Trade Adjustments Made:</b>")
        for act in actions:
            lines.append(f"• {act}")
        lines.append("")
    else:
        lines.append("✅ <i>All open positions holding within normal target/SL bands.</i>\n")

    still_open = [t for t in engine.ledger if t.get("status") == "OPEN"]
    for t in still_open:
        lines.append(
            f"• <b>{t['symbol']} ({t['action']})</b>: LTP ₹{t.get('current_price', t['entry_price'])} | "
            f"Active T1: ₹{t['target_1']} | Active SL: ₹{t['stop_loss']} | "
            f"Live P&L: <b>₹{t.get('unrealized_pnl_inr', 0):+,.0f}</b>"
        )
    notifier.send_message("\n".join(lines))


def run_eod_workflow(notifier):
    """03:35 PM / 04:15 PM IST: Squares off remaining trades, runs Mistake Autopsy, updates Rs 1.5L wallet."""
    engine = AdaptiveTradeEngine()
    eval_result = engine.evaluate_open_trades_and_learn()
    perf = engine.get_performance_metrics()

    lines = [
        f"🏁 <b>EOD PORTFOLIO AUTOPSY & ₹1.5L WALLET REPORT</b>\n",
        f"💼 <b>Starting Capital:</b> ₹{perf['initial_capital']:,.0f}",
        f"🏦 <b>Current Wallet Equity:</b> <b>₹{perf['current_equity']:,.0f}</b> ({perf['total_pnl_pct']:+.2f}%)",
        f"📊 <b>Net Realized Profit/Loss:</b> <b>₹{perf['total_pnl_inr']:+,.0f}</b>",
        f"🎯 <b>Win Rate:</b> {perf['win_rate']}% ({perf['settled_trades']} settled trades)\n",
        f"🧠 <b>Self-Learning Update:</b> <i>{eval_result['message']}</i>"
    ]
    mistakes = engine.portfolio.get("mistakes_identified", [])
    if mistakes:
        lines.append(f"\n🔍 <b>Latest Mistake Diagnosis:</b>\n• <i>{mistakes[0]}</i>")

    notifier.send_message("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Rs 1,50,000 AI Stock Trader")
    parser.add_argument(
        "--mode",
        choices=["morning_prep", "premarket_update", "live_manage", "intraday", "premarket", "eod"],
        default="live_manage"
    )
    args = parser.parse_args()

    telegram = TelegramNotifier()
    if args.mode == "morning_prep":
        run_morning_prep_workflow(telegram)
    elif args.mode in ["premarket_update", "premarket"]:
        run_premarket_update_workflow(telegram)
    elif args.mode in ["live_manage", "intraday"]:
        run_live_manage_workflow(telegram)
    elif args.mode == "eod":
        run_eod_workflow(telegram)