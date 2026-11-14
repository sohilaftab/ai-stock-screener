import argparse
from datetime import datetime
from dotenv import load_dotenv
from nse_client import NSEClient
from market_data import MarketDataScanner
from sentiment_engine import FinBERTSentimentEngine
from telegram_notifier import TelegramNotifier

load_dotenv()

def run_premarket_workflow(notifier):
    print("--- Running Phase 2: Pre-Market Scanner ---")
    nse = NSEClient()
    movers = nse.get_pre_open_movers(key="FO", min_gap_pct=2.0, min_volume=50000, top_n=5)
    
    if movers.empty:
        notifier.send_message("📊 <b>NSE Pre-Market (9:10 AM)</b>\nNo F&O stocks gapping > 2% with high volume today.")
        return

    sentiment_ai = FinBERTSentimentEngine()
    lines = ["🚀 <b>TOP 5 NSE PRE-MARKET MOVERS</b>\n"]

    for _, row in movers.iterrows():
        sym = row["symbol"]
        news_data = sentiment_ai.analyze_stock_sentiment(sym)
        direction_emoji = "🟢" if row["pChange"] > 0 else "🔴"
        
        lines.append(
            f"{direction_emoji} <b>{sym}</b> | IEP: ₹{row['iep']} (<b>{row['pChange']:+.2f}%</b>)\n"
            f"   • Pre-Open Vol: {row['volume']:,} | Buy/Sell Ratio: {row['buy_sell_ratio']}x\n"
            f"   • AI Sentiment: <b>{news_data['sentiment']}</b> ({news_data['top_headline']})\n"
        )

    notifier.send_message("\n".join(lines))

def run_intraday_workflow(notifier):
    print("--- Running Phase 3 & 4: Intraday Volume Breakout + FinBERT ---")
    nse = NSEClient()
    tickers = nse.get_nifty_universe_tickers()
    
    scanner = MarketDataScanner(tickers)
    breakouts = scanner.scan_volume_breakouts(vol_multiplier=1.5, min_price_change_pct=1.5, top_n=5)

    if breakouts.empty:
        notifier.send_message("📡 <b>Intraday Volume Scan</b>\nNo stocks currently exceeding 1.5x 20-day Average Volume.")
        return

    sentiment_ai = FinBERTSentimentEngine()
    now_str = datetime.now().strftime("%I:%M %p")
    lines = [f"🔥 <b>INTRADAY VOLUME BREAKOUTS ({now_str})</b>\n"]

    for _, row in breakouts.iterrows():
        sym = row["symbol"]
        news_data = sentiment_ai.analyze_stock_sentiment(sym)
        
        lines.append(
            f"⚡ <b>{sym} trending:</b> Vol <b>{row['vol_surge_pct']}%</b> of avg ({row['vol_ratio']}x)\n"
            f"   • Price: ₹{row['ltp']} (<b>{row['change_pct']:+.2f}%</b>)\n"
            f"   • Sentiment: <b>{news_data['sentiment']}</b> ({news_data['top_headline']})\n"
        )

    notifier.send_message("\n".join(lines))

def run_eod_delivery_workflow(notifier):
    print("--- Running Phase 3 (EOD): Institutional Delivery Scanner ---")
    nse = NSEClient()
    tickers = nse.get_nifty_universe_tickers()
    scanner = MarketDataScanner(tickers)
    
    # First get top 15 volume movers of the day, then check NSE Delivery % for those
    top_vol_df = scanner.scan_volume_breakouts(vol_multiplier=1.2, min_price_change_pct=1.0, top_n=15)
    if top_vol_df.empty:
        return

    candidate_symbols = top_vol_df["symbol"].tolist()
    high_deliv_df = nse.get_eod_high_delivery_stocks(candidate_symbols, min_delivery_pct=60.0)

    if high_deliv_df.empty:
        notifier.send_message("🏦 <b>EOD Delivery Scan</b>\nNo volume breakout stocks recorded > 60% delivery today.")
        return

    lines = ["🏦 <b>INSTITUTIONAL ACCUMULATION (Delivery > 60%)</b>\n"]
    for _, row in high_deliv_df.iterrows():
        lines.append(
            f"✅ <b>{row['symbol']}</b> — Delivery: <b>{row['delivery_pct']}%</b> "
            f"(Traded Qty: {row['quantity_traded']:,})"
        )
    notifier.send_message("\n".join(lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Driven NSE Stock Screener")
    parser.add_argument(
        "--mode",
        choices=["premarket", "intraday", "eod"],
        default="intraday",
        help="Which screening phase to execute"
    )
    args = parser.parse_args()
    
    telegram = TelegramNotifier()
    if args.mode == "premarket":
        run_premarket_workflow(telegram)
    elif args.mode == "intraday":
        run_intraday_workflow(telegram)
    elif args.mode == "eod":
        run_eod_delivery_workflow(telegram)