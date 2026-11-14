import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from nse_client import NSEClient
from market_data import MarketDataScanner
from sentiment_engine import FinBERTSentimentEngine
from telegram_notifier import TelegramNotifier

load_dotenv()

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI NSE Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CACHED RESOURCES & DATA FETCHERS
# ==========================================
@st.cache_resource(show_spinner="Initializing NSE Session & Cookies...")
def get_nse_client():
    """Maintains a persistent NSE session across Streamlit reruns."""
    return NSEClient()

@st.cache_resource(show_spinner="Loading Local ProsusAI/FinBERT Model into RAM...")
def get_sentiment_engine():
    """Loads the Hugging Face FinBERT pipeline once into memory."""
    return FinBERTSentimentEngine()

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_universe():
    """Caches the Nifty 200 ticker list for 1 hour."""
    client = get_nse_client()
    return client.get_nifty_universe_tickers()

@st.cache_data(ttl=120, show_spinner="Fetching live NSE Pre-Market order book...")
def fetch_premarket_data(universe_key, min_gap, min_vol, top_n):
    client = get_nse_client()
    return client.get_pre_open_movers(
        key=universe_key,
        min_gap_pct=min_gap,
        min_volume=min_vol,
        top_n=top_n
    )

@st.cache_data(ttl=300, show_spinner="Scanning Nifty universe via Yahoo Finance batch API...")
def fetch_intraday_breakouts(vol_mult, min_price_chg, top_n):
    tickers = get_cached_universe()
    scanner = MarketDataScanner(tickers)
    return scanner.scan_volume_breakouts(
        vol_multiplier=vol_mult,
        min_price_change_pct=min_price_chg,
        top_n=top_n
    )

def enrich_with_finbert(df, symbol_col="symbol"):
    """Runs FinBERT sentiment analysis on every ticker in a filtered DataFrame."""
    if df.empty:
        return df
    
    engine = get_sentiment_engine()
    sentiments = []
    confidences = []
    headlines = []

    progress_bar = st.progress(0, text="Running FinBERT NLP on Google News RSS feeds...")
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        sym = row[symbol_col]
        res = engine.analyze_stock_sentiment(sym)
        
        # Add visual badge to sentiment label
        label = res["sentiment"]
        if label == "POSITIVE":
            badge = "🟢 POSITIVE"
        elif label == "NEGATIVE":
            badge = "🔴 NEGATIVE"
        else:
            badge = "⚪ NEUTRAL"

        sentiments.append(badge)
        confidences.append(int(res["confidence"] * 100))
        headlines.append(res["top_headline"])
        progress_bar.progress((idx + 1) / total, text=f"Analyzed news for {sym} ({idx + 1}/{total})")

    progress_bar.empty()
    enriched = df.copy()
    enriched["AI Sentiment"] = sentiments
    enriched["Confidence (%)"] = confidences
    enriched["Catalyst Headline"] = headlines
    return enriched

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("⚙️ Screener Controls")
    st.caption("Adjust real-time filtering parameters & AI settings.")

    if st.button("🔄 Refresh Market Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("1. Pre-Market Filters (9:00–9:15 AM)")
    pre_key = st.selectbox("NSE Segment", options=["FO", "NIFTY", "ALL"], index=0, help="FO = Futures & Options stocks (most liquid)")
    pre_min_gap = st.slider("Min Gap % (Up or Down)", 0.5, 10.0, 2.0, 0.5)
    pre_min_vol = st.number_input("Min Pre-Open Volume", min_value=5000, max_value=1000000, value=50000, step=10000)
    pre_top_n = st.slider("Max Pre-Market Stocks to Analyze", 3, 20, 8)

    st.divider()
    st.subheader("2. Intraday Breakout Filters")
    intra_vol_mult = st.slider("Volume Surge Multiplier (vs 20d Avg)", 1.0, 5.0, 1.5, 0.1)
    intra_min_chg = st.slider("Min Intraday Price Change (%)", 0.5, 10.0, 1.5, 0.25)
    intra_top_n = st.slider("Max Breakout Stocks to Analyze", 3, 25, 10)

    st.divider()
    st.subheader("3. EOD Delivery Filters")
    eod_min_deliv = st.slider("Min Institutional Delivery (%)", 40.0, 90.0, 60.0, 5.0)

    st.divider()
    notifier = TelegramNotifier()
    tg_status = "🟢 Connected" if (notifier.bot_token and notifier.chat_id) else "🔴 Missing .env Keys"
    st.markdown(f"**Telegram Bot Status:** {tg_status}")

# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.title("📈 AI-Driven NSE Stock Screener & FinBERT Terminal")
st.markdown(
    f"**Live Market Status:** `{datetime.now().strftime('%d %b %Y | %I:%M:%S %p IST')}`  •  "
    "**Data Sources:** `NSE Hidden JSON API` + `Yahoo Finance v8` + `Google News RSS`  •  "
    "**NLP Engine:** `ProsusAI/finbert`"
)

tab_intra, tab_pre, tab_eod, tab_deepdive = st.tabs([
    "🔥 Intraday Volume Breakouts",
    "🚀 Pre-Market Movers (9:00–9:15 AM)",
    "🏦 EOD Institutional Delivery",
    "🔍 Single-Stock FinBERT Inspector"
])

# ==========================================
# TAB 1: INTRADAY VOLUME BREAKOUTS
# ==========================================
with tab_intra:
    col_head, col_btn = st.columns([3, 1])
    with col_head:
        st.subheader("Unusual Volume & Momentum Breakouts")
        st.caption(f"Scanning Nifty 200 universe for Volume ≥ {intra_vol_mult}x of 20-Day Average and Price Change ≥ {intra_min_chg}%.")
    with col_btn:
        run_intra = st.button("⚡ Scan Intraday Breakouts", type="primary", use_container_width=True)

    if run_intra or "intraday_df" not in st.session_state:
        raw_intra = fetch_intraday_breakouts(intra_vol_mult, intra_min_chg, intra_top_n)
        st.session_state["intraday_df"] = enrich_with_finbert(raw_intra, symbol_col="symbol")

    intra_df = st.session_state.get("intraday_df", pd.DataFrame())

    if intra_df.empty:
        st.info("No stocks currently meet your volume multiplier and price breakout criteria. Try lowering the 'Volume Surge Multiplier' in the sidebar.")
    else:
        # Summary KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        top_vol_row = intra_df.iloc[0]
        bullish_count = len(intra_df[intra_df["AI Sentiment"].str.contains("POSITIVE")])
        bearish_count = len(intra_df[intra_df["AI Sentiment"].str.contains("NEGATIVE")])

        m1.metric("Breakout Stocks Found", len(intra_df))
        m2.metric("Top Volume Leader", top_vol_row["symbol"], f"{top_vol_row['vol_ratio']}x Avg Vol")
        m3.metric("Positive News Catalysts", bullish_count)
        m4.metric("Negative News Catalysts", bearish_count)

        # Styled Interactive Dataframe
        st.dataframe(
            intra_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Ticker", width="small"),
                "ltp": st.column_config.NumberColumn("LTP (₹)", format="₹%.2f"),
                "change_pct": st.column_config.NumberColumn("Price Chg %", format="%+.2f%%"),
                "live_volume": st.column_config.NumberColumn("Live Volume", format="%d"),
                "avg_volume": st.column_config.NumberColumn("20D Avg Vol", format="%d"),
                "vol_ratio": st.column_config.NumberColumn("Vol Multiplier", format="%.2fx"),
                "vol_surge_pct": st.column_config.NumberColumn("Vol % of Avg", format="%d%%"),
                "AI Sentiment": st.column_config.TextColumn("FinBERT Signal", width="small"),
                "Confidence (%)": st.column_config.ProgressColumn("AI Confidence", min_value=0, max_value=100, format="%d%%"),
                "Catalyst Headline": st.column_config.TextColumn("Top Google News Headline", width="large"),
            }
        )

        if st.button("📲 Push Intraday Breakouts to Telegram"):
            lines = [f"🔥 <b>INTRADAY VOLUME BREAKOUTS ({datetime.now().strftime('%I:%M %p')})</b>\n"]
            for _, r in intra_df.head(5).iterrows():
                lines.append(
                    f"⚡ <b>{r['symbol']}</b>: Vol <b>{r['vol_surge_pct']}%</b> of avg ({r['vol_ratio']}x) | "
                    f"₹{r['ltp']} ({r['change_pct']:+.2f}%)\n"
                    f"   • {r['AI Sentiment']} ({r['Confidence (%)']}%): {r['Catalyst Headline']}\n"
                )
            if notifier.send_message("\n".join(lines)):
                st.success("Alert sent to your Telegram phone!")
            else:
                st.error("Failed to send Telegram alert. Check your .env credentials.")

# ==========================================
# TAB 2: PRE-MARKET MOVERS
# ==========================================
with tab_pre:
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.subheader("NSE Pre-Open Session Anomalies (IEP Gap Scanner)")
        st.caption(f"Fetching directly from NSE hidden endpoint (`/api/market-data-pre-open?key={pre_key}`).")
    with col_p2:
        run_pre = st.button("🚀 Fetch Pre-Market Data", type="primary", use_container_width=True)

    if run_pre:
        raw_pre = fetch_premarket_data(pre_key, pre_min_gap, pre_min_vol, pre_top_n)
        st.session_state["premarket_df"] = enrich_with_finbert(raw_pre, symbol_col="symbol")

    pre_df = st.session_state.get("premarket_df", pd.DataFrame())

    if pre_df.empty:
        st.info("Click **Fetch Pre-Market Data** to pull the latest NSE Pre-Open session equilibrium prices and run FinBERT.")
    else:
        st.dataframe(
            pre_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="small"),
                "iep": st.column_config.NumberColumn("IEP (₹)", format="₹%.2f"),
                "prev_close": st.column_config.NumberColumn("Prev Close (₹)", format="₹%.2f"),
                "pChange": st.column_config.NumberColumn("Gap %", format="%+.2f%%"),
                "abs_change": None, # Hide helper sorting column
                "volume": st.column_config.NumberColumn("Pre-Open Vol", format="%d"),
                "buy_sell_ratio": st.column_config.NumberColumn("Buy/Sell Ratio", format="%.2fx"),
                "AI Sentiment": st.column_config.TextColumn("FinBERT Signal", width="small"),
                "Confidence (%)": st.column_config.ProgressColumn("AI Confidence", min_value=0, max_value=100, format="%d%%"),
                "Catalyst Headline": st.column_config.TextColumn("Top Google News Headline", width="large"),
            }
        )

        if st.button("📲 Push Pre-Market Movers to Telegram"):
            lines = ["🚀 <b>TOP NSE PRE-MARKET MOVERS</b>\n"]
            for _, r in pre_df.head(5).iterrows():
                lines.append(
                    f"• <b>{r['symbol']}</b> | IEP: ₹{r['iep']} (<b>{r['pChange']:+.2f}%</b>) | Buy/Sell: {r['buy_sell_ratio']}x\n"
                    f"  {r['AI Sentiment']}: {r['Catalyst Headline']}\n"
                )
            if notifier.send_message("\n".join(lines)):
                st.success("Pre-market alert delivered to Telegram!")

# ==========================================
# TAB 3: EOD INSTITUTIONAL DELIVERY
# ==========================================
with tab_eod:
    st.subheader("End-of-Day Institutional Accumulation (Delivery % Scanner)")
    st.caption("Queries NSE Equity Quote `trade_info` endpoint for high-volume breakout stocks with strong delivery holding.")

    if st.button("🏦 Check Delivery % for Top Volume Stocks", type="primary"):
        base_candidates = fetch_intraday_breakouts(vol_mult=1.2, min_price_chg=1.0, top_n=15)
        if base_candidates.empty:
            st.warning("No volume movers found to inspect.")
        else:
            client = get_nse_client()
            with st.spinner("Querying NSE Trade Info API for Delivery Percentages..."):
                deliv_df = client.get_eod_high_delivery_stocks(
                    base_candidates["symbol"].tolist(),
                    min_delivery_pct=eod_min_deliv
                )
                if not deliv_df.empty:
                    # Merge price and volume ratios from base_candidates
                    merged_eod = pd.merge(deliv_df, base_candidates, on="symbol", how="left")
                    st.session_state["eod_df"] = enrich_with_finbert(merged_eod, symbol_col="symbol")
                else:
                    st.session_state["eod_df"] = pd.DataFrame()

    eod_df = st.session_state.get("eod_df", pd.DataFrame())
    if not eod_df.empty:
        st.dataframe(
            eod_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol"),
                "delivery_pct": st.column_config.ProgressColumn("Delivery %", min_value=0, max_value=100, format="%.1f%%"),
                "quantity_traded": st.column_config.NumberColumn("NSE Traded Qty", format="%d"),
                "ltp": st.column_config.NumberColumn("Close Price (₹)", format="₹%.2f"),
                "change_pct": st.column_config.NumberColumn("Day Change %", format="%+.2f%%"),
                "vol_ratio": st.column_config.NumberColumn("Vol vs 20D Avg", format="%.2fx"),
                "AI Sentiment": st.column_config.TextColumn("FinBERT Signal"),
                "Catalyst Headline": st.column_config.TextColumn("Catalyst Headline", width="large"),
            }
        )
    else:
        st.info(f"Click the button above to scan today's volume leaders for NSE Delivery ≥ {eod_min_deliv}%.")

# ==========================================
# TAB 4: SINGLE-STOCK FINBERT DEEP DIVE
# ==========================================
with tab_deepdive:
    st.subheader("🔍 Deep-Dive Ticker News & FinBERT Inspector")
    st.caption("Type any NSE symbol (or company name) to inspect every individual Google News headline and its exact FinBERT classification score.")

    col_in, col_action = st.columns([3, 1])
    with col_in:
        custom_query = st.text_input("Enter NSE Symbol or Company Name", value="TATAMOTORS")
    with col_action:
        st.write("")
        st.write("")
        inspect_btn = st.button("🧠 Run FinBERT Analysis", use_container_width=True)

    if inspect_btn and custom_query.strip():
        engine = get_sentiment_engine()
        ticker_clean = custom_query.strip().upper()
        headlines = engine.fetch_google_news(ticker_clean, max_headlines=10)

        if not headlines:
            st.warning(f"No recent Google News RSS entries found for '{ticker_clean}'.")
        else:
            raw_scores = engine.nlp(headlines)
            breakdown_rows = []
            for h, s in zip(headlines, raw_scores):
                lbl = s["label"].upper()
                icon = "🟢" if lbl == "POSITIVE" else ("🔴" if lbl == "NEGATIVE" else "⚪")
                breakdown_rows.append({
                    "Headline": h,
                    "FinBERT Classification": f"{icon} {lbl}",
                    "Model Probability": round(float(s["score"]) * 100, 1)
                })

            summary = engine.analyze_stock_sentiment(ticker_clean)
            st.success(
                f"**Aggregate FinBERT Verdict for {ticker_clean}:** {summary['sentiment']} "
                f"(Avg Weight: {int(summary['confidence'] * 100)}%)"
            )

            st.dataframe(
                pd.DataFrame(breakdown_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Headline": st.column_config.TextColumn("Scraped Google News Headline (Last 48h)", width="large"),
                    "FinBERT Classification": st.column_config.TextColumn("FinBERT Label", width="small"),
                    "Model Probability": st.column_config.ProgressColumn("Confidence Score", min_value=0, max_value=100, format="%.1f%%")
                }
            )