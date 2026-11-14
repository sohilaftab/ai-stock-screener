import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from nse_client import NSEClient
from market_data import MarketDataScanner
from sentiment_engine import FinBERTSentimentEngine
from telegram_notifier import TelegramNotifier
from trade_engine import AdaptiveTradeEngine

load_dotenv()

st.set_page_config(
    page_title="PulseAI | ₹1.5L Autonomous Intraday Trader",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white; padding: 20px 24px; border-radius: 14px;
        margin-bottom: 18px; border: 1px solid #334155;
    }
    .stock-card {
        background-color: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 16px; margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    @media (prefers-color-scheme: dark) {
        .stock-card { background-color: #1e293b; border-color: #334155; }
    }
    .badge-buy { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 99px; font-weight: 700; font-size: 0.82rem; }
    .badge-sell { background: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 99px; font-weight: 700; font-size: 0.82rem; }
    .step-box {
        background: rgba(59, 130, 246, 0.08); border-left: 4px solid #3b82f6;
        padding: 10px 14px; border-radius: 6px; font-size: 0.9rem; margin-bottom: 16px;
    }
    .level-box {
        display: flex; justify-content: space-between; background: rgba(148, 163, 184, 0.12);
        padding: 8px 12px; border-radius: 8px; margin: 8px 0; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Connecting to NSE India...")
def get_nse_client():
    return NSEClient()


@st.cache_resource(show_spinner="Initializing FinBERT Financial AI Model...")
def get_sentiment_engine():
    return FinBERTSentimentEngine()


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_universe():
    return get_nse_client().get_nifty_universe_tickers()


@st.cache_data(ttl=120, show_spinner="Scanning NSE Pre-Open Order Book...")
def fetch_premarket_data(universe_key, min_gap, min_vol, top_n):
    return get_nse_client().get_pre_open_movers(
        key=universe_key, min_gap_pct=min_gap, min_volume=min_vol, top_n=top_n
    )


@st.cache_data(ttl=300, show_spinner="Scanning Nifty charts (EMA 20/50, ATR, RSI, Volume)...")
def fetch_intraday_breakouts(vol_mult, min_price_chg, top_n):
    tickers = get_cached_universe()
    scanner = MarketDataScanner(tickers)
    result = scanner.scan_volume_breakouts(
        vol_multiplier=vol_mult, min_price_change_pct=min_price_chg, top_n=top_n
    )
    return result if isinstance(result, tuple) else (result, False)


def enrich_with_finbert(df, symbol_col="symbol"):
    if df.empty:
        return df

    engine = get_sentiment_engine()
    sentiments, raw_labels, confidences, headlines, action_tags = [], [], [], [], []
    progress = st.progress(0, text="🧠 FinBERT AI is reading the latest news headlines...")
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        sym = row[symbol_col]
        res = engine.analyze_stock_sentiment(sym)
        lbl = res["sentiment"]
        conf = int(res["confidence"] * 100)
        chg = row.get("change_pct", row.get("pChange", 0))

        if lbl == "POSITIVE" and chg >= 0:
            tag, badge = "🟢 Bullish News + Chart", "🟢 POSITIVE"
        elif lbl == "NEGATIVE" and chg < 0:
            tag, badge = "🔴 Bearish Breakdown", "🔴 NEGATIVE"
        elif lbl == "NEGATIVE" and chg >= 0:
            tag, badge = "⚠️ Price Up on Bad News", "🔴 NEGATIVE"
        elif lbl == "POSITIVE" and chg < 0:
            tag, badge = "👀 Dip on Positive News", "🟢 POSITIVE"
        else:
            tag, badge = "⚡ Chart & Volume Driven", "⚪ NEUTRAL"

        raw_labels.append(lbl)
        sentiments.append(badge)
        confidences.append(conf)
        headlines.append(res["top_headline"])
        action_tags.append(tag)
        progress.progress((idx + 1) / total, text=f"🧠 Analyzing news for {sym} ({idx + 1}/{total})...")

    progress.empty()
    enriched = df.copy()
    enriched["raw_sentiment"] = raw_labels
    enriched["AI Verdict"] = sentiments
    enriched["Signal Takeaway"] = action_tags
    enriched["AI Confidence"] = confidences
    enriched["Why Is It Moving? (Latest News)"] = headlines
    return enriched


trade_engine = AdaptiveTradeEngine()
perf = trade_engine.get_performance_metrics()

# ==========================================
# SIDEBAR: ₹1,50,000 WALLET & AI BRAIN STATUS
# ==========================================
with st.sidebar:
    st.markdown("## 💼 ₹1.5L Autonomous Wallet")
    st.metric("Current Portfolio Equity", f"₹{perf['current_equity']:,.2f}", f"₹{perf['total_pnl_inr']:+,.0f} ({perf['total_pnl_pct']:+.2f}%)")
    c_cash, c_alloc = st.columns(2)
    c_cash.metric("Available Cash", f"₹{perf['available_cash']:,.0f}")
    c_alloc.metric("In Active Trades", f"₹{perf['allocated_capital']:,.0f}")

    st.divider()
    st.markdown(f"### 🧠 Self-Learning Brain (Gen #{trade_engine.weights['generation']})")
    st.caption(f"Max Risk Per Trade: **{trade_engine.weights.get('risk_per_trade_pct', 1.0)}%** of capital")
    st.progress(float(trade_engine.weights["volume_weight"]), text=f"📊 Volume Surge: {int(trade_engine.weights['volume_weight']*100)}%")
    st.progress(float(trade_engine.weights["sentiment_weight"]), text=f"📰 FinBERT News AI: {int(trade_engine.weights['sentiment_weight']*100)}%")
    st.progress(float(trade_engine.weights["premarket_weight"]), text=f"🌅 Pre-Market Flow: {int(trade_engine.weights['premarket_weight']*100)}%")
    st.progress(float(trade_engine.weights["momentum_weight"]), text=f"📈 Chart (EMA/RSI/VWAP): {int(trade_engine.weights['momentum_weight']*100)}%")
    st.caption(f"*{trade_engine.weights['learning_notes']}*")

    st.divider()
    preset = st.radio(
        "🎯 Screening Strictness:",
        options=["⚖️ Balanced (Recommended)", "🔥 High Conviction Only", "🌊 Show More Stocks"],
        index=0
    )
    if "High Conviction" in preset:
        intra_vol_mult, intra_min_chg, pre_min_gap, max_results = 1.8, 2.0, 2.5, 6
    elif "Show More" in preset:
        intra_vol_mult, intra_min_chg, pre_min_gap, max_results = 1.0, 0.5, 1.0, 12
    else:
        intra_vol_mult, intra_min_chg, pre_min_gap, max_results = 1.3, 1.2, 1.8, 8

    if st.button("🔄 Refresh All Live Data", width="stretch", type="primary"):
        st.cache_data.clear()
        for key in ["intraday_df", "premarket_df", "eod_df"]:
            st.session_state.pop(key, None)
        st.rerun()

    notifier = TelegramNotifier()

# ==========================================
# HERO BANNER
# ==========================================
st.markdown(f"""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <div>
            <h2 style="margin:0; color:white; font-size:1.55rem;">⚡ PulseAI — Autonomous ₹1,50,000 Intraday Portfolio Manager</h2>
            <p style="margin:4px 0 0 0; color:#cbd5e1; font-size:0.92rem;">
                Pre-9 AM Multi-Timeframe Research • 9:10 AM Pre-Market Validation • Dynamic Live SL Trailing & Target Extension • EOD Mistake Autopsy
            </p>
        </div>
        <div style="text-align:right; margin-top:8px;">
            <span style="background:#334155; padding:6px 14px; border-radius:99px; font-weight:600; font-size:0.85rem;">
                💰 Equity: ₹{perf['current_equity']:,.0f} • {datetime.now().strftime('%I:%M %p IST')}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_advisor, tab_intra, tab_pre, tab_eod, tab_search = st.tabs([
    "🎯 1. Autonomous Trader & ₹1.5L Portfolio",
    "🔥 2. Live Chart & Volume Breakouts",
    "🌅 3. Pre-Market Gappers (9:00 AM)",
    "🏦 4. Big Player Delivery (>60%)",
    "🔍 5. Search Any Stock's News"
])

if "intraday_df" not in st.session_state:
    raw_df, is_fallback = fetch_intraday_breakouts(intra_vol_mult, intra_min_chg, max_results)
    st.session_state["intraday_fallback"] = is_fallback
    st.session_state["intraday_df"] = enrich_with_finbert(raw_df, symbol_col="symbol")

intra_df = st.session_state.get("intraday_df", pd.DataFrame())
pre_df = st.session_state.get("premarket_df", pd.DataFrame())

# ==========================================
# TAB 1: AUTONOMOUS TRADER & ₹1.5L PORTFOLIO
# ==========================================
with tab_advisor:
    w1, w2, w3, w4, w5 = st.columns(5)
    w1.metric("Starting Capital", f"₹{perf['initial_capital']:,.0f}")
    w2.metric("Current Wallet Equity", f"₹{perf['current_equity']:,.0f}", f"{perf['total_pnl_pct']:+.2f}%")
    w3.metric("Available Cash", f"₹{perf['available_cash']:,.0f}", f"In Trades: ₹{perf['allocated_capital']:,.0f}")
    w4.metric("Live Unrealized P&L", f"₹{perf['unrealized_pnl_inr']:+,.0f}", f"{perf['open_trades']} Active Trades")
    w5.metric("Win Rate (Settled)", f"{perf['win_rate']}%", f"Realized: ₹{perf['total_pnl_inr']:+,.0f}")

    st.divider()
    rec_df = trade_engine.generate_intraday_recommendations(intra_df, pre_df, top_k=5)

    if not rec_df.empty:
        st.subheader("🚀 Position-Sized Intraday Setups (Allocated from ₹1,50,000 Capital)")
        card_cols = st.columns(min(3, len(rec_df)))
        for idx, col in enumerate(card_cols):
            r = rec_df.iloc[idx]
            badge_cls = "badge-buy" if r["action"] == "BUY" else "badge-sell"
            with col:
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.25rem; font-weight:800;">{r['symbol']}</span>
                        <span class="{badge_cls}">{r['action']} • {r['conviction']}/100</span>
                    </div>
                    <div style="font-size:0.85rem; margin-top:6px; color:#3b82f6; font-weight:600;">
                        📦 Qty: {r['qty']} shares (Capital: ₹{r['capital_used']:,.0f}) | Chart: {r['chart_trend']}
                    </div>
                    <div class="level-box">
                        <div><b>Entry:</b><br>₹{r['entry_price']}</div>
                        <div style="color:#16a34a;"><b>Target 1:</b><br>₹{r['target_1']} (+₹{r['expected_profit_inr']:,.0f})</div>
                        <div style="color:#15803d;"><b>Target 2:</b><br>₹{r['target_2']}</div>
                        <div style="color:#dc2626;"><b>Stop-Loss:</b><br>₹{r['stop_loss']} (-₹{r['max_loss_inr']:,.0f})</div>
                    </div>
                    <div style="font-size:0.78rem; opacity:0.85; border-top:1px solid #e2e8f0; padding-top:6px;">
                        📰 <i>"{r['headline']}"</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 4 Autonomous Control Buttons
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("🌅 1. Save Pre-9 AM Watchlist", width="stretch"):
                trade_engine.save_morning_watchlist(rec_df)
                st.success("✅ Saved Pre-9 AM Watchlist! At 9:10 AM, any pre-market momentum shifts will be flagged.")
        with b2:
            if st.button("💼 2. Take Virtual Trades (₹1.5L)", type="primary", width="stretch"):
                added = trade_engine.record_recommendations(rec_df)
                if added > 0:
                    st.success(f"✅ Executed {added} virtual trades from your ₹1,50,000 wallet!")
                    st.rerun()
                else:
                    st.info("ℹ️ Today's trades are already open or cash is fully allocated.")
        with b3:
            if st.button("📡 3. Manage Live Trades (Trail/Cut)", width="stretch"):
                with st.spinner("Inspecting 5-minute VWAP candles & adjusting active targets/SLs..."):
                    actions = trade_engine.manage_live_positions()
                    if actions:
                        for a in actions:
                            st.toast(a, icon="⚡")
                    st.rerun()
        with b4:
            if st.button("🏁 4. EOD Settle & Learn from Mistakes", width="stretch"):
                with st.spinner("Settling EOD P&L, running Mistake Autopsy & updating AI weights..."):
                    res = trade_engine.evaluate_open_trades_and_learn()
                    st.success(f"🎉 Settled {res['evaluated']} trades! {res['message']}")
                    st.rerun()

    st.divider()
    st.subheader("📋 Autonomous Trade Ledger & Live Position Tracker")
    if not trade_engine.ledger:
        st.info("No trades recorded yet. Click **'2. Take Virtual Trades (₹1.5L)'** above to deploy capital!")
    else:
        ledger_df = pd.DataFrame(trade_engine.ledger).iloc[::-1]
        for col_name, default_val in [
            ("qty", 1), ("capital_used", 0.0), ("current_price", 0.0),
            ("unrealized_pnl_inr", 0.0), ("trade_management_note", ""), ("mistake_diagnosis", "")
        ]:
            if col_name not in ledger_df.columns:
                ledger_df[col_name] = default_val

        st.dataframe(
            ledger_df[[
                "date", "symbol", "action", "qty", "capital_used", "entry_price",
                "current_price", "target_1", "stop_loss", "status", "pnl_inr",
                "trade_management_note", "mistake_diagnosis"
            ]],
            width="stretch",
            hide_index=True,
            column_config={
                "date": st.column_config.TextColumn("Date"),
                "symbol": st.column_config.TextColumn("Stock"),
                "action": st.column_config.TextColumn("Side"),
                "qty": st.column_config.NumberColumn("Qty", format="%d"),
                "capital_used": st.column_config.NumberColumn("Capital Used", format="₹%.0f"),
                "entry_price": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                "current_price": st.column_config.NumberColumn("LTP / Exit", format="₹%.2f"),
                "target_1": st.column_config.NumberColumn("Active Target", format="₹%.2f"),
                "stop_loss": st.column_config.NumberColumn("Active SL", format="₹%.2f"),
                "status": st.column_config.TextColumn("Status"),
                "pnl_inr": st.column_config.NumberColumn("Realized P&L", format="₹%+,.0f"),
                "trade_management_note": st.column_config.TextColumn("Live AI Trade Management", width="medium"),
                "mistake_diagnosis": st.column_config.TextColumn("EOD Mistake Autopsy", width="large"),
            }
        )

    mistakes = trade_engine.portfolio.get("mistakes_identified", [])
    if mistakes:
        with st.expander("🔍 View AI Self-Reflection & Mistake Autopsy Log", expanded=True):
            for m in mistakes[:5]:
                st.markdown(f"- 🧠 {m}")

# ==========================================
# TAB 2: LIVE CHART & VOLUME BREAKOUTS
# ==========================================
with tab_intra:
    if not intra_df.empty:
        st.dataframe(
            intra_df[[
                "symbol", "ltp", "change_pct", "chart_trend", "trend_30d",
                "vol_ratio", "rsi", "Signal Takeaway", "AI Verdict", "Why Is It Moving? (Latest News)"
            ]],
            width="stretch",
            hide_index=True
        )

# ==========================================
# TAB 3: PRE-MARKET MOVERS
# ==========================================
with tab_pre:
    if st.button("🚀 Scan NSE Pre-Open Session & Compare with Morning Watchlist", type="primary"):
        raw_pre = fetch_premarket_data("FO", pre_min_gap, 25000, max_results)
        st.session_state["premarket_df"] = enrich_with_finbert(raw_pre, symbol_col="symbol")
        st.rerun()

    if not pre_df.empty:
        new_recs = trade_engine.generate_intraday_recommendations(intra_df, pre_df, top_k=5)
        shifts = trade_engine.compare_with_morning_watchlist(new_recs)
        if shifts:
            st.warning("⚡ **Pre-Market Momentum Shifts Detected vs Pre-9 AM List:**\n\n" + "\n".join(f"- {s}" for s in shifts))
        st.dataframe(pre_df, width="stretch", hide_index=True)

# ==========================================
# TAB 4: EOD INSTITUTIONAL DELIVERY
# ==========================================
with tab_eod:
    if st.button("🏦 Scan Today's Leaders for High Delivery (>60%)", type="primary"):
        base_df, _ = fetch_intraday_breakouts(vol_mult=1.1, min_price_chg=0.5, top_n=15)
        if not base_df.empty:
            deliv_df = get_nse_client().get_eod_high_delivery_stocks(base_df["symbol"].tolist(), min_delivery_pct=60.0)
            if not deliv_df.empty:
                merged = pd.merge(deliv_df, base_df, on="symbol", how="left")
                st.session_state["eod_df"] = enrich_with_finbert(merged, symbol_col="symbol")
    eod_df = st.session_state.get("eod_df", pd.DataFrame())
    if not eod_df.empty:
        st.dataframe(eod_df, width="stretch", hide_index=True)

# ==========================================
# TAB 5: SINGLE STOCK NEWS INSPECTOR
# ==========================================
with tab_search:
    c1, c2 = st.columns([3, 1])
    with c1:
        query_sym = st.text_input("Type Stock Symbol or Company Name:", value="TATAMOTORS")
    with c2:
        st.write("")
        st.write("")
        run_search = st.button("🧠 Analyze Stock News", type="primary", width="stretch")

    if run_search and query_sym.strip():
        engine = get_sentiment_engine()
        clean_q = query_sym.strip().upper()
        headlines = engine.fetch_google_news(clean_q, max_headlines=8)
        if headlines:
            scores = engine.nlp(headlines)
            rows = [{"News Headline": h, "AI Sentiment": s["label"].upper(), "Confidence": round(float(s["score"]) * 100, 1)} for h, s in zip(headlines, scores)]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)