import json
import os
from datetime import datetime
import pytz
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. PAGE CONFIG & CLEAN RESPONSIVE CSS (Fixes Clipping)
# ---------------------------------------------------------
st.set_page_config(
    page_title="PulseAI — NSE Paper Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

IST = pytz.timezone("Asia/Kolkata")

st.markdown(
    """
    <style>
        /* Fix top tab clipping below Streamlit header */
        .block-container {
            padding-top: 3.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px;
        }
        /* Clean readable tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding-bottom: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
        }
        /* Hero banner */
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 1.25rem;
        }
        .status-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
            margin-top: 8px;
        }
        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 14px 16px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. SAFE JSON LOADERS
# ---------------------------------------------------------
def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


portfolio = load_json(
    "portfolio_state.json",
    {"starting_capital": 150000.0, "available_cash": 150000.0, "realized_pnl": 0.0},
)
trades = load_json("trade_history.json", [])
watchlist = load_json("morning_watchlist.json", [])

open_trades = [t for t in trades if str(t.get("status", "")).upper() == "OPEN"]
closed_trades = [t for t in trades if str(t.get("status", "")).upper() != "OPEN"]

starting_cap = float(portfolio.get("starting_capital", 150000.0))
available_cash = float(portfolio.get("available_cash", starting_cap))
invested_cash = sum(float(t.get("allocated_capital", t.get("entry_price", 0) * t.get("qty", 0))) for t in open_trades)
unrealized_pnl = sum(float(t.get("unrealized_pnl", 0.0)) for t in open_trades)
realized_pnl = float(portfolio.get("realized_pnl", sum(float(t.get("pnl", 0.0)) for t in closed_trades)))
current_equity = available_cash + invested_cash + unrealized_pnl
total_return_pct = ((current_equity - starting_cap) / starting_cap * 100) if starting_cap else 0.0

wins = sum(1 for t in closed_trades if float(t.get("pnl", 0.0)) > 0)
win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0.0

# ---------------------------------------------------------
# 4. CLEAN SIDEBAR (No Truncated Numbers)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 💼 Wallet Summary")
    st.metric(
        label="Total Portfolio Value",
        value=f"₹{current_equity:,.0f}",
        delta=f"{total_return_pct:+.2f}% overall",
    )
    # Full-width metrics prevent "₹142,..." truncation
    st.metric(label="Free Cash Ready to Trade", value=f"₹{available_cash:,.0f}")
    st.metric(label="Money Currently in Stocks", value=f"₹{invested_cash:,.0f}")

    st.divider()
    st.markdown("### 🕒 Daily AI Schedule (IST)")
    st.caption(
        "• **08:30 AM** — Scans overnight news & filings\n"
        "• **09:10 AM** — Confirms gap-up & enters trades\n"
        "• **10:15 / 12:15 / 02:15** — Protects profits & SL\n"
        "• **03:40 PM** — Closes day & trains AI model"
    )

    st.divider()
    with st.expander("⚙️ Developer Controls (Keep OFF)", expanded=False):
        st.caption("GitHub Actions runs automatically on schedule. Keep these OFF to avoid duplicate trades.")
        st.toggle("Enable Browser Autopilot", value=False, key="ui_autopilot")
        st.checkbox("Simulate After-Hours Entry", value=False, key="ui_sim")

# ---------------------------------------------------------
# 5. MAIN HEADER & BEGINNER GUIDE
# ---------------------------------------------------------
now_ist = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
st.markdown(
    f"""
    <div class="hero-card">
        <h2 style="margin:0; color:#f8fafc;">📈 PulseAI — Smart NSE Paper Trading Dashboard</h2>
        <p style="margin:6px 0 0 0; color:#94a3b8; font-size:0.95rem;">
            Tracks a virtual <strong>₹1,50,000</strong> intraday portfolio using verified NSE corporate filings, news sentiment, and volume breakouts.
        </p>
        <div class="status-pill">● Cloud Autopilot Active • Last Synced: {now_ist}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️ New here? Click to understand how this dashboard works in 30 seconds", expanded=False):
    st.markdown(
        """
        1. **No Real Money Is Used:** This is a 100% virtual ₹1,50,000 paper wallet used to test AI stock picks safely.
        2. **How Stocks Are Picked:** Every morning before market open, the AI reads Indian financial news, verifies announcements against official NSE filings, and checks pre-market volume.
        3. **Automatic Risk Protection:** Every trade has a fixed **Target Price** (to book profit) and **Stop-Loss** (to cap losses automatically).
        """
    )

# Top 4 KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Value", f"₹{current_equity:,.0f}", f"Start: ₹{starting_cap:,.0f}")
c2.metric("Available Cash", f"₹{available_cash:,.0f}", f"In Trades: ₹{invested_cash:,.0f}")
c3.metric("Open Trade P&L", f"₹{unrealized_pnl:+,.0f}", f"{len(open_trades)} Active Positions")
c4.metric("Win Rate", f"{win_rate:.0f}%", f"{wins}W / {len(closed_trades) - wins}L ({len(closed_trades)} Closed)")

st.write("")

# ---------------------------------------------------------
# 6. CLEAN HUMAN-READABLE TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        f"📊 Active & Past Trades ({len(open_trades)} Open)",
        f"🌅 Morning AI Watchlist ({len(watchlist)})",
        "🧠 How the AI Scores Stocks",
    ]
)

with tab1:
    st.subheader("Currently Open Positions")
    if not open_trades:
        st.info("No active trades right now. New positions are evaluated automatically at 09:10 AM IST on market days.")
    else:
        rows = []
        for t in open_trades:
            entry = float(t.get("entry_price", 0))
            target = float(t.get("target_price", t.get("target", 0)))
            sl = float(t.get("stop_loss", t.get("sl", 0)))
            qty = int(t.get("qty", 0))
            prob = float(t.get("ml_win_prob", t.get("confidence", 0)))
            if prob <= 1.0:
                prob *= 100

            rows.append(
                {
                    "Stock": t.get("symbol", "N/A").replace(".NS", ""),
                    "Buy Price (₹)": f"₹{entry:,.2f}",
                    "Target (₹)": f"₹{target:,.2f}",
                    "Stop-Loss (₹)": f"₹{sl:,.2f}",
                    "Shares": qty,
                    "Invested (₹)": f"₹{entry * qty:,.0f}",
                    "AI Confidence": f"{prob:.0f}%",
                    "Why AI Picked It": t.get("reason", t.get("headline", "Verified news + volume breakout")),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Completed Trade History")
    if not closed_trades:
        st.caption("No settled trades yet. Open positions settle automatically at 03:40 PM IST.")
    else:
        closed_rows = []
        for t in reversed(closed_trades[-20:]):
            pnl = float(t.get("pnl", 0))
            closed_rows.append(
                {
                    "Date": t.get("date", "—"),
                    "Stock": t.get("symbol", "N/A").replace(".NS", ""),
                    "Buy (₹)": f"₹{float(t.get('entry_price', 0)):,.2f}",
                    "Exit (₹)": f"₹{float(t.get('exit_price', 0)):,.2f}",
                    "Profit / Loss": f"₹{pnl:+,.0f}",
                    "Result": "✅ Profit" if pnl > 0 else "❌ Stop-Loss / Exit",
                }
            )
        st.dataframe(pd.DataFrame(closed_rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Today's Pre-Market AI Shortlist (08:30 AM Scan)")
    if not watchlist:
        st.info("Watchlist updates every weekday at 08:30 AM IST before the NSE pre-open session.")
    else:
        st.dataframe(pd.DataFrame(watchlist), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("How PulseAI Filters Out Fake News & Bad Setups")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Step 1: Official NSE Filing Cross-Check**
            Headlines from financial news are cross-checked against official NSE corporate announcements so rumors without filings are ignored.

            **Step 2: FinBERT Sentiment Scoring**
            An Indian market NLP model (`finbert-sentfin`) reads whether the news is genuinely bullish (order wins, earnings beat) or just noise.
            """
        )
    with col_b:
        st.markdown(
            """
            **Step 3: Volume & Pre-Market Confirmation**
            Even good news is skipped unless real buyer volume shows up at 09:10 AM IST.

            **Step 4: Self-Improving Daily Learning**
            At 03:40 PM IST, the model compares its morning predictions against actual closing prices and adjusts its weights automatically.
            """
        )