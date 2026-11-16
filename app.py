import base64
import json
import os
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from trade_engine import AdaptiveTradeEngine

IST = ZoneInfo("Asia/Kolkata")

st.set_page_config(
    page_title="PulseAI — Autonomous NSE Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 1360px;
        }
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
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 1.25rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 12px;
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
        }
        .regime-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
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


def persist_portfolio_to_github(portfolio_dict: dict) -> bool:
    token = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
    repo = st.secrets.get("GITHUB_REPO", "sohilaftab/ai-stock-screener")
    if not token:
        return False

    api_url = f"https://api.github.com/repos/{repo}/contents/portfolio_state.json"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "PulseAI-Streamlit",
    }
    try:
        req = urllib.request.Request(api_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=8) as resp:
            existing = json.loads(resp.read().decode("utf-8"))
        sha = existing.get("sha")

        encoded = base64.b64encode(json.dumps(portfolio_dict, indent=2).encode("utf-8")).decode("utf-8")
        payload = json.dumps({
            "message": "chore(user): manual capital deposit via PulseAI dashboard",
            "content": encoded,
            "sha": sha,
            "branch": "main",
        }).encode("utf-8")

        put_req = urllib.request.Request(api_url, data=payload, headers=headers, method="PUT")
        with urllib.request.urlopen(put_req, timeout=8):
            return True
    except Exception:
        return False


def get_regime_style(regime_name: str):
    r = str(regime_name).upper()
    if r == "BULL_TREND":
        return "🟢", "rgba(16, 185, 129, 0.16)", "#34d399", "rgba(16, 185, 129, 0.40)"
    if r in ("BEAR_TREND", "HIGH_VOLATILITY_PANIC"):
        return "🔴", "rgba(239, 68, 68, 0.16)", "#f87171", "rgba(239, 68, 68, 0.40)"
    if r == "SIDEWAYS_CHOP":
        return "🟡", "rgba(245, 158, 11, 0.16)", "#fbbf24", "rgba(245, 158, 11, 0.40)"
    return "⚪", "rgba(148, 163, 184, 0.16)", "#cbd5e1", "rgba(148, 163, 184, 0.35)"


engine = AdaptiveTradeEngine()
perf = engine.get_performance_metrics()
trades = engine.ledger
open_trades = [t for t in trades if str(t.get("status", "")).upper() == "OPEN"]
closed_trades = [t for t in trades if str(t.get("status", "")).upper() != "OPEN"]

regime_info = engine.weights.get("last_market_regime", {})
regime_name = regime_info.get("regime", "NEUTRAL")
nifty_ltp = float(regime_info.get("nifty_ltp", 0.0))
nifty_chg = float(regime_info.get("nifty_chg_pct", 0.0))
vix_ltp = float(regime_info.get("vix_ltp", 14.0))
vix_chg = float(regime_info.get("vix_chg_pct", 0.0))
size_mult = float(regime_info.get("size_multiplier", 1.0))
r_icon, r_bg, r_fg, r_border = get_regime_style(regime_name)

nifty_str = f"Nifty {nifty_ltp:,.2f} ({nifty_chg:+.2f}%)" if nifty_ltp > 0 else f"Nifty {nifty_chg:+.2f}%"
regime_badge_text = f"{r_icon} Regime: {regime_name} • {nifty_str} • India VIX {vix_ltp:.2f} ({vix_chg:+.1f}%)"

watchlist_data = []
if os.path.exists("morning_watchlist.json"):
    try:
        with open("morning_watchlist.json", "r", encoding="utf-8") as f:
            raw_wl = json.load(f)
            watchlist_data = raw_wl.get("items", []) if isinstance(raw_wl, dict) else raw_wl
    except Exception:
        watchlist_data = []

# ---------------------------------------------------------
# SIDEBAR: WALLET, LIVE REGIME & USER-ONLY MANUAL DEPOSITS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 💼 Wallet Summary")
    st.metric(
        label="Total Portfolio Value",
        value=f"₹{perf['current_equity']:,.0f}",
        delta=f"{perf['total_pnl_pct']:+.2f}% net return",
    )
    st.metric(label="Free Cash Ready to Trade", value=f"₹{perf['available_cash']:,.0f}")
    st.metric(label="Money Currently in Stocks", value=f"₹{perf['allocated_capital']:,.0f}")
    st.caption(
        f"Base Seed: ₹{perf['initial_capital']:,.0f} • "
        f"Manual Deposits: ₹{perf['manual_deposits']:,.0f}"
    )

    st.divider()
    st.markdown("### 🌐 Live Market Regime")
    st.markdown(
        f"""
        <div style="background:{r_bg}; border:1px solid {r_border}; border-radius:10px; padding:10px 12px; margin-bottom:8px;">
            <div style="color:{r_fg}; font-weight:700; font-size:0.92rem;">{r_icon} {regime_name}</div>
            <div style="color:#e2e8f0; font-size:0.82rem; margin-top:4px;">{nifty_str}</div>
            <div style="color:#cbd5e1; font-size:0.82rem;">India VIX: {vix_ltp:.2f} ({vix_chg:+.1f}%) • Size Scale: {int(size_mult * 100)}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### ➕ Add Money Manually")
    st.caption("Only you can add capital here. The AI never adds funds on its own.")
    with st.form("manual_deposit_form", clear_on_submit=True):
        deposit_amt = st.number_input(
            "Deposit Amount (₹)",
            min_value=1000,
            max_value=10000000,
            value=25000,
            step=5000,
        )
        deposit_note = st.text_input("Deposit Note (Optional)", placeholder="e.g., Added extra trading funds")
        if st.form_submit_button("Deposit Into Wallet", use_container_width=True):
            updated_port = engine.add_manual_funds(
                deposit_amt,
                note=deposit_note or "Manual deposit via Dashboard",
                notify_telegram=True,
            )
            synced = persist_portfolio_to_github(updated_port)
            if synced:
                st.success(f"Added ₹{deposit_amt:,.0f} and synced permanently to GitHub!")
            else:
                st.success(f"Added ₹{deposit_amt:,.0f} to wallet! (Add GITHUB_TOKEN in Streamlit Secrets to auto-sync to GitHub.)")
            st.rerun()

    st.divider()
    with st.expander("⚙️ Developer Controls (Keep OFF)", expanded=False):
        st.caption("Cloud Autopilot runs via GitHub Actions. Keep browser overrides OFF.")
        st.toggle("Enable Browser Autopilot", value=False, key="ui_autopilot")
        st.checkbox("Simulate After-Hours Entry", value=False, key="ui_sim")

# ---------------------------------------------------------
# MAIN HEADER & KPI CARDS
# ---------------------------------------------------------
now_ist = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
st.markdown(
    f"""
    <div class="hero-card">
        <h2 style="margin:0; color:#f8fafc;">📈 PulseAI — Quantitative Bandit & Half-Kelly Portfolio Manager</h2>
        <p style="margin:6px 0 0 0; color:#94a3b8; font-size:0.95rem;">
            Self-learning NSE engine powered by Thompson Sampling Multi-Armed Bandits, Half-Kelly sizing, Mansfield Relative Strength vs. Nifty 50, and real-time news emotion guards.
        </p>
        <div class="pill-row">
            <span class="status-pill">● Brain Gen #{engine.weights.get('generation', 1)} • Synced: {now_ist}</span>
            <span class="regime-pill" style="background:{r_bg}; color:{r_fg}; border:1px solid {r_border};">{regime_badge_text}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Value", f"₹{perf['current_equity']:,.0f}", f"Contributed: ₹{perf['total_capital_contributed']:,.0f}")
c2.metric("Available Cash", f"₹{perf['available_cash']:,.0f}", f"In Trades: ₹{perf['allocated_capital']:,.0f}")
c3.metric("Open Trade P&L", f"₹{perf['unrealized_pnl_inr']:+,.0f}", f"{len(open_trades)} Active Positions")
c4.metric("Settled Win Rate", f"{perf['win_rate']:.1f}%", f"Realized P&L: ₹{perf['total_pnl_inr']:+,.0f}")

st.write("")

# ---------------------------------------------------------
# TABS: TRADES, POST-TRADE RESEARCH, QUANT BRAIN & WATCHLIST
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        f"📊 Active & Past Trades ({len(open_trades)} Open)",
        "🧠 AI Research & Mistake Autopsy",
        "🎰 Bandit Arms, Half-Kelly & Timing",
        f"🌅 Watchlist ({len(watchlist_data)})",
    ]
)

with tab1:
    st.subheader("Currently Open Positions")
    if not open_trades:
        st.info("No active trades right now. The AI enters automatically when a setup passes its Bandit, Mansfield RS, Regime, and Sector filters.")
    else:
        rows = []
        sector_summary = {}
        for t in open_trades:
            entry = float(t.get("entry_price", 0))
            target = float(t.get("target_1", t.get("initial_target", 0)))
            sl = float(t.get("stop_loss", 0))
            qty = int(t.get("qty", 0))
            invested = float(t.get("capital_used", entry * qty))
            unrl = float(t.get("unrealized_pnl_inr", 0))
            peak = float(t.get("peak_pnl_inr", 0))
            conv = float(t.get("conviction", 0))
            sec = str(t.get("sector", "General"))
            mrs = float(t.get("mansfield_rs", 0.0))
            kelly = float(t.get("kelly_pct", 15.0))

            sector_summary.setdefault(sec, {"count": 0, "capital": 0.0})
            sector_summary[sec]["count"] += 1
            sector_summary[sec]["capital"] += invested

            headline = str(t.get("headline", "")).strip()
            note = str(t.get("trade_management_note", "")).strip()
            why_text = f"{headline} — {note}" if (headline and note) else (headline or note)

            rows.append(
                {
                    "Stock": t.get("symbol", "N/A").replace(".NS", ""),
                    "Sector": sec,
                    "Side": t.get("action", "BUY"),
                    "Bandit Arm": t.get("strategy_arm", "MOMENTUM_BREAKOUT"),
                    "Mansfield RS": f"{mrs:+.2f}%",
                    "Buy/Entry (₹)": f"₹{entry:,.2f}",
                    "Target (₹)": f"₹{target:,.2f}",
                    "Stop-Loss (₹)": f"₹{sl:,.2f}",
                    "Shares": qty,
                    "Invested (₹)": f"₹{invested:,.0f} ({kelly:.0f}% Kelly)",
                    "Live P&L (₹)": f"₹{unrl:+,.0f} (Peak ₹{peak:+,.0f})",
                    "Conviction": f"{conv:.1f}/100",
                    "Catalyst & Live AI Note": why_text,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        eq = max(float(perf["current_equity"]), 1.0)
        sec_tags = [
            f"**{sec}**: {vals['count']}/2 trades (₹{vals['capital']:,.0f} • {vals['capital'] / eq * 100:.0f}% of equity)"
            for sec, vals in sector_summary.items()
        ]
        st.caption("🛡️ **Sector Correlation Guard (Max 2 trades / 40% equity per sector):** " + " | ".join(sec_tags))

    st.subheader("Completed Trade History & Lessons")
    if not closed_trades:
        st.caption("No settled trades yet. Whenever the AI closes a trade, its exact exit reason and research lesson appear here.")
    else:
        closed_rows = []
        for t in reversed(closed_trades[-30:]):
            pnl = float(t.get("pnl_inr", 0))
            closed_rows.append(
                {
                    "Date": t.get("date", "—"),
                    "Stock": t.get("symbol", "N/A").replace(".NS", ""),
                    "Sector": t.get("sector", "General"),
                    "Arm": t.get("strategy_arm", "—"),
                    "Side": t.get("action", "BUY"),
                    "Entry (₹)": f"₹{float(t.get('entry_price', 0)):,.2f}",
                    "Exit (₹)": f"₹{float(t.get('exit_price', 0)):,.2f}",
                    "Hold": f"{float(t.get('hold_minutes', 0)):.0f}m",
                    "Profit / Loss": f"₹{pnl:+,.0f} ({float(t.get('pnl_pct', 0)):+.2f}%)",
                    "AI Exit Reason": t.get("status", "CLOSED"),
                    "Post-Trade Research Lesson": t.get("mistake_diagnosis", "—"),
                }
            )
        st.dataframe(pd.DataFrame(closed_rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🧠 Live Post-Trade Research & Mistake Autopsy Log")
    st.caption(
        "Immediately after every trade closes, the AI scans breaking news emotion for that stock, "
        "identifies why it won or lost, and updates its Bayesian Bandit arms."
    )
    lessons = engine.weights.get("trade_lessons_log", [])
    if not lessons:
        st.info("No post-trade research lessons recorded yet. They populate automatically as trades settle.")
    else:
        for item in lessons[:15]:
            icon = "✅" if float(item.get("pnl_inr", 0)) >= 0 else "🔍"
            regime_tag = f" `[{item['regime']}]`" if item.get("regime") else ""
            st.markdown(f"{icon} **{item.get('timestamp', '')}**{regime_tag} — {item.get('lesson', '')}")

    st.divider()
    st.subheader("💰 Manual Deposit History")
    deposits = engine.portfolio.get("deposit_history", [])
    if not deposits:
        st.caption("No manual deposits added yet. Use the 'Add Money Manually' form in the left sidebar whenever you want to add funds.")
    else:
        st.dataframe(pd.DataFrame(deposits), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🎰 Thompson Sampling Multi-Armed Bandit (Beta α / β Posteriors)")
    st.caption(
        "Each strategy arm models its learned win probability as Beta(α, β). "
        "Higher α/β means higher posterior win probability and larger Half-Kelly position sizing."
    )
    arms = engine.weights.get("bandit_arms", {})
    if arms:
        arm_rows = []
        for arm_name, stats in arms.items():
            a = float(stats.get("alpha", 2.0))
            b = float(stats.get("beta", 2.0))
            post_p = (a / (a + b)) * 100.0
            arm_rows.append(
                {
                    "Strategy Arm": arm_name,
                    "Alpha (Wins)": round(a, 2),
                    "Beta (Losses)": round(b, 2),
                    "Posterior Win Prob": f"{post_p:.1f}%",
                    "Trades Settled": int(stats.get("trades", 0)),
                    "Net P&L (₹)": f"₹{float(stats.get('net_pnl_inr', 0.0)):+,.0f}",
                }
            )
        st.dataframe(pd.DataFrame(arm_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("⏱️ Learned Time-of-Day Session Performance")
    sess_mem = engine.weights.get("session_memory", {})
    if not sess_mem:
        st.info("Session timing stats will appear automatically once trades complete across different market windows.")
    else:
        sess_rows = [
            {
                "Market Window (IST)": k,
                "Trades Taken": v.get("trades", 0),
                "Win Rate": f"{v.get('win_rate', 0)}%",
                "Net P&L (₹)": f"₹{v.get('net_pnl_inr', 0):+,.0f}",
                "AI Score Adjustment": f"{v.get('score_adj', 0):+.1f} pts",
            }
            for k, v in sess_mem.items()
        ]
        st.dataframe(pd.DataFrame(sess_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🎛️ Current Learned AI Parameters")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Min Conviction Bar", f"{engine.weights.get('min_conviction_threshold', 52.0)}/100")
    p2.metric("Profit Giveback Cutoff", f"{engine.weights.get('max_profit_giveback_pct', 45.0)}%")
    p3.metric("Optimal Max Hold", f"{engine.weights.get('learned_max_hold_mins', 180.0):.0f} mins")
    p4.metric("ATR SL / Target", f"{engine.weights.get('atr_sl_multiplier', 0.55)}x / {engine.weights.get('atr_tp_multiplier', 1.60)}x")

with tab4:
    st.subheader("Latest AI Watchlist Snapshot")
    if not watchlist_data:
        st.info("Watchlist updates automatically during market scans.")
    else:
        st.dataframe(pd.DataFrame(watchlist_data), use_container_width=True, hide_index=True)