import base64
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
from trade_engine import AdaptiveTradeEngine

IST = ZoneInfo("Asia/Kolkata")
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = "sohilaftab/ai-stock-screener"

st.set_page_config(
    page_title="PulseAI — Smart NSE Trading & Portfolio Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        code, pre, [data-testid="stMetricValue"], [data-testid="stMetricDelta"], .mono-num {
            font-family: 'JetBrains Mono', monospace !important;
            font-variant-numeric: tabular-nums;
        }

        .stApp {
            background: radial-gradient(circle at top right, #13203b 0%, #090e1a 48%, #070b14 100%);
            color: #e2e8f0;
        }

        .block-container {
            padding-top: 2.2rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1400px;
        }

        /* Top Live Bar */
        .top-ticker-bar {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 12px;
            padding: 10px 16px;
            margin-bottom: 14px;
            backdrop-filter: blur(8px);
        }

        .brand-mark {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 800;
            font-size: 1rem;
            letter-spacing: -0.01em;
            color: #f8fafc;
        }

        .brand-badge {
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 999px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .meta-chip {
            font-size: 0.8rem;
            color: #cbd5e1;
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.16);
            padding: 4px 10px;
            border-radius: 999px;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(125deg, rgba(15, 23, 42, 0.96) 0%, rgba(23, 37, 84, 0.65) 60%, rgba(15, 23, 42, 0.96) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 18px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
        }

        .hero-title {
            margin: 0;
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #f8fafc;
        }

        .hero-subtitle {
            margin: 6px 0 0 0;
            color: #94a3b8;
            font-size: 0.94rem;
            line-height: 1.5;
            max-width: 920px;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .pill-ai {
            background: rgba(16, 185, 129, 0.14);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.16);
            padding: 15px 18px;
            border-radius: 14px;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            border-color: rgba(56, 189, 248, 0.42);
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }

        /* Section / Filter Panel */
        .panel-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 16px;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            padding-bottom: 8px;
            margin-bottom: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 6px 16px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.9rem;
            background: rgba(30, 41, 59, 0.45);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(56, 189, 248, 0.18) !important;
            color: #38bdf8 !important;
        }

        /* Footer */
        .app-footer {
            margin-top: 2.5rem;
            padding: 18px 22px;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: #94a3b8;
            font-size: 0.8rem;
            line-height: 1.55;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get_repo_slug() -> str:
    try:
        if "GITHUB_REPO" in st.secrets and st.secrets["GITHUB_REPO"]:
            return str(st.secrets["GITHUB_REPO"]).strip()
    except Exception:
        pass
    return os.getenv("GITHUB_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO


def _read_local_json(filename: str):
    path = BASE_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_github_raw_json(filename: str):
    repo = _get_repo_slug()
    raw_url = f"https://raw.githubusercontent.com/{repo}/main/{filename}?t={int(time.time())}"
    try:
        req = urllib.request.Request(raw_url, headers={"User-Agent": "PulseAI-Dashboard"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _load_state_json(filename: str, default, prefer_remote: bool = True):
    if prefer_remote:
        remote_data = _read_github_raw_json(filename)
        if remote_data is not None:
            return remote_data

    local_data = _read_local_json(filename)
    if local_data is not None:
        return local_data

    if not prefer_remote:
        remote_data = _read_github_raw_json(filename)
        if remote_data is not None:
            return remote_data

    return default


def refresh_engine_from_disk(eng: AdaptiveTradeEngine, prefer_remote: bool = True) -> AdaptiveTradeEngine:
    eng.ledger = _load_state_json("trade_history.json", [], prefer_remote=prefer_remote)
    eng.portfolio = _load_state_json("portfolio_state.json", eng.portfolio, prefer_remote=prefer_remote)
    eng.weights = _load_state_json("model_weights.json", eng.weights, prefer_remote=prefer_remote)
    eng.shadow_ledger = _load_state_json("shadow_ledger.json", getattr(eng, "shadow_ledger", []), prefer_remote=prefer_remote)
    eng.sector_cache = _load_state_json("sector_cache.json", getattr(eng, "sector_cache", {}), prefer_remote=prefer_remote)
    if hasattr(eng, "_recalculate_cash") and isinstance(eng.portfolio, dict):
        eng._recalculate_cash(eng.portfolio)
    return eng


def persist_portfolio_to_github(portfolio_dict: dict) -> bool:
    token = ""
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
    except Exception:
        token = ""
    token = token or os.getenv("GITHUB_TOKEN", "")
    repo = _get_repo_slug()
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
        return "🟢", "Bullish Market", "rgba(16, 185, 129, 0.16)", "#34d399", "rgba(16, 185, 129, 0.40)"
    if r in ("BEAR_TREND", "HIGH_VOLATILITY_PANIC"):
        return "🔴", "Caution / Bearish", "rgba(244, 63, 94, 0.16)", "#fb7185", "rgba(244, 63, 94, 0.40)"
    if r == "SIDEWAYS_CHOP":
        return "🟡", "Choppy / Range-Bound", "rgba(245, 158, 11, 0.16)", "#fbbf24", "rgba(245, 158, 11, 0.40)"
    return "⚪", "Balanced Market", "rgba(148, 163, 184, 0.16)", "#cbd5e1", "rgba(148, 163, 184, 0.35)"


def humanize_signal(orb_sig: str, mtf_trend: str) -> str:
    orb = str(orb_sig).upper()
    mtf = str(mtf_trend).upper()
    if "BREAKOUT_UP" in orb:
        orb_txt = "Broke above morning range"
    elif "BREAKDOWN_DOWN" in orb:
        orb_txt = "Broke below morning range"
    else:
        orb_txt = "Inside morning range"

    if mtf == "BULLISH":
        mtf_txt = "15m uptrend"
    elif mtf == "BEARISH":
        mtf_txt = "15m downtrend"
    else:
        mtf_txt = "15m neutral"
    return f"{orb_txt} • {mtf_txt}"


def humanize_arm(arm_name: str) -> str:
    mapping = {
        "MOMENTUM_BREAKOUT": "Volume & Price Breakout",
        "NEWS_CATALYST": "News Reaction Setup",
        "PREMARKET_GAP": "Opening Gap Move",
        "RELATIVE_STRENGTH_LEADER": "Market Outperformer",
    }
    return mapping.get(str(arm_name).upper(), str(arm_name).replace("_", " ").title())


def humanize_archetype(arch_name: str) -> str:
    mapping = {
        "LARGE_CAP_STEADY": "Steady Large-Cap",
        "MOMENTUM_MOVER": "Active Momentum Stock",
        "HIGH_VOL_MIDCAP": "Fast-Moving Midcap",
    }
    return mapping.get(str(arch_name).upper(), str(arch_name).replace("_", " ").title())


if "pulse_engine" not in st.session_state:
    st.session_state.pulse_engine = AdaptiveTradeEngine()

engine = refresh_engine_from_disk(st.session_state.pulse_engine, prefer_remote=True)
perf = engine.get_performance_metrics()
trades = engine.ledger if isinstance(engine.ledger, list) else []
open_trades = [t for t in trades if str(t.get("status", "")).upper() == "OPEN"]
closed_trades = [t for t in trades if str(t.get("status", "")).upper() != "OPEN"]

initial_cap = float(perf.get("initial_capital", 150000.0))
manual_dep = float(perf.get("manual_deposits", 0.0))
total_contrib = float(perf.get("total_capital_contributed", initial_cap + manual_dep))
current_eq = float(perf.get("current_equity", total_contrib))
avail_cash = float(perf.get("available_cash", current_eq))
alloc_cap = float(perf.get("allocated_capital", 0.0))
unrl_total = float(perf.get("unrealized_pnl_inr", 0.0))
win_rate_val = float(perf.get("win_rate", 0.0))
total_pnl_pct = float(perf.get("total_pnl_pct", 0.0))
total_pnl_inr = float(perf.get("total_pnl_inr", 0.0))
total_taxes = float(perf.get("total_taxes_paid", engine.portfolio.get("total_taxes_paid", 0.0)))
shadow_learned = int(perf.get("shadow_samples_learned", engine.weights.get("shadow_samples_learned", 0)))

regime_info = engine.weights.get("last_market_regime", {})
regime_name = regime_info.get("regime", "NEUTRAL")
nifty_ltp = float(regime_info.get("nifty_ltp", 0.0))
nifty_chg = float(regime_info.get("nifty_chg_pct", 0.0))
vix_ltp = float(regime_info.get("vix_ltp", 14.0))
vix_chg = float(regime_info.get("vix_chg_pct", 0.0))
fii_net = float(regime_info.get("fii_net_cr", 0.0))
size_mult = float(regime_info.get("size_multiplier", 1.0))
macro_info = regime_info.get("global_macro", {})
macro_summary = macro_info.get("summary", "Macro Neutral")
exp_info = regime_info.get("expiry_status", {})
exp_note = exp_info.get("note", "Normal NSE session")
ml_summary = engine.weights.get("ml_model_summary", "Collecting walk-forward samples")

r_icon, r_friendly, r_bg, r_fg, r_border = get_regime_style(regime_name)
nifty_str = f"Nifty {nifty_ltp:,.2f} ({nifty_chg:+.2f}%)" if nifty_ltp > 0 else f"Nifty {nifty_chg:+.2f}%"
fii_str = f" • FII Net ₹{fii_net:+,.0f} Cr" if fii_net != 0 else ""
regime_badge_text = f"{r_icon} {r_friendly} • {nifty_str} • Volatility Index {vix_ltp:.2f}{fii_str}"

raw_wl = _load_state_json("morning_watchlist.json", [], prefer_remote=True)
watchlist_data = raw_wl.get("items", []) if isinstance(raw_wl, dict) else (raw_wl if isinstance(raw_wl, list) else [])

# ---------------------------------------------------------
# SIDEBAR: WALLET, MARKET MOOD, DISPLAY PREFS & MANUAL FUNDS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 💼 Wallet & Controls")
    auto_live_sync = st.toggle("⚡ Auto-Refresh Every 30s", value=True, key="auto_live_sync")
    show_quant_jargon = st.toggle("🔬 Show Raw Quant Codes", value=False, key="show_quant_jargon", help="Toggle between plain-English descriptions and raw technical/quant labels.")
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

    st.metric(
        label="Total Portfolio Value (After Taxes)",
        value=f"₹{current_eq:,.0f}",
        delta=f"{total_pnl_pct:+.2f}% net return",
    )
    st.metric(label="Free Cash Ready to Trade", value=f"₹{avail_cash:,.0f}")
    st.metric(label="Money Currently in Trades", value=f"₹{alloc_cap:,.0f}")
    st.caption(
        f"Starting Capital: ₹{initial_cap:,.0f} • "
        f"Added Funds: ₹{manual_dep:,.0f} • "
        f"NSE Taxes/Fees: ₹{total_taxes:,.0f}"
    )

    st.divider()
    st.markdown("### 🌐 Market & Risk Context")
    st.markdown(
        f"""
        <div style="background:{r_bg}; border:1px solid {r_border}; border-radius:12px; padding:12px 14px; margin-bottom:8px;">
            <div style="color:{r_fg}; font-weight:700; font-size:0.94rem;">{r_icon} {r_friendly} ({regime_name})</div>
            <div style="color:#e2e8f0; font-size:0.83rem; margin-top:5px;">{nifty_str}</div>
            <div style="color:#cbd5e1; font-size:0.82rem;">VIX: {vix_ltp:.2f} ({vix_chg:+.1f}%) • Position Sizing: {int(size_mult * 100)}%</div>
            <div style="color:#cbd5e1; font-size:0.80rem; margin-top:4px;">FII Cash Flow: ₹{fii_net:+,.0f} Cr</div>
            <div style="color:#cbd5e1; font-size:0.79rem; margin-top:4px;">Global Cues: {macro_summary}</div>
            <div style="color:#fbbf24; font-size:0.79rem; margin-top:4px;">Session Note: {exp_note}</div>
            <div style="color:#94a3b8; font-size:0.76rem; margin-top:4px;">AI Model: {ml_summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### ➕ Add Capital Manually")
    st.caption("Only you can add money to the wallet. The AI never adds funds automatically.")
    with st.form("manual_deposit_form", clear_on_submit=True):
        deposit_amt = st.number_input(
            "Deposit Amount (₹)",
            min_value=1000,
            max_value=10000000,
            value=25000,
            step=5000,
        )
        deposit_note = st.text_input("Note (Optional)", placeholder="e.g., Added extra trading funds")
        if st.form_submit_button("Deposit Into Wallet", use_container_width=True):
            updated_port = engine.add_manual_funds(
                deposit_amt,
                note=deposit_note or "Manual deposit via Dashboard",
                notify_telegram=True,
            )
            synced = persist_portfolio_to_github(updated_port)
            if synced:
                st.success(f"Added ₹{deposit_amt:,.0f} and synced to GitHub.")
            else:
                st.success(f"Added ₹{deposit_amt:,.0f} to your wallet.")
            st.rerun()

    st.divider()
    with st.expander("⚙️ Advanced Flags (Keep OFF)", expanded=False):
        st.caption("Automation runs via GitHub Actions / Worker. Keep manual browser overrides off.")
        st.toggle("Enable Browser Autopilot", value=False, key="ui_autopilot")
        st.checkbox("Simulate After-Hours Entry", value=False, key="ui_sim")

# ---------------------------------------------------------
# TOP BAR, HERO HEADER & KPI SUMMARY
# ---------------------------------------------------------
now_ist_dt = datetime.now(IST)
now_ist_str = now_ist_dt.strftime("%d %b %Y, %I:%M %p IST")
mins_now = now_ist_dt.hour * 60 + now_ist_dt.minute
market_open_now = (now_ist_dt.weekday() < 5) and ((9 * 60 + 15) <= mins_now <= (15 * 60 + 30))
session_badge = "🟢 NSE Session Live" if market_open_now else "🌙 NSE Market Closed"

st.markdown(
    f"""
    <div class="top-ticker-bar">
        <div class="brand-mark">
            <span>⚡ PULSEAI</span>
            <span class="brand-badge">NSE Intraday AI</span>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center;">
            <span class="meta-chip">{session_badge}</span>
            <span class="meta-chip">🕒 {now_ist_str}</span>
            <span class="meta-chip">🔄 Sync: {"30s Auto" if auto_live_sync else "Manual"}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-banner">
        <h1 class="hero-title">Smart NSE Portfolio & AI Trade Monitor</h1>
        <p class="hero-subtitle">
            Tracks intraday equity calls, automatic 15:15 IST square-offs, net after-tax profit & loss, and how the AI adapts from real and simulated setups—in plain, clear numbers.
        </p>
        <div class="pill-row">
            <span class="pill pill-ai">● AI Brain Gen #{engine.weights.get('generation', 1)} • {shadow_learned} Shadow Setups Learned • {now_ist_str}</span>
            <span class="pill" style="background:{r_bg}; color:{r_fg}; border:1px solid {r_border};">{regime_badge_text}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

drift_msg = engine.weights.get("drift_alert", "")
if drift_msg:
    st.warning(f"⚡ **Market Shift Detected:** {drift_msg}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Portfolio Value", f"₹{current_eq:,.0f}", f"Capital Added: ₹{total_contrib:,.0f}")
k2.metric("Cash Available", f"₹{avail_cash:,.0f}", f"Locked in Trades: ₹{alloc_cap:,.0f}")
k3.metric("Open Trade P&L", f"₹{unrl_total:+,.0f}", f"{len(open_trades)} Active Trades")
k4.metric("Win Rate (After Tax)", f"{win_rate_val:.1f}%", f"Net Realized: ₹{total_pnl_inr:+,.0f}")

st.write("")

# ---------------------------------------------------------
# SEARCH & FILTER BAR FOR TRADES
# ---------------------------------------------------------
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
f_col1, f_col2, f_col3, f_col4 = st.columns([2.2, 1.3, 1.3, 1.6])
with f_col1:
    search_query = st.text_input(
        "🔍 Search Stock or Sector",
        placeholder="Type e.g. TATASTEEL, HDFCBANK, Energy, Technology...",
        key="trade_search",
    ).strip().upper()
with f_col2:
    side_filter = st.selectbox(
        "Direction",
        options=["All", "BUY", "SELL (SHORT)"],
        key="trade_side_filter",
    )
with f_col3:
    status_filter = st.selectbox(
        "Trade State",
        options=["All", "Open Only", "Settled Only"],
        key="trade_status_filter",
    )
with f_col4:
    min_conv = st.slider(
        "Min AI Confidence",
        min_value=0,
        max_value=95,
        value=0,
        step=5,
        key="trade_min_conv",
    )
st.markdown("</div>", unsafe_allow_html=True)


def _matches_trade_filters(t: dict, is_open: bool) -> bool:
    if status_filter == "Open Only" and not is_open:
        return False
    if status_filter == "Settled Only" and is_open:
        return False
    if side_filter != "All" and str(t.get("action", "")).upper() != side_filter:
        return False
    if float(t.get("conviction", 0.0)) < float(min_conv):
        return False
    if search_query:
        sym = str(t.get("symbol", "")).replace(".NS", "").upper()
        sec = str(t.get("sector", "")).upper()
        reason = str(t.get("headline", "")).upper()
        if search_query not in sym and search_query not in sec and search_query not in reason:
            return False
    return True


filtered_open = [t for t in open_trades if _matches_trade_filters(t, True)]
filtered_closed = [t for t in closed_trades if _matches_trade_filters(t, False)]

# ---------------------------------------------------------
# VISUAL CHARTS (EQUITY & DAILY P&L + TRADE OUTCOMES)
# ---------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 📈 Portfolio Equity Trend")
    eq_log = engine.portfolio.get("daily_equity_log", [])
    if eq_log and len(eq_log) >= 1:
        eq_df = pd.DataFrame(eq_log)
        eq_df["equity"] = pd.to_numeric(eq_df.get("equity", current_eq), errors="coerce").fillna(current_eq)
        eq_df["date"] = eq_df.get("date", now_ist_dt.strftime("%Y-%m-%d")).astype(str)
        eq_chart = (
            alt.Chart(eq_df)
            .mark_area(
                line={"color": "#38bdf8", "strokeWidth": 2.5},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(56, 189, 248, 0.35)", offset=0),
                        alt.GradientStop(color="rgba(56, 189, 248, 0.02)", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
            )
            .encode(
                x=alt.X("date:N", title="Date"),
                y=alt.Y("equity:Q", title="Equity (₹)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("date:N", title="Date"),
                    alt.Tooltip("equity:Q", title="Equity (₹)", format=",.0f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(eq_chart, use_container_width=True)
    else:
        st.info("Equity history will appear here as trading days complete.")

with chart_col2:
    st.markdown("#### 📊 Recent Settled Trade P&L (After Taxes)")
    if closed_trades:
        recent_c = closed_trades[-12:]
        pnl_df = pd.DataFrame([
            {
                "Stock": f"{str(t.get('symbol', '')).replace('.NS', '')} ({t.get('date', '')[-5:]})",
                "Net P&L (₹)": round(float(t.get("pnl_inr", 0.0)), 2),
                "Outcome": "Profit" if float(t.get("pnl_inr", 0.0)) >= 0 else "Loss",
            }
            for t in recent_c
        ])
        pnl_chart = (
            alt.Chart(pnl_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Stock:N", sort=None, title="Trade"),
                y=alt.Y("Net P&L (₹):Q", title="Net P&L (₹)"),
                color=alt.Color(
                    "Outcome:N",
                    scale=alt.Scale(domain=["Profit", "Loss"], range=["#10b981", "#f43f5e"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Stock:N", title="Trade"),
                    alt.Tooltip("Net P&L (₹):Q", title="Net P&L (₹)", format="+,.2f"),
                    alt.Tooltip("Outcome:N", title="Result"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(pnl_chart, use_container_width=True)
    else:
        st.info("Settled trade profit/loss bars will appear once intraday positions close.")

st.write("")

# ---------------------------------------------------------
# MAIN TABS: TRADES, AI NOTES, STRATEGY PERFORMANCE & WATCHLIST
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        f"📊 Trades ({len(filtered_open)} Open • {len(filtered_closed)} Settled)",
        "🧠 AI Research & Learning Log",
        "🎯 Strategy & Timing Breakdown",
        f"🌅 Morning Watchlist ({len(watchlist_data)})",
    ]
)

with tab1:
    st.subheader("Active Open Trades")
    if not filtered_open:
        st.info("No open trades match your current filter. Positions close automatically at 15:15 IST or when target/stop rules trigger.")
    else:
        rows = []
        sector_summary = {}
        for t in filtered_open:
            entry = float(t.get("entry_price", 0))
            target = float(t.get("target_1", t.get("initial_target", 0)))
            sl = float(t.get("stop_loss", 0))
            qty = int(t.get("qty", 0))
            invested = float(t.get("capital_used", entry * qty))
            unrl = float(t.get("unrealized_pnl_inr", 0))
            peak = float(t.get("peak_pnl_inr", 0))
            conv = float(t.get("conviction", 0))
            sec = str(t.get("sector", "General"))
            rvol = float(t.get("rvol_normalized", t.get("vol_ratio", 1.0)))
            orb_s = str(t.get("orb_signal", "INSIDE_ORB"))
            mtf_s = str(t.get("mtf_15m_trend", "NEUTRAL"))
            deliv = float(t.get("delivery_pct", 0.0))
            arm_raw = str(t.get("strategy_arm", "MOMENTUM_BREAKOUT"))

            sector_summary.setdefault(sec, {"count": 0, "capital": 0.0})
            sector_summary[sec]["count"] += 1
            sector_summary[sec]["capital"] += invested

            setup_label = f"{mtf_s} / {orb_s}" if show_quant_jargon else humanize_signal(orb_s, mtf_s)
            arm_label = arm_raw if show_quant_jargon else humanize_arm(arm_raw)
            headline = str(t.get("headline", "")).strip()
            note = str(t.get("trade_management_note", "")).strip()
            why_text = f"{headline} — {note}" if (headline and note) else (headline or note)

            rows.append(
                {
                    "Stock": t.get("symbol", "N/A").replace(".NS", ""),
                    "Sector": sec,
                    "Direction": t.get("action", "BUY"),
                    "Chart Signal": setup_label,
                    "Vol / Delivery": f"{rvol:.1f}x avg • {deliv:.0f}% deliv",
                    "Setup Type": arm_label,
                    "Entry": f"₹{entry:,.2f}",
                    "Target": f"₹{target:,.2f}",
                    "Stop-Loss": f"₹{sl:,.2f}",
                    "Qty": qty,
                    "Capital": f"₹{invested:,.0f}",
                    "Live P&L": f"₹{unrl:+,.0f} (Best ₹{peak:+,.0f})",
                    "AI Score": f"{conv:.0f}/100",
                    "Reason & Status": why_text,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        eq = max(current_eq, 1.0)
        sec_tags = [
            f"**{sec}**: {vals['count']}/2 trades (₹{vals['capital']:,.0f} • {vals['capital'] / eq * 100:.0f}%)"
            for sec, vals in sector_summary.items()
        ]
        st.caption("🛡️ **Sector Limit (Max 2 trades & 40% capital per sector):** " + " | ".join(sec_tags))

    st.subheader("Settled Trade History (Net After Taxes & Charges)")
    if not filtered_closed:
        st.caption("No settled trades match your filter yet.")
    else:
        closed_rows = []
        for t in reversed(filtered_closed[-40:]):
            pnl = float(t.get("pnl_inr", 0))
            charges = float(t.get("charges_inr", 0))
            arm_raw = str(t.get("strategy_arm", "—"))
            closed_rows.append(
                {
                    "Date": t.get("date", "—"),
                    "Stock": t.get("symbol", "N/A").replace(".NS", ""),
                    "Sector": t.get("sector", "General"),
                    "Setup Type": arm_raw if show_quant_jargon else humanize_arm(arm_raw),
                    "Direction": t.get("action", "BUY"),
                    "Entry": f"₹{float(t.get('entry_price', 0)):,.2f}",
                    "Exit": f"₹{float(t.get('exit_price', 0)):,.2f}",
                    "Duration": f"{float(t.get('hold_minutes', 0)):.0f}m",
                    "Taxes/Fees": f"₹{charges:.2f}",
                    "Net P&L": f"₹{pnl:+,.0f} ({float(t.get('pnl_pct', 0)):+.2f}%)",
                    "Exit Reason": t.get("status", "CLOSED"),
                    "What AI Learned": t.get("mistake_diagnosis", "—"),
                }
            )
        st.dataframe(pd.DataFrame(closed_rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🧠 Post-Trade AI Notes")
    st.caption("Every time a trade closes, the AI checks news and price action to record what worked or went wrong.")
    lessons = engine.weights.get("trade_lessons_log", [])
    if not lessons:
        st.info("No post-trade notes recorded yet.")
    else:
        for item in lessons[:15]:
            icon = "✅" if float(item.get("pnl_inr", 0)) >= 0 else "🔍"
            regime_tag = f" `[{item['regime']}]`" if item.get("regime") else ""
            st.markdown(f"{icon} **{item.get('timestamp', '')}**{regime_tag} — {item.get('lesson', '')}")

    st.divider()
    st.subheader("👻 Skipped & Filtered Setups (Shadow Practice)")
    st.caption("Setups that didn't pass risk or confidence rules are tracked silently so the AI learns without risking real cash.")
    shadows = getattr(engine, "shadow_ledger", [])
    if not shadows:
        st.caption("No shadow setups recorded yet.")
    else:
        s_rows = [
            {
                "Date": s.get("date", "—"),
                "Stock": s.get("symbol", "").replace(".NS", ""),
                "Direction": s.get("action", "BUY"),
                "Stock Style": s.get("archetype", "—") if show_quant_jargon else humanize_archetype(s.get("archetype", "")),
                "AI Score": s.get("conviction", 0),
                "Why Skipped": s.get("skip_reason", "—"),
                "Status": s.get("status", "SHADOW_OPEN"),
                "Simulated P&L": f"{float(s.get('shadow_pnl_pct', 0)):+.2f}%",
            }
            for s in reversed(shadows[-15:])
        ]
        st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("💰 Manual Deposits")
    deposits = engine.portfolio.get("deposit_history", [])
    if not deposits:
        st.caption("No manual deposits added yet.")
    else:
        st.dataframe(pd.DataFrame(deposits), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🎯 Strategy Performance")
    arms = engine.weights.get("bandit_arms", {})
    if not arms:
        st.info("Strategy stats appear once scans and trades run.")
    else:
        arm_rows = []
        for arm_name, stats in arms.items():
            a = float(stats.get("alpha", 2.0))
            b = float(stats.get("beta", 2.0))
            post_p = (a / (a + b)) * 100.0
            arm_rows.append(
                {
                    "Strategy": arm_name if show_quant_jargon else humanize_arm(arm_name),
                    "Est. Win Likelihood": f"{post_p:.1f}%",
                    "Completed Trades": int(stats.get("trades", 0)),
                    "Net P&L": f"₹{float(stats.get('net_pnl_inr', 0.0)):+,.0f}",
                    "Alpha / Beta": f"{a:.1f} / {b:.1f}",
                }
            )
        st.dataframe(pd.DataFrame(arm_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🧬 Stock Style Performance")
    arch_mem = engine.weights.get("archetype_memory", {})
    if not arch_mem:
        st.info("Stock style stats populate automatically as real and shadow trades settle.")
    else:
        arch_rows = []
        for arch_name, st_a in arch_mem.items():
            w_wins = float(st_a.get("weighted_wins", 1.0))
            w_losses = float(st_a.get("weighted_losses", 1.0))
            wp = (w_wins / max(w_wins + w_losses, 1.0)) * 100.0
            arch_rows.append(
                {
                    "Stock Style": arch_name if show_quant_jargon else humanize_archetype(arch_name),
                    "Win Rate": f"{wp:.1f}%",
                    "Trades": int(st_a.get("trades", 0)),
                    "Stop-Loss Cushion": f"{float(st_a.get('sl_mult_adj', 1.0)):.2f}x",
                    "Score Boost": f"{float(st_a.get('score_adj', 0.0)):+.1f} pts",
                }
            )
        st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("⏱️ Time-of-Day Performance (IST)")
    sess_mem = engine.weights.get("session_memory", {})
    if not sess_mem:
        st.info("Time-of-day stats appear once trades complete across intraday sessions.")
    else:
        sess_rows = [
            {
                "Time Window (IST)": k,
                "Trades": v.get("trades", 0),
                "Win Rate": f"{v.get('win_rate', 0)}%",
                "Net P&L": f"₹{v.get('net_pnl_inr', 0):+,.0f}",
                "Score Boost": f"{v.get('score_adj', 0):+.1f} pts",
            }
            for k, v in sess_mem.items()
        ]
        st.dataframe(pd.DataFrame(sess_rows), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("🌅 Morning Watchlist")
    if not watchlist_data:
        st.info("Watchlist updates automatically during market prep and intraday scans.")
    else:
        wl_df = pd.DataFrame(watchlist_data)
        if search_query and "symbol" in wl_df.columns:
            wl_df = wl_df[wl_df["symbol"].astype(str).str.upper().str.contains(search_query, na=False)]
        st.dataframe(wl_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="app-footer">
        <strong>PulseAI NSE Portfolio Monitor</strong> • Repo: <code>{_get_repo_slug()}</code> •
        Intraday session runs Mon–Fri (09:15–15:25 IST; mandatory square-off from 15:15 IST; NSE closed on listed holidays such as 24 Nov 2026).<br/>
        <em>Disclaimer: This dashboard tracks automated model paper/simulation state and analytical signals for research & learning—not SEBI-registered investment advice.</em>
    </div>
    """,
    unsafe_allow_html=True,
)

if auto_live_sync:
    time.sleep(30)
    st.rerun()