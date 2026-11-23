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
    page_title="PulseAI — NSE AI Trading App",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        .mono {
            font-family: 'JetBrains Mono', monospace !important;
            font-variant-numeric: tabular-nums;
        }

        .stApp {
            background: #0b0f19;
            color: #f1f5f9;
        }

        .block-container {
            padding-top: 3.6rem !important;
            padding-bottom: 3rem !important;
            padding-left: clamp(0.75rem, 2.5vw, 2rem) !important;
            padding-right: clamp(0.75rem, 2.5vw, 2rem) !important;
            max-width: 1180px;
        }

        /* Mobile-first App Top Navigation Bar */
        .app-navbar {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            background: #121826;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            padding: 14px 18px;
            margin-bottom: 14px;
        }

        .app-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
            flex: 1 1 210px;
        }

        .app-logo {
            width: 36px;
            height: 36px;
            flex-shrink: 0;
            border-radius: 10px;
            background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            color: #fff;
            font-size: 1rem;
        }

        .app-title-wrap {
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        .app-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.25;
        }

        .app-sub {
            font-size: 0.76rem;
            color: #94a3b8;
            line-height: 1.3;
        }

        .nav-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
        }

        .chip {
            font-size: 0.74rem;
            font-weight: 600;
            padding: 5px 10px;
            border-radius: 999px;
            background: #1a2234;
            color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.16);
            white-space: nowrap;
        }

        /* Hero Portfolio Card */
        .wallet-hero {
            background: linear-gradient(145deg, #141c2e 0%, #0f1624 100%);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 20px;
            padding: clamp(16px, 3vw, 24px);
            margin-bottom: 14px;
        }

        .wallet-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .wallet-amount-row {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 12px;
            margin-top: 6px;
        }

        .wallet-amount {
            font-family: 'JetBrains Mono', monospace;
            font-size: clamp(1.75rem, 4.5vw, 2.45rem);
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }

        .return-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.84rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 999px;
        }

        /* Responsive Stat Grid */
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
            gap: 10px;
            margin-top: 16px;
        }

        .stat-box {
            background: rgba(11, 15, 25, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 14px;
            padding: 12px 14px;
        }

        .stat-box-title {
            font-size: 0.75rem;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .stat-box-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.08rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .stat-box-sub {
            font-size: 0.72rem;
            color: #64748b;
            margin-top: 2px;
        }

        /* Filtered P&L Strip */
        .filter-summary-box {
            background: #121826;
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 16px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 14px;
        }

        .filter-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: 10px;
            margin-top: 8px;
        }

        /* Mobile-Friendly Trade Position Card */
        .trade-card {
            background: #121826;
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 16px;
            padding: 15px 16px;
            margin-bottom: 12px;
            transition: border-color 0.2s ease;
        }

        .trade-card:hover {
            border-color: rgba(56, 189, 248, 0.38);
        }

        .tc-top {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: flex-start;
            gap: 8px;
        }

        .tc-sym {
            font-size: 1.08rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.01em;
        }

        .tc-sector {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 2px;
        }

        .side-pill {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 8px;
            letter-spacing: 0.03em;
        }

        .tc-pnl-wrap {
            text-align: right;
        }

        .tc-pnl {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .tc-pnl-sub {
            font-size: 0.72rem;
            color: #94a3b8;
        }

        .tc-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 8px;
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid rgba(148, 163, 184, 0.1);
        }

        .tc-m-label {
            font-size: 0.69rem;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
        }

        .tc-m-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.86rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-top: 1px;
        }

        .tc-reason {
            margin-top: 10px;
            padding: 8px 11px;
            border-radius: 10px;
            background: rgba(30, 41, 59, 0.55);
            font-size: 0.8rem;
            color: #cbd5e1;
            line-height: 1.45;
            word-break: break-word;
        }

        .tc-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }

        .mini-tag {
            font-size: 0.71rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 6px;
            background: rgba(56, 189, 248, 0.1);
            color: #7dd3fc;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: #121826;
            padding: 6px;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            overflow-x: auto;
        }

        .stTabs [data-baseweb="tab"] {
            height: 38px;
            padding: 6px 14px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.84rem;
            color: #94a3b8;
            white-space: nowrap;
        }

        .stTabs [aria-selected="true"] {
            background: #1e293b !important;
            color: #38bdf8 !important;
        }

        /* Footer */
        .app-footer {
            margin-top: 2rem;
            padding: 16px 18px;
            border-radius: 16px;
            background: #121826;
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: #94a3b8;
            font-size: 0.78rem;
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
        req = urllib.request.Request(raw_url, headers={"User-Agent": "PulseAI-App"})
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
        "User-Agent": "PulseAI-App",
    }
    try:
        req = urllib.request.Request(api_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=8) as resp:
            existing = json.loads(resp.read().decode("utf-8"))
        sha = existing.get("sha")

        encoded = base64.b64encode(json.dumps(portfolio_dict, indent=2).encode("utf-8")).decode("utf-8")
        payload = json.dumps({
            "message": "chore(user): manual capital deposit via PulseAI app",
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
        return "🟢", "Bullish Market", "rgba(16, 185, 129, 0.16)", "#34d399", "rgba(16, 185, 129, 0.35)"
    if r in ("BEAR_TREND", "HIGH_VOLATILITY_PANIC"):
        return "🔴", "Bearish / Cautious", "rgba(244, 63, 94, 0.16)", "#fb7185", "rgba(244, 63, 94, 0.35)"
    if r == "SIDEWAYS_CHOP":
        return "🟡", "Sideways / Choppy", "rgba(245, 158, 11, 0.16)", "#fbbf24", "rgba(245, 158, 11, 0.35)"
    return "⚪", "Balanced Market", "rgba(148, 163, 184, 0.16)", "#cbd5e1", "rgba(148, 163, 184, 0.30)"


def humanize_signal(orb_sig: str, mtf_trend: str) -> str:
    orb = str(orb_sig).upper()
    mtf = str(mtf_trend).upper()
    if "BREAKOUT_UP" in orb:
        orb_txt = "Broke above opening range"
    elif "BREAKDOWN_DOWN" in orb:
        orb_txt = "Broke below opening range"
    else:
        orb_txt = "Trading inside morning range"

    if mtf == "BULLISH":
        mtf_txt = "15m uptrend"
    elif mtf == "BEARISH":
        mtf_txt = "15m downtrend"
    else:
        mtf_txt = "15m sideways"
    return f"{orb_txt} ({mtf_txt})"


def humanize_arm(arm_name: str) -> str:
    mapping = {
        "MOMENTUM_BREAKOUT": "Volume & Price Breakout",
        "NEWS_CATALYST": "News Catalyst Move",
        "PREMARKET_GAP": "Morning Gap Play",
        "RELATIVE_STRENGTH_LEADER": "Sector Outperformer",
    }
    return mapping.get(str(arm_name).upper(), str(arm_name).replace("_", " ").title())


def humanize_archetype(arch_name: str) -> str:
    mapping = {
        "LARGE_CAP_STEADY": "Steady Large-Cap",
        "MOMENTUM_MOVER": "High-Momentum Stock",
        "HIGH_VOL_MIDCAP": "Fast-Moving Midcap",
    }
    return mapping.get(str(arch_name).upper(), str(arch_name).replace("_", " ").title())


def _extract_trade_date_str(trade: dict) -> str:
    raw_d = str(trade.get("date") or trade.get("entry_date") or "").strip()
    if raw_d:
        return raw_d[:10]
    raw_ts = str(trade.get("entry_time") or trade.get("timestamp") or "").strip()
    return raw_ts[:10] if len(raw_ts) >= 10 else ""


if "pulse_engine" not in st.session_state:
    st.session_state.pulse_engine = AdaptiveTradeEngine()

# ---------------------------------------------------------
# SIDEBAR: APP SETTINGS & MANUAL CAPITAL DEPOSIT
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ App Preferences")
    auto_live_sync = st.toggle("Live Auto-Sync (30s)", value=True, key="auto_live_sync")
    show_quant_jargon = st.toggle("Show Technical Quant Codes", value=False, key="show_quant_jargon")
    view_mode = st.radio("Trade Display Style", ["Mobile App Cards", "Compact Table"], index=0, key="view_mode")
    if st.button("Refresh Data Now", icon=":material/refresh:", use_container_width=True):
        st.rerun()

    st.divider()
    st.markdown("### ➕ Add Capital Manually")
    st.caption("Only you can deposit funds here. The AI never adds capital on its own.")
    with st.form("manual_deposit_form", clear_on_submit=True):
        deposit_amt = st.number_input(
            "Amount (₹)",
            min_value=1000,
            max_value=10000000,
            value=25000,
            step=5000,
        )
        deposit_note = st.text_input("Note (Optional)", placeholder="e.g., Added trading capital")
        if st.form_submit_button("Deposit Into Wallet", use_container_width=True):
            eng_tmp = refresh_engine_from_disk(st.session_state.pulse_engine, prefer_remote=True)
            updated_port = eng_tmp.add_manual_funds(
                deposit_amt,
                note=deposit_note or "Manual deposit via PulseAI App",
                notify_telegram=True,
            )
            synced = persist_portfolio_to_github(updated_port)
            if synced:
                st.success(f"Added ₹{deposit_amt:,.0f} and synced to GitHub.")
            else:
                st.success(f"Added ₹{deposit_amt:,.0f} to your wallet.")
            st.rerun()


@st.fragment(run_every="30s")
def render_live_app():
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
    fii_net = float(regime_info.get("fii_net_cr", 0.0))
    size_mult = float(regime_info.get("size_multiplier", 1.0))
    macro_summary = regime_info.get("global_macro", {}).get("summary", "Macro Neutral")
    exp_note = regime_info.get("expiry_status", {}).get("note", "Normal NSE session")
    ml_summary = engine.weights.get("ml_model_summary", "Learning from live & shadow setups")

    r_icon, r_friendly, r_bg, r_fg, r_border = get_regime_style(regime_name)
    nifty_str = f"Nifty {nifty_ltp:,.2f} ({nifty_chg:+.2f}%)" if nifty_ltp > 0 else f"Nifty {nifty_chg:+.2f}%"

    raw_wl = _load_state_json("morning_watchlist.json", [], prefer_remote=True)
    watchlist_data = raw_wl.get("items", []) if isinstance(raw_wl, dict) else (raw_wl if isinstance(raw_wl, list) else [])

    now_ist_dt = datetime.now(IST)
    now_ist_str = now_ist_dt.strftime("%d %b %Y • %I:%M %p IST")
    mins_now = now_ist_dt.hour * 60 + now_ist_dt.minute
    market_open_now = (now_ist_dt.weekday() < 5) and ((9 * 60 + 15) <= mins_now <= (15 * 60 + 30))
    session_badge = "🟢 NSE Live" if market_open_now else "🌙 NSE Closed"

    # ---------------------------------------------------------
    # TOP APP BAR
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <div class="app-navbar">
            <div class="app-brand">
                <div class="app-logo">⚡</div>
                <div class="app-title-wrap">
                    <span class="app-title">PulseAI</span>
                    <span class="app-sub">Autonomous NSE Intraday Assistant</span>
                </div>
            </div>
            <div class="nav-badges">
                <span class="chip">{session_badge}</span>
                <span class="chip" style="background:{r_bg}; color:{r_fg}; border-color:{r_border};">{r_icon} {r_friendly}</span>
                <span class="chip">🕒 {now_ist_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # HERO PORTFOLIO WALLET CARD (MOBILE-FIRST 2x2 GRID)
    # ---------------------------------------------------------
    ret_bg = "rgba(16, 185, 129, 0.16)" if total_pnl_pct >= 0 else "rgba(244, 63, 94, 0.16)"
    ret_fg = "#34d399" if total_pnl_pct >= 0 else "#fb7185"
    ret_arrow = "▲" if total_pnl_pct >= 0 else "▼"
    unrl_color = "#34d399" if unrl_total >= 0 else "#fb7185"
    real_color = "#34d399" if total_pnl_inr >= 0 else "#fb7185"

    st.markdown(
        f"""
        <div class="wallet-hero">
            <div class="wallet-label">Net Portfolio Value (After Taxes & Slippage)</div>
            <div class="wallet-amount-row">
                <span class="wallet-amount">₹{current_eq:,.0f}</span>
                <span class="return-badge" style="background:{ret_bg}; color:{ret_fg};">{ret_arrow} {total_pnl_pct:+.2f}% (₹{current_eq - total_contrib:+,.0f})</span>
            </div>
            <div style="font-size:0.78rem; color:#94a3b8; margin-top:6px;">
                {nifty_str} • VIX {vix_ltp:.2f} • Sizing {int(size_mult * 100)}% • {exp_note}
            </div>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-box-title">Free Cash</div>
                    <div class="stat-box-val">₹{avail_cash:,.0f}</div>
                    <div class="stat-box-sub">Ready to deploy</div>
                </div>
                <div class="stat-box">
                    <div class="stat-box-title">In Active Trades</div>
                    <div class="stat-box-val">₹{alloc_cap:,.0f}</div>
                    <div class="stat-box-sub">{len(open_trades)} open position{"s" if len(open_trades) != 1 else ""}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-box-title">Open Trade P&L</div>
                    <div class="stat-box-val" style="color:{unrl_color};">₹{unrl_total:+,.0f}</div>
                    <div class="stat-box-sub">Live unrealized</div>
                </div>
                <div class="stat-box">
                    <div class="stat-box-title">Realized Net P&L</div>
                    <div class="stat-box-val" style="color:{real_color};">₹{total_pnl_inr:+,.0f}</div>
                    <div class="stat-box-sub">Win rate {win_rate_val:.1f}% • Fees ₹{total_taxes:,.0f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    drift_msg = engine.weights.get("drift_alert", "")
    if drift_msg:
        st.warning(f"⚡ **Market Shift Alert:** {drift_msg}")

    # ---------------------------------------------------------
    # MOBILE-FRIENDLY SEARCH, DATE & QUICK FILTER BAR
    # ---------------------------------------------------------
    st.markdown("#### 🔎 Find Trades")
    q_col1, q_col2 = st.columns([2, 1])
    with q_col1:
        search_query = st.text_input(
            "Search stock or sector",
            placeholder="Search e.g. TATASTEEL, HDFCBANK, Energy...",
            label_visibility="collapsed",
            key="search_query",
        ).strip().upper()
    with q_col2:
        quick_filter = st.selectbox(
            "Filter",
            ["All Trades", "Open Only", "Settled Only", "Profitable Only", "BUY Only", "SELL (SHORT) Only"],
            label_visibility="collapsed",
            key="quick_filter",
        )

    available_dates = sorted(
        {d for d in (_extract_trade_date_str(t) for t in trades) if d},
        reverse=True,
    )
    selected_dates = st.multiselect(
        "Filter by Single or Multiple Dates (leave empty for all days)",
        options=available_dates,
        default=[],
        placeholder="Select one or more dates (e.g. 2026-11-23)",
        key="selected_trade_dates",
    )
    selected_date_set = set(selected_dates)

    def _keep_trade(t: dict, is_open: bool) -> bool:
        if selected_date_set and _extract_trade_date_str(t) not in selected_date_set:
            return False
        if quick_filter == "Open Only" and not is_open:
            return False
        if quick_filter == "Settled Only" and is_open:
            return False
        if quick_filter == "BUY Only" and str(t.get("action", "")).upper() != "BUY":
            return False
        if quick_filter == "SELL (SHORT) Only" and "SELL" not in str(t.get("action", "")).upper():
            return False
        if quick_filter == "Profitable Only":
            pnl_val = float(t.get("unrealized_pnl_inr", 0.0) if is_open else t.get("pnl_inr", 0.0))
            if pnl_val <= 0:
                return False
        if search_query:
            sym = str(t.get("symbol", "")).replace(".NS", "").upper()
            sec = str(t.get("sector", "")).upper()
            headline = str(t.get("headline", "")).upper()
            if search_query not in sym and search_query not in sec and search_query not in headline:
                return False
        return True

    f_open = [t for t in open_trades if _keep_trade(t, True)]
    f_closed = [t for t in closed_trades if _keep_trade(t, False)]

    # Dynamic Profit / Loss summary for the currently applied filter(s)
    f_closed_pnl = sum(float(t.get("pnl_inr", 0.0)) for t in f_closed)
    f_open_pnl = sum(float(t.get("unrealized_pnl_inr", 0.0)) for t in f_open)
    f_combined_pnl = f_closed_pnl + f_open_pnl
    f_wins = sum(1 for t in f_closed if float(t.get("pnl_inr", 0.0)) > 0)
    f_win_rate = (f_wins / len(f_closed) * 100.0) if f_closed else 0.0
    comb_color = "#34d399" if f_combined_pnl >= 0 else "#fb7185"
    closed_color = "#34d399" if f_closed_pnl >= 0 else "#fb7185"

    if selected_dates:
        date_scope_label = f"Selected Date{'s' if len(selected_dates) > 1 else ''}: {', '.join(selected_dates)}"
    else:
        date_scope_label = "All Recorded Dates"

    st.markdown(
        f"""
        <div class="filter-summary-box">
            <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:6px;">
                <span style="font-size:0.82rem; font-weight:700; color:#e2e8f0;">📋 Filtered P&L Summary • {date_scope_label}</span>
                <span style="font-size:0.74rem; color:#94a3b8;">{len(f_open) + len(f_closed)} matching trade(s)</span>
            </div>
            <div class="filter-summary-grid">
                <div class="stat-box">
                    <div class="stat-box-title">Total Filtered P&L</div>
                    <div class="stat-box-val" style="color:{comb_color};">₹{f_combined_pnl:+,.0f}</div>
                    <div class="stat-box-sub">Settled + Open in filter</div>
                </div>
                <div class="stat-box">
                    <div class="stat-box-title">Settled P&L (Filter)</div>
                    <div class="stat-box-val" style="color:{closed_color};">₹{f_closed_pnl:+,.0f}</div>
                    <div class="stat-box-sub">{len(f_closed)} settled trade(s)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-box-title">Filter Win Rate</div>
                    <div class="stat-box-val">{f_win_rate:.1f}%</div>
                    <div class="stat-box-sub">{f_wins} winning settled</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # PRIMARY APP TABS
    # ---------------------------------------------------------
    t_positions, t_insights, t_ai, t_watchlist = st.tabs(
        [
            f"📱 Positions ({len(f_open)}/{len(f_closed)})",
            "📈 Performance",
            "🧠 AI Brain",
            f"🌅 Watchlist ({len(watchlist_data)})",
        ]
    )

    with t_positions:
        st.markdown(f"**Active Positions ({len(f_open)})**")
        if not f_open:
            st.info("No active positions match the selected filter.")
        else:
            if view_mode == "Mobile App Cards":
                for t in f_open:
                    sym = str(t.get("symbol", "N/A")).replace(".NS", "")
                    sec = str(t.get("sector", "General"))
                    action = str(t.get("action", "BUY")).upper()
                    is_buy = action == "BUY"
                    side_bg = "rgba(16, 185, 129, 0.16)" if is_buy else "rgba(244, 63, 94, 0.16)"
                    side_fg = "#34d399" if is_buy else "#fb7185"

                    entry = float(t.get("entry_price", 0))
                    target = float(t.get("target_1", t.get("initial_target", 0)))
                    sl = float(t.get("stop_loss", 0))
                    qty = int(t.get("qty", 0))
                    invested = float(t.get("capital_used", entry * qty))
                    unrl = float(t.get("unrealized_pnl_inr", 0))
                    peak = float(t.get("peak_pnl_inr", 0))
                    conv = float(t.get("conviction", 0))
                    rvol = float(t.get("rvol_normalized", t.get("vol_ratio", 1.0)))
                    deliv = float(t.get("delivery_pct", 0.0))
                    orb_s = str(t.get("orb_signal", "INSIDE_ORB"))
                    mtf_s = str(t.get("mtf_15m_trend", "NEUTRAL"))
                    arm_raw = str(t.get("strategy_arm", "MOMENTUM_BREAKOUT"))

                    pnl_fg = "#34d399" if unrl >= 0 else "#fb7185"
                    setup_txt = f"{mtf_s} / {orb_s}" if show_quant_jargon else humanize_signal(orb_s, mtf_s)
                    arm_txt = arm_raw if show_quant_jargon else humanize_arm(arm_raw)
                    headline = str(t.get("headline", "")).strip()
                    note = str(t.get("trade_management_note", "")).strip()
                    reason_line = f"{headline} — {note}" if (headline and note) else (headline or note or "Monitoring live price & volume.")

                    st.markdown(
                        f"""
                        <div class="trade-card">
                            <div class="tc-top">
                                <div>
                                    <span class="tc-sym">{sym}</span>
                                    <span class="side-pill" style="background:{side_bg}; color:{side_fg}; margin-left:6px;">{action}</span>
                                    <div class="tc-sector">{_extract_trade_date_str(t) or "—"} • {sec} • {qty} shares (₹{invested:,.0f})</div>
                                </div>
                                <div class="tc-pnl-wrap">
                                    <div class="tc-pnl" style="color:{pnl_fg};">₹{unrl:+,.0f}</div>
                                    <div class="tc-pnl-sub">Peak ₹{peak:+,.0f} • Score {conv:.0f}/100</div>
                                </div>
                            </div>
                            <div class="tc-metrics">
                                <div>
                                    <div class="tc-m-label">Entry</div>
                                    <div class="tc-m-val">₹{entry:,.2f}</div>
                                </div>
                                <div>
                                    <div class="tc-m-label">Target</div>
                                    <div class="tc-m-val" style="color:#34d399;">₹{target:,.2f}</div>
                                </div>
                                <div>
                                    <div class="tc-m-label">Stop-Loss</div>
                                    <div class="tc-m-val" style="color:#fb7185;">₹{sl:,.2f}</div>
                                </div>
                                <div>
                                    <div class="tc-m-label">Volume / Deliv</div>
                                    <div class="tc-m-val">{rvol:.1f}x • {deliv:.0f}%</div>
                                </div>
                            </div>
                            <div class="tc-tags">
                                <span class="mini-tag">📌 {arm_txt}</span>
                                <span class="mini-tag">📊 {setup_txt}</span>
                            </div>
                            <div class="tc-reason">💡 {reason_line}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                table_rows = [
                    {
                        "Date": _extract_trade_date_str(t) or "—",
                        "Stock": str(t.get("symbol", "")).replace(".NS", ""),
                        "Side": t.get("action", "BUY"),
                        "Qty": int(t.get("qty", 0)),
                        "Entry (₹)": round(float(t.get("entry_price", 0)), 2),
                        "Target (₹)": round(float(t.get("target_1", t.get("initial_target", 0))), 2),
                        "Stop-Loss (₹)": round(float(t.get("stop_loss", 0)), 2),
                        "Live P&L (₹)": round(float(t.get("unrealized_pnl_inr", 0)), 2),
                        "Score": f"{float(t.get('conviction', 0)):.0f}/100",
                    }
                    for t in f_open
                ]
                st.dataframe(pd.DataFrame(table_rows), hide_index=True)

        st.write("")
        st.markdown(f"**Settled History ({len(f_closed)})**")
        if not f_closed:
            st.caption("No settled trades match the selected filter.")
        else:
            if view_mode == "Mobile App Cards":
                for t in reversed(f_closed[-25:]):
                    sym = str(t.get("symbol", "N/A")).replace(".NS", "")
                    sec = str(t.get("sector", "General"))
                    action = str(t.get("action", "BUY")).upper()
                    pnl = float(t.get("pnl_inr", 0))
                    pnl_pct = float(t.get("pnl_pct", 0))
                    charges = float(t.get("charges_inr", 0))
                    entry = float(t.get("entry_price", 0))
                    exit_p = float(t.get("exit_price", 0))
                    hold_m = float(t.get("hold_minutes", 0))
                    status_txt = str(t.get("status", "CLOSED"))
                    lesson = str(t.get("mistake_diagnosis", "Trade settled cleanly.")).strip()

                    pnl_fg = "#34d399" if pnl >= 0 else "#fb7185"
                    badge_bg = "rgba(16, 185, 129, 0.16)" if pnl >= 0 else "rgba(244, 63, 94, 0.16)"

                    st.markdown(
                        f"""
                        <div class="trade-card">
                            <div class="tc-top">
                                <div>
                                    <span class="tc-sym">{sym}</span>
                                    <span class="side-pill" style="background:{badge_bg}; color:{pnl_fg}; margin-left:6px;">{status_txt}</span>
                                    <div class="tc-sector">{_extract_trade_date_str(t) or "—"} • {sec} • {action}</div>
                                </div>
                                <div class="tc-pnl-wrap">
                                    <div class="tc-pnl" style="color:{pnl_fg};">₹{pnl:+,.0f} ({pnl_pct:+.2f}%)</div>
                                    <div class="tc-pnl-sub">Held {hold_m:.0f}m • Fees ₹{charges:.2f}</div>
                                </div>
                            </div>
                            <div class="tc-metrics">
                                <div>
                                    <div class="tc-m-label">Entry</div>
                                    <div class="tc-m-val">₹{entry:,.2f}</div>
                                </div>
                                <div>
                                    <div class="tc-m-label">Exit</div>
                                    <div class="tc-m-val">₹{exit_p:,.2f}</div>
                                </div>
                                <div>
                                    <div class="tc-m-label">Taxes & Slippage</div>
                                    <div class="tc-m-val">₹{charges:.2f}</div>
                                </div>
                            </div>
                            <div class="tc-reason">🧠 {lesson}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                c_rows = [
                    {
                        "Date": _extract_trade_date_str(t) or "—",
                        "Stock": str(t.get("symbol", "")).replace(".NS", ""),
                        "Side": t.get("action", "BUY"),
                        "Entry (₹)": round(float(t.get("entry_price", 0)), 2),
                        "Exit (₹)": round(float(t.get("exit_price", 0)), 2),
                        "Net P&L (₹)": round(float(t.get("pnl_inr", 0)), 2),
                        "Status": t.get("status", "CLOSED"),
                    }
                    for t in reversed(f_closed[-30:])
                ]
                st.dataframe(pd.DataFrame(c_rows), hide_index=True)

    with t_insights:
        st.markdown("**Portfolio Growth Curve**")
        eq_log = engine.portfolio.get("daily_equity_log", [])
        if eq_log:
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
                    x=alt.X("date:N", title="Trading Day"),
                    y=alt.Y("equity:Q", title="Portfolio Value (₹)", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("date:N", title="Date"),
                        alt.Tooltip("equity:Q", title="Equity (₹)", format=",.0f"),
                    ],
                )
                .properties(height=210)
            )
            st.altair_chart(eq_chart)
        else:
            st.info("Daily equity curve updates automatically as trading sessions complete.")

        st.markdown("**Trade-by-Trade Profit & Loss**")
        chart_source_trades = f_closed if selected_dates else closed_trades
        if chart_source_trades:
            pnl_df = pd.DataFrame([
                {
                    "Trade": f"{str(t.get('symbol', '')).replace('.NS', '')} ({_extract_trade_date_str(t)[-5:]})",
                    "Net P&L (₹)": round(float(t.get("pnl_inr", 0.0)), 2),
                    "Result": "Win" if float(t.get("pnl_inr", 0.0)) >= 0 else "Loss",
                }
                for t in chart_source_trades[-15:]
            ])
            bar_chart = (
                alt.Chart(pnl_df)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X("Trade:N", sort=None, title="Stock"),
                    y=alt.Y("Net P&L (₹):Q", title="Net P&L (₹)"),
                    color=alt.Color(
                        "Result:N",
                        scale=alt.Scale(domain=["Win", "Loss"], range=["#10b981", "#f43f5e"]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Trade:N", title="Stock"),
                        alt.Tooltip("Net P&L (₹):Q", title="Net P&L (₹)", format="+,.2f"),
                    ],
                )
                .properties(height=210)
            )
            st.altair_chart(bar_chart)
        else:
            st.caption("Profit/loss bars appear once settled trades match the selected view.")

        st.markdown("**Market & Capital Breakdown**")
        st.caption(
            f"Starting Capital: **₹{initial_cap:,.0f}** • Manual Deposits: **₹{manual_dep:,.0f}** • "
            f"FII Cash Net: **₹{fii_net:+,.0f} Cr** • Global Cues: **{macro_summary}**"
        )

    with t_ai:
        st.markdown(f"**AI Brain Gen #{engine.weights.get('generation', 1)}** • `{ml_summary}`")
        st.caption(f"Learned from **{shadow_learned}** simulated/skipped setups without risking real cash.")

        st.markdown("**Recent AI Trade Lessons**")
        lessons = engine.weights.get("trade_lessons_log", [])
        if not lessons:
            st.info("Lessons appear automatically as positions close.")
        else:
            for item in lessons[:12]:
                icon = "✅" if float(item.get("pnl_inr", 0)) >= 0 else "🔍"
                st.markdown(f"{icon} **{item.get('timestamp', '')}** — {item.get('lesson', '')}")

        st.divider()
        st.markdown("**Strategy Success Rates**")
        arms = engine.weights.get("bandit_arms", {})
        if arms:
            arm_rows = []
            for arm_name, stats in arms.items():
                a = float(stats.get("alpha", 2.0))
                b = float(stats.get("beta", 2.0))
                arm_rows.append({
                    "Strategy": arm_name if show_quant_jargon else humanize_arm(arm_name),
                    "Win Chance": f"{(a / (a + b)) * 100:.1f}%",
                    "Trades": int(stats.get("trades", 0)),
                    "Net P&L (₹)": f"₹{float(stats.get('net_pnl_inr', 0.0)):+,.0f}",
                })
            st.dataframe(pd.DataFrame(arm_rows), hide_index=True)

        st.markdown("**Stock Style Performance**")
        arch_mem = engine.weights.get("archetype_memory", {})
        if arch_mem:
            arch_rows = []
            for arch_name, st_a in arch_mem.items():
                ww = float(st_a.get("weighted_wins", 1.0))
                wl = float(st_a.get("weighted_losses", 1.0))
                arch_rows.append({
                    "Stock Style": arch_name if show_quant_jargon else humanize_archetype(arch_name),
                    "Win Rate": f"{(ww / max(ww + wl, 1.0)) * 100:.1f}%",
                    "Trades": int(st_a.get("trades", 0)),
                    "SL Cushion": f"{float(st_a.get('sl_mult_adj', 1.0)):.2f}x",
                })
            st.dataframe(pd.DataFrame(arch_rows), hide_index=True)

        shadows = getattr(engine, "shadow_ledger", [])
        if shadows:
            with st.expander(f"👻 View Recent Skipped / Shadow Setups ({len(shadows)})", expanded=False):
                s_rows = [
                    {
                        "Date": s.get("date", "—"),
                        "Stock": str(s.get("symbol", "")).replace(".NS", ""),
                        "Side": s.get("action", "BUY"),
                        "Why Skipped": s.get("skip_reason", "—"),
                        "Sim P&L": f"{float(s.get('shadow_pnl_pct', 0)):+.2f}%",
                    }
                    for s in reversed(shadows[-15:])
                ]
                st.dataframe(pd.DataFrame(s_rows), hide_index=True)

    with t_watchlist:
        st.markdown("**Pre-Market & Intraday Watchlist**")
        if not watchlist_data:
            st.info("Watchlist updates automatically during morning prep and market scans.")
        else:
            wl_df = pd.DataFrame(watchlist_data)
            if search_query and "symbol" in wl_df.columns:
                wl_df = wl_df[wl_df["symbol"].astype(str).str.upper().str.contains(search_query, na=False)]
            st.dataframe(wl_df, hide_index=True)

    # ---------------------------------------------------------
    # FOOTER
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <div class="app-footer">
            <strong>PulseAI Mobile & Web Portfolio Studio</strong> • Synced with <code>{_get_repo_slug()}</code><br/>
            NSE Equity Window: Mon–Fri, 09:15–15:25 IST (Auto square-off at 15:15 IST; closed on NSE holidays including 24 Nov 2026).<br/>
            <em>Educational & research simulation dashboard only — not SEBI-registered investment advice.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_live_app()