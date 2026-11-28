"""
audit_dashboard_panel.py
All-in-one PulseAI Streamlit Audit Tab:
1. Automation Heartbeat, 15:15 IST Hygiene & GitHub Commit Proof
2. Full AI Research Feed (Executed vs. Researched/Skipped)
3. Telegram Executed-Alert Audit (1-to-1 Ledger Match)
4. Live NSE Market Price Verification (yfinance)
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from audit_tracker import (
    check_remote_ledger_sync,
    clean_symbol,
    fetch_recent_github_commits,
    verify_trades_against_nse,
)

IST = ZoneInfo("Asia/Kolkata")


def _compute_eod_hygiene(trade_history: list[dict]) -> tuple[str, str]:
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    open_trades = [t for t in trade_history if str(t.get("status", "")).upper() == "OPEN"]
    stale_open = [t for t in open_trades if str(t.get("date", ""))[:10] != today_str]

    if stale_open:
        return "ACTION NEEDED", f"{len(stale_open)} prior-day trade(s) still OPEN"

    mins = now_ist.hour * 60 + now_ist.minute
    is_weekday = now_ist.weekday() < 5
    if is_weekday and mins >= (15 * 60 + 20) and open_trades:
        return "WARNING", f"{len(open_trades)} trade(s) still OPEN past 15:20 IST"

    if open_trades:
        return "ACTIVE", f"{len(open_trades)} live position(s) within 5-slot cap"
    return "CLEAN", "0 stuck open positions"


def render_audit_tab(
    trade_history: list[dict],
    load_state_fn: Callable[[str, object], object],
    repo_slug: str = "sohilaftab/ai-stock-screener",
) -> None:
    st.markdown("### 🛡️ Automation, Research & Live Market Audit")

    research_log = load_state_fn("ai_research_log.json", [])
    shadow_log = load_state_fn("shadow_ledger.json", [])
    sent_alerts = load_state_fn("sent_alerts_log.json", [])

    if not isinstance(research_log, list):
        research_log = []
    if not isinstance(shadow_log, list):
        shadow_log = []
    if not isinstance(sent_alerts, list):
        sent_alerts = []

    latest_hb = research_log[-1].get("timestamp_ist", "Waiting for first run") if research_log else "Waiting for first run"
    hygiene_status, hygiene_detail = _compute_eod_hygiene(trade_history)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last AI Scan (IST)", latest_hb.replace(" IST", ""))
    m2.metric("Research Signals Logged", len(research_log))
    m3.metric("Verified Telegram Alerts", len(sent_alerts))
    m4.metric("15:15 Square-Off Hygiene", hygiene_status, delta=hygiene_detail, delta_color="off")

    # ---------------------------------------------------------
    # 1. AUTOMATION & GITHUB SYNC PROOF
    # ---------------------------------------------------------
    with st.expander("🔄 Verify Automatic GitHub Actions Runs & Ledger Sync", expanded=False):
        b1, b2 = st.columns(2)
        if b1.button("Check Ledger Sync Now", use_container_width=True, key="btn_audit_sync"):
            st.session_state["audit_sync_result"] = check_remote_ledger_sync(repo_slug, len(trade_history))
        if b2.button("Load Recent Automation Commits", use_container_width=True, key="btn_audit_commits"):
            st.session_state["audit_recent_commits"] = fetch_recent_github_commits(repo_slug, limit=8)

        sync_res = st.session_state.get("audit_sync_result")
        if sync_res:
            if sync_res["ok"] and sync_res["in_sync"]:
                st.success(f"In Sync: Dashboard and GitHub main both contain {sync_res['remote_count']} trades.")
            elif sync_res["ok"]:
                st.warning(
                    f"Sync Mismatch: Dashboard shows {sync_res['local_count']} trades vs GitHub main {sync_res['remote_count']}."
                )
            else:
                st.error(f"GitHub check failed: {sync_res.get('error')}")

        commits = st.session_state.get("audit_recent_commits")
        if commits is not None:
            if commits:
                st.dataframe(pd.DataFrame(commits), use_container_width=True, hide_index=True)
            else:
                st.info("Could not load commits (repo may be private or rate-limited).")

    st.divider()

    # ---------------------------------------------------------
    # 2. FULL AI RESEARCH FEED (EXECUTED VS RESEARCH-ONLY)
    # ---------------------------------------------------------
    st.markdown("#### 🔬 1. AI Research Feed (Executed vs. Researched Only)")
    st.caption(
        "Every stock researched by PulseAI appears here with its exact decision and veto/skip reason. "
        "Telegram only fires for EXECUTED trades committed to trade_history.json."
    )

    filter_choice = st.radio(
        "Filter feed:",
        ["All Activity", "Executed Trades Only", "Researched / Skipped Only", "Heartbeats Only"],
        horizontal=True,
        key="audit_feed_filter",
    )

    ledger_pairs = {
        (str(t.get("date", ""))[:10], clean_symbol(t.get("symbol")))
        for t in trade_history
    }

    feed_rows = []
    for item in reversed(research_log[-150:]):
        sym = clean_symbol(item.get("symbol"))
        dt = str(item.get("date", ""))[:10]
        is_hb = sym == "HEARTBEAT"
        in_ledger = (not is_hb) and ((dt, sym) in ledger_pairs)

        if filter_choice == "Executed Trades Only" and not in_ledger:
            continue
        if filter_choice == "Researched / Skipped Only" and (in_ledger or is_hb):
            continue
        if filter_choice == "Heartbeats Only" and not is_hb:
            continue

        if is_hb:
            badge = "HEARTBEAT"
        elif in_ledger:
            badge = "EXECUTED (In Ledger)"
        else:
            badge = "RESEARCH ONLY (Not Traded)"

        feed_rows.append({
            "Time (IST)": item.get("timestamp_ist", ""),
            "Mode": item.get("mode", ""),
            "Stock": sym,
            "Signal": item.get("signal", ""),
            "LTP (₹)": item.get("ltp", 0.0),
            "Conviction": item.get("conviction", 0.0),
            "Ledger Status": badge,
            "Reason / Veto": item.get("reason", ""),
            "News": item.get("headline", ""),
        })

    if feed_rows:
        st.dataframe(pd.DataFrame(feed_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No entries match this filter yet. Research entries will populate automatically on the next cycle.")

    # ---------------------------------------------------------
    # 3. TELEGRAM ALERT AUDIT (1-TO-1 MATCH WITH LEDGER)
    # ---------------------------------------------------------
    with st.expander(f"📨 2. Verified Telegram Alert Log ({len(sent_alerts)} sent)", expanded=False):
        if not sent_alerts:
            st.info("No verified executed-trade Telegram alerts logged yet in sent_alerts_log.json.")
        else:
            tg_rows = []
            for a in reversed(sent_alerts[-60:]):
                if not isinstance(a, dict):
                    continue
                pair = (str(a.get("date", ""))[:10], clean_symbol(a.get("symbol")))
                tg_rows.append({
                    "Time (IST)": a.get("timestamp_ist", ""),
                    "Event": a.get("event", ""),
                    "Stock": clean_symbol(a.get("symbol")),
                    "Action": a.get("action", ""),
                    "Price (₹)": a.get("price", 0.0),
                    "In Dashboard Ledger?": "YES" if pair in ledger_pairs else "MISSING",
                })
            st.dataframe(pd.DataFrame(tg_rows), use_container_width=True, hide_index=True)

    with st.expander(f"👻 3. Shadow / Vetoed Setups ({len(shadow_log)} records)", expanded=False):
        if shadow_log:
            st.dataframe(pd.DataFrame(list(reversed(shadow_log[-60:]))), use_container_width=True, hide_index=True)
        else:
            st.info("No shadow setups recorded in shadow_ledger.json.")

    st.divider()

    # ---------------------------------------------------------
    # 4. LIVE NSE MARKET PRICE VERIFICATION
    # ---------------------------------------------------------
    st.markdown("#### 📈 4. Live NSE Market Price Verification")
    st.caption(
        "Cross-checks every executed trade's Entry and Exit price against real NSE intraday High–Low bars "
        "from Yahoo Finance (allowing a 0.6% modeled slippage buffer)."
    )

    available_dates = sorted(
        {str(t.get("date", ""))[:10] for t in trade_history if len(str(t.get("date", ""))) >= 10},
        reverse=True,
    )
    if not available_dates:
        st.info("No executed trades found in trade_history.json yet.")
        return

    c_date, c_go = st.columns([2, 1])
    selected_date = c_date.selectbox("Select Trading Date", available_dates, key="audit_nse_date")
    if c_go.button("Verify Prices vs. NSE", type="primary", use_container_width=True, key="btn_verify_nse_bars"):
        with st.spinner(f"Fetching NSE market bars for {selected_date}..."):
            st.session_state[f"nse_audit_{selected_date}"] = verify_trades_against_nse(
                trade_history,
                selected_date,
            )

    verified_rows = st.session_state.get(f"nse_audit_{selected_date}")
    if verified_rows is not None:
        if not verified_rows:
            st.warning(f"No trades found for {selected_date}.")
        else:
            st.dataframe(pd.DataFrame(verified_rows), use_container_width=True, hide_index=True)
            bad_e = sum(1 for r in verified_rows if r["Entry Check"] == "OUT OF RANGE")
            bad_x = sum(1 for r in verified_rows if r["Exit Check"] == "OUT OF RANGE")
            if bad_e == 0 and bad_x == 0:
                st.success(f"All {len(verified_rows)} trade(s) on {selected_date} fall cleanly inside real NSE session ranges.")
            else:
                st.error(f"Detected {bad_e} out-of-range entry price(s) and {bad_x} out-of-range exit price(s).")