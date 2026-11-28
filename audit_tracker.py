"""
audit_tracker.py
Research logger, post-push executed-only Telegram guard, GitHub commit/sync checker,
and live NSE price auditor for PulseAI.
"""

from __future__ import annotations

import html
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
BASE_DIR = Path(__file__).resolve().parent
RESEARCH_LOG_PATH = BASE_DIR / "ai_research_log.json"
TRADE_HISTORY_PATH = BASE_DIR / "trade_history.json"
SENT_ALERTS_PATH = BASE_DIR / "sent_alerts_log.json"
PENDING_ALERTS_PATH = BASE_DIR / "pending_telegram_alerts.json"


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _read_json_safe(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def clean_symbol(raw_symbol: Any) -> str:
    return str(raw_symbol or "").replace(".NS", "").strip().upper()


def trade_signature(trade: dict) -> tuple:
    t_id = str(trade.get("trade_id") or "").strip()
    if t_id:
        return ("ID", t_id)
    return (
        "KEY",
        str(trade.get("date") or "")[:10],
        clean_symbol(trade.get("symbol")),
        str(trade.get("action") or "").upper(),
        round(float(trade.get("entry_price") or 0.0), 2),
    )


def _df_or_list_to_records(obj: Any) -> list[dict]:
    if obj is None:
        return []
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict(orient="records")
        except Exception:
            return []
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    return []


def log_research_cycle(
    candidates: Any = None,
    rec_candidates: Any = None,
    executed_symbols: set[str] | list[str] | None = None,
    skip_reasons: dict[str, str] | None = None,
    cycle_mode: str = "autonomous_cycle",
    market_note: str = "",
    max_entries: int = 400,
) -> list[dict]:
    """
    Record every researched stock (executed or skipped) plus cycle heartbeats
    into ai_research_log.json so the webapp displays complete AI activity.
    """
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    ts_str = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")

    exec_set = {clean_symbol(s) for s in (executed_symbols or [])}
    reasons_map = {clean_symbol(k): str(v) for k, v in (skip_reasons or {}).items()}

    raw_rows = _df_or_list_to_records(candidates)
    rec_rows = _df_or_list_to_records(rec_candidates)

    merged_by_sym: dict[str, dict] = {}
    for row in raw_rows:
        sym = clean_symbol(row.get("symbol") or row.get("Symbol") or row.get("ticker"))
        if sym:
            merged_by_sym[sym] = dict(row)
    for row in rec_rows:
        sym = clean_symbol(row.get("symbol") or row.get("Symbol") or row.get("ticker"))
        if sym:
            base = merged_by_sym.get(sym, {})
            base.update(row)
            merged_by_sym[sym] = base

    existing = _read_json_safe(RESEARCH_LOG_PATH, default=[])
    if not isinstance(existing, list):
        existing = []

    new_entries: list[dict] = []
    if not merged_by_sym:
        new_entries.append({
            "timestamp_ist": ts_str,
            "date": today_str,
            "mode": cycle_mode,
            "symbol": "HEARTBEAT",
            "signal": "CYCLE_OK",
            "ltp": 0.0,
            "conviction": 0.0,
            "sentiment": "NEUTRAL",
            "headline": market_note or "Automated cycle ran cleanly with 0 new setups.",
            "decision": "HEARTBEAT",
            "reason": market_note or "No qualifying trade candidates in this scan.",
        })
    else:
        for sym, row in merged_by_sym.items():
            is_exec = sym in exec_set
            ltp = float(
                row.get("entry_price")
                or row.get("ltp")
                or row.get("LTP")
                or row.get("price")
                or row.get("close")
                or 0.0
            )
            conviction = float(
                row.get("conviction")
                or row.get("conviction_score")
                or row.get("AI Confidence")
                or row.get("score")
                or 0.0
            )
            signal = str(
                row.get("action")
                or row.get("Action")
                or row.get("signal")
                or "WATCH"
            ).upper()
            sentiment = str(
                row.get("raw_sentiment")
                or row.get("sentiment")
                or "NEUTRAL"
            ).upper()
            headline = str(
                row.get("Why Is It Moving? (Latest News)")
                or row.get("headline")
                or row.get("news")
                or ""
            ).strip()[:200]

            if is_exec:
                reason_txt = "Executed and committed to trade_history.json"
            elif sym in reasons_map:
                reason_txt = reasons_map[sym]
            elif cycle_mode in ("morning_prep", "premarket_research", "premarket_update"):
                reason_txt = "Pre-market research only (entries open at 09:15 IST)"
            else:
                reason_txt = "Researched only — did not clear final entry/slot/sector gates"

            new_entries.append({
                "timestamp_ist": ts_str,
                "date": today_str,
                "mode": cycle_mode,
                "symbol": sym,
                "signal": signal,
                "ltp": round(ltp, 2),
                "conviction": round(conviction, 2),
                "sentiment": sentiment,
                "headline": headline,
                "decision": "EXECUTED" if is_exec else "RESEARCH_ONLY",
                "reason": reason_txt,
            })

    combined = (existing + new_entries)[-max_entries:]
    _atomic_write_json(RESEARCH_LOG_PATH, combined)
    return new_entries


def _trade_exists_in_ledger(
    trade: dict,
    expected_closed: bool = False,
    ledger_path: Path = TRADE_HISTORY_PATH,
) -> bool:
    """Confirm the trade is persisted on disk inside trade_history.json."""
    ledger = _read_json_safe(ledger_path, default=[])
    if not isinstance(ledger, list) or not ledger:
        return False

    target_sig = trade_signature(trade)
    for row in reversed(ledger):
        if trade_signature(row) == target_sig:
            row_status = str(row.get("status") or "").upper()
            if expected_closed:
                return row_status != "OPEN"
            return True
    return False


def _build_alert_payload(trade: dict, event_type: str) -> dict:
    ev = event_type.upper()
    sym = clean_symbol(trade.get("symbol"))
    t_date = str(trade.get("date") or datetime.now(IST).strftime("%Y-%m-%d"))[:10]
    entry_px = float(trade.get("entry_price") or 0.0)
    exit_px = float(trade.get("exit_price") or 0.0)
    t_id = str(trade.get("trade_id") or f"{t_date}_{sym}_{trade.get('action')}_{entry_px:.2f}")
    dedupe_key = f"{ev}::{t_id}"

    action = html.escape(str(trade.get("action", "TRADE")).upper())
    qty = int(trade.get("qty") or 0)
    target_px = float(trade.get("target_1") or trade.get("initial_target") or 0.0)
    sl_px = float(trade.get("stop_loss") or 0.0)
    pnl = float(trade.get("pnl_inr") or 0.0)
    pnl_pct = float(trade.get("pnl_pct") or 0.0)
    charges = float(trade.get("charges_inr") or 0.0)
    status = html.escape(str(trade.get("status", "OPEN")).upper())
    safe_sym = html.escape(sym)
    safe_id = html.escape(t_id)

    if ev == "ENTRY":
        msg = (
            f"✅ <b>EXECUTED TRADE ENTRY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>Stock:</b> <code>{safe_sym}</code> ({action})\n"
            f"📦 <b>Qty:</b> {qty} | <b>Entry:</b> ₹{entry_px:,.2f}\n"
            f"🎯 <b>Target:</b> ₹{target_px:,.2f} | 🛑 <b>SL:</b> ₹{sl_px:,.2f}\n"
            f"🧾 <b>Ledger Ref:</b> <code>{safe_id}</code>"
        )
    else:
        icon = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{icon} <b>EXECUTED TRADE EXIT ({status})</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>Stock:</b> <code>{safe_sym}</code> ({action})\n"
            f"💰 <b>Entry:</b> ₹{entry_px:,.2f} → <b>Exit:</b> ₹{exit_px:,.2f}\n"
            f"📊 <b>Net P&amp;L:</b> <b>₹{pnl:+,.2f}</b> ({pnl_pct:+.2f}%) | Fees ₹{charges:.2f}\n"
            f"🧾 <b>Ledger Ref:</b> <code>{safe_id}</code>"
        )

    return {
        "dedupe_key": dedupe_key,
        "event": ev,
        "date": t_date,
        "symbol": sym,
        "action": str(trade.get("action", "")).upper(),
        "price": round(entry_px if ev == "ENTRY" else exit_px, 2),
        "status": str(trade.get("status", "")).upper(),
        "trade_snapshot": {
            "trade_id": trade.get("trade_id"),
            "date": t_date,
            "symbol": sym,
            "action": trade.get("action"),
            "entry_price": entry_px,
            "status": trade.get("status"),
        },
        "message": msg,
    }


def _dispatch_message(notifier: Any, msg: str) -> bool:
    try:
        if hasattr(notifier, "send_message"):
            try:
                return bool(notifier.send_message(msg, force=True))
            except TypeError:
                return bool(notifier.send_message(msg))
        if hasattr(notifier, "send_alert"):
            notifier.send_alert(msg)
            return True
    except Exception as exc:
        print(f"[AUDIT GUARD] Telegram dispatch failed: {exc}")
    return False


def send_verified_trade_alert(
    notifier: Any,
    trade: dict,
    event_type: str = "ENTRY",
) -> bool:
    """
    Executed-Only Telegram Gate:
    If DEFER_TELEGRAM_UNTIL_PUSH=1 (GitHub Actions), queues the verified alert into
    pending_telegram_alerts.json so it is ONLY sent after `git push origin main` succeeds.
    Otherwise sends immediately after verifying trade_history.json on disk.
    """
    if not isinstance(trade, dict):
        return False

    ev = event_type.upper()
    if not _trade_exists_in_ledger(trade, expected_closed=(ev == "EXIT")):
        print(f"[AUDIT GUARD] Blocked {ev} Telegram alert for {trade.get('symbol')}: not verified in trade_history.json")
        return False

    payload = _build_alert_payload(trade, ev)
    dedupe_key = payload["dedupe_key"]

    sent_log = _read_json_safe(SENT_ALERTS_PATH, default=[])
    if not isinstance(sent_log, list):
        sent_log = []
    sent_keys = {
        (item.get("dedupe_key") if isinstance(item, dict) else str(item))
        for item in sent_log
    }
    if dedupe_key in sent_keys:
        return False

    defer = os.getenv("DEFER_TELEGRAM_UNTIL_PUSH", "0").strip() == "1"
    if defer:
        pending = _read_json_safe(PENDING_ALERTS_PATH, default=[])
        if not isinstance(pending, list):
            pending = []
        pending_keys = {str(p.get("dedupe_key")) for p in pending if isinstance(p, dict)}
        if dedupe_key not in pending_keys:
            pending.append(payload)
            _atomic_write_json(PENDING_ALERTS_PATH, pending)
            print(f"[AUDIT QUEUE] Staged {ev} alert for {payload['symbol']} (will send strictly after git push).")
        return True

    if notifier is None:
        return False

    if _dispatch_message(notifier, payload["message"]):
        sent_log.append({
            "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
            "date": payload["date"],
            "event": payload["event"],
            "symbol": payload["symbol"],
            "action": payload["action"],
            "price": payload["price"],
            "status": payload["status"],
            "dedupe_key": dedupe_key,
        })
        _atomic_write_json(SENT_ALERTS_PATH, sent_log[-500:])
        return True
    return False


def flush_pending_alerts() -> int:
    """
    Called ONLY after `git push origin main` succeeds in GitHub Actions.
    Sends all queued verified trade alerts and records them in sent_alerts_log.json.
    """
    pending = _read_json_safe(PENDING_ALERTS_PATH, default=[])
    if not isinstance(pending, list) or not pending:
        print("[AUDIT FLUSH] No pending Telegram alerts to send.")
        return 0

    from telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier()
    sent_log = _read_json_safe(SENT_ALERTS_PATH, default=[])
    if not isinstance(sent_log, list):
        sent_log = []
    sent_keys = {
        (item.get("dedupe_key") if isinstance(item, dict) else str(item))
        for item in sent_log
    }

    sent_count = 0
    for item in pending:
        if not isinstance(item, dict):
            continue
        dedupe_key = str(item.get("dedupe_key") or "")
        if not dedupe_key or dedupe_key in sent_keys:
            continue

        snap = item.get("trade_snapshot") or {}
        ev = str(item.get("event") or "ENTRY").upper()
        if not _trade_exists_in_ledger(snap, expected_closed=(ev == "EXIT")):
            print(f"[AUDIT FLUSH] Skipped {dedupe_key}: missing from committed trade_history.json.")
            continue

        if _dispatch_message(notifier, str(item.get("message") or "")):
            sent_log.append({
                "timestamp_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
                "date": item.get("date", ""),
                "event": ev,
                "symbol": item.get("symbol", ""),
                "action": item.get("action", ""),
                "price": item.get("price", 0.0),
                "status": item.get("status", ""),
                "dedupe_key": dedupe_key,
            })
            sent_keys.add(dedupe_key)
            sent_count += 1

    if sent_count > 0:
        _atomic_write_json(SENT_ALERTS_PATH, sent_log[-500:])

    try:
        PENDING_ALERTS_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"[AUDIT FLUSH] Dispatched {sent_count} verified post-push Telegram alert(s).")
    return sent_count


def check_remote_ledger_sync(repo: str, local_count: int) -> dict:
    """Compare dashboard trade count with live GitHub main trade_history.json."""
    url = f"https://raw.githubusercontent.com/{repo}/main/trade_history.json?t={int(datetime.now().timestamp())}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PulseAI-Auditor"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            remote_data = json.loads(resp.read().decode("utf-8"))
        remote_count = len(remote_data) if isinstance(remote_data, list) else 0
        return {
            "ok": True,
            "remote_count": remote_count,
            "local_count": local_count,
            "in_sync": remote_count == local_count,
        }
    except Exception as exc:
        return {
            "ok": False,
            "remote_count": None,
            "local_count": local_count,
            "in_sync": False,
            "error": str(exc),
        }


def fetch_recent_github_commits(repo: str, limit: int = 8) -> list[dict]:
    """Fetch recent GitHub commits on main to verify automatic CI/CD runs."""
    api_url = f"https://api.github.com/repos/{repo}/commits?sha=main&per_page={limit}"
    try:
        req = urllib.request.Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "PulseAI-Auditor",
            },
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            commits = json.loads(resp.read().decode("utf-8"))
        rows = []
        for c in commits if isinstance(commits, list) else []:
            sha = str(c.get("sha", ""))[:7]
            commit_obj = c.get("commit", {})
            author = commit_obj.get("author", {})
            raw_iso = str(author.get("date", ""))
            ist_str = raw_iso
            if raw_iso:
                try:
                    dt_utc = datetime.fromisoformat(raw_iso.replace("Z", "+00:00"))
                    ist_str = dt_utc.astimezone(IST).strftime("%Y-%m-%d %I:%M %p IST")
                except Exception:
                    pass
            rows.append({
                "Commit": sha,
                "Time (IST)": ist_str,
                "Author": author.get("name", "Unknown"),
                "Message": str(commit_obj.get("message", "")).splitlines()[0][:110],
            })
        return rows
    except Exception:
        return []


def verify_trades_against_nse(
    trades: list[dict],
    target_date_str: str,
    slippage_tolerance_pct: float = 0.006,
) -> list[dict]:
    """
    Verify entry_price and exit_price for all trades on target_date_str
    against real NSE High-Low market data via yfinance.
    """
    day_trades = [t for t in trades if str(t.get("date", ""))[:10] == target_date_str]
    if not day_trades:
        return []

    target_d = date.fromisoformat(target_date_str)
    next_d = target_d + timedelta(days=1)
    days_ago = (datetime.now(IST).date() - target_d).days
    interval = "5m" if days_ago <= 58 else "1d"

    results: list[dict] = []
    bars_cache: dict[str, tuple[float, float, float] | None] = {}

    for t in day_trades:
        sym = clean_symbol(t.get("symbol"))
        entry_px = float(t.get("entry_price") or 0.0)
        exit_px = float(t.get("exit_price") or 0.0)
        status = str(t.get("status", "OPEN")).upper()

        if sym not in bars_cache:
            try:
                hist = yf.Ticker(f"{sym}.NS").history(
                    start=target_d.isoformat(),
                    end=next_d.isoformat(),
                    interval=interval,
                    auto_adjust=False,
                )
                if hist is not None and not hist.empty:
                    idx = hist.index
                    if getattr(idx, "tz", None) is not None:
                        hist = hist.copy()
                        hist.index = idx.tz_convert(IST)
                    day_slice = hist[hist.index.strftime("%Y-%m-%d") == target_date_str]
                    if day_slice.empty:
                        day_slice = hist
                    bars_cache[sym] = (
                        round(float(day_slice["Low"].min()), 2),
                        round(float(day_slice["High"].max()), 2),
                        round(float(day_slice["Close"].iloc[-1]), 2),
                    )
                else:
                    bars_cache[sym] = None
            except Exception:
                bars_cache[sym] = None

        mkt_range = bars_cache[sym]
        if mkt_range is None:
            results.append({
                "Symbol": sym,
                "Side": str(t.get("action", "")).upper(),
                "Status": status,
                "Entry (₹)": round(entry_px, 2),
                "Exit (₹)": round(exit_px, 2) if exit_px else None,
                "NSE Low – High": "Unavailable",
                "NSE Close": None,
                "Entry Check": "NO DATA",
                "Exit Check": "NO DATA",
                "Bars": interval,
            })
            continue

        low_px, high_px, close_px = mkt_range
        lo_bound = low_px * (1.0 - slippage_tolerance_pct)
        hi_bound = high_px * (1.0 + slippage_tolerance_pct)

        entry_ok = lo_bound <= entry_px <= hi_bound
        if exit_px == 0.0 and status == "OPEN":
            exit_label = "OPEN"
        else:
            exit_label = "PASS" if (lo_bound <= exit_px <= hi_bound) else "OUT OF RANGE"

        results.append({
            "Symbol": sym,
            "Side": str(t.get("action", "")).upper(),
            "Status": status,
            "Entry (₹)": round(entry_px, 2),
            "Exit (₹)": round(exit_px, 2) if exit_px else None,
            "NSE Low – High": f"₹{low_px:,.2f} – ₹{high_px:,.2f}",
            "NSE Close": round(close_px, 2),
            "Entry Check": "PASS" if entry_ok else "OUT OF RANGE",
            "Exit Check": exit_label,
            "Bars": interval,
        })

    return results


if __name__ == "__main__":
    if "--flush-alerts" in sys.argv:
        flush_pending_alerts()