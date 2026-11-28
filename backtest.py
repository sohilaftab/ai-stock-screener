"""
PulseAI baseline back-tester.

A deliberately SIMPLE benchmark - opening-range breakout (ORB) with a VWAP trend filter on 5-minute
NSE bars - run with the SAME execution rules and cost model as the live engine:
  * entry at the OPEN of the bar after the signal (no look-ahead), with slippage,
  * stop/target resolved bar-by-bar, stop first if both are inside one bar, gap-through fills at the open,
  * real NSE intraday charges (STT, stamp, brokerage, exchange, SEBI, GST),
  * no entries before 09:30 or after 14:30, forced square-off at 15:15.

Why it exists: the full PulseAI engine (news + regime + bandits + ML) is only worth its complexity if it
beats a dumb baseline out-of-sample.  Run this first to learn what "dumb" earns after costs.

Usage
-----
  python backtest.py --synthetic                         # offline smoke test (random walk => no edge expected)
  python backtest.py --days 55                           # Yahoo 5m data (Yahoo keeps ~60 days)
  python backtest.py --symbols RELIANCE,TCS,HDFCBANK     # custom universe
  python backtest.py --csv-dir data/5m                   # SYMBOL.csv with datetime,Open,High,Low,Close,Volume

Outputs go to backtest_output/ (trades.csv, summary.json).
"""
import argparse
import json
import math
import os
from datetime import time as dtime

import numpy as np
import pandas as pd

from exec_sim import apply_slippage, resolve_barrier_outcome

ORB_END = dtime(9, 30)
LATEST_ENTRY = dtime(14, 30)
SQUARE_OFF = dtime(15, 15)


# ------------------------------------------------------------------ costs (kept identical to the live engine)
def nse_intraday_charges(side, qty, entry_price, exit_price):
    qty = max(1, int(qty))
    buy_val = (entry_price * qty) if side == "BUY" else (exit_price * qty)
    sell_val = (exit_price * qty) if side == "BUY" else (entry_price * qty)
    turnover = buy_val + sell_val
    stt = round(0.00025 * sell_val)
    stamp = round(0.00003 * buy_val)
    brokerage = min(20.0, 0.0003 * buy_val) + min(20.0, 0.0003 * sell_val)
    nse_txn = 0.0000307 * turnover
    sebi_fee = 0.000001 * turnover
    gst = 0.18 * (brokerage + nse_txn + sebi_fee)
    return round(stt + stamp + brokerage + nse_txn + sebi_fee + gst, 2)


# ------------------------------------------------------------------ data
def load_yf(symbols, days):
    import yfinance as yf

    period = f"{min(int(days), 59)}d"
    raw = yf.download([f"{s}.NS" for s in symbols], period=period, interval="5m",
                      group_by="ticker", threads=True, progress=False)
    out = {}
    if raw is None or raw.empty:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    for s in symbols:
        try:
            df = raw[f"{s}.NS"] if multi else raw
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if df.empty:
                continue
            idx = df.index.tz_localize("Asia/Kolkata") if df.index.tz is None else df.index.tz_convert("Asia/Kolkata")
            out[s] = df.set_axis(idx)
        except Exception:
            continue
    return out


def load_csv(folder):
    out = {}
    for fn in os.listdir(folder):
        if fn.lower().endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, fn), parse_dates=["datetime"], index_col="datetime")
            df.index = df.index.tz_localize("Asia/Kolkata") if df.index.tz is None else df.index.tz_convert("Asia/Kolkata")
            out[os.path.splitext(fn)[0].upper()] = df
    return out


def synthetic_data(symbols, days=60, seed=7):
    """Driftless random walk with realistic bar structure. A correct back-tester must find NO edge here."""
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(end=pd.Timestamp("2026-11-20"), periods=days)
    out = {}
    for s in symbols:
        price, rows = float(rng.uniform(300, 3000)), []
        for d in bdays:
            ts = pd.date_range(f"{d.date()} 09:15", f"{d.date()} 15:25", freq="5min", tz="Asia/Kolkata")
            for t in ts:
                o = price
                c = o * math.exp(rng.normal(0, 0.0012))
                h = max(o, c) * (1 + abs(rng.normal(0, 0.0005)))
                l = min(o, c) * (1 - abs(rng.normal(0, 0.0005)))
                rows.append((t, o, h, l, c, float(rng.lognormal(10, 0.5))))
                price = c
        out[s] = pd.DataFrame(rows, columns=["datetime", "Open", "High", "Low", "Close", "Volume"]).set_index("datetime")
    return out


# ------------------------------------------------------------------ strategy
def find_signal_and_trade(day_df, p):
    """ORB + VWAP breakout for one symbol-day. Returns a trade dict or None."""
    bars = day_df.sort_index()
    if len(bars) < 20:
        return None
    t_of = bars.index.time
    orb = bars[[t < ORB_END for t in t_of]]
    if len(orb) < 2:
        return None
    orb_high, orb_low = float(orb["High"].max()), float(orb["Low"].min())

    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    cum_v = bars["Volume"].cumsum().replace(0, np.nan)
    vwap = ((tp * bars["Volume"]).cumsum() / cum_v).ffill()

    n_orb = len(orb)
    for i in range(n_orb, len(bars) - 2):
        ts = bars.index[i]
        if ts.time() >= LATEST_ENTRY:
            break
        close = float(bars["Close"].iloc[i])
        v = float(vwap.iloc[i]) if not math.isnan(vwap.iloc[i]) else close
        side = None
        if close > orb_high and close > v:
            side = "BUY"
        elif close < orb_low and close < v:
            side = "SELL"
        if side is None:
            continue

        is_buy = side == "BUY"
        fill_bar = bars.iloc[i + 1]
        entry = apply_slippage(float(fill_bar["Open"]), is_buy, p["slippage_pct"])
        raw_risk = (entry - orb_low) if is_buy else (orb_high - entry)
        risk = min(max(raw_risk, entry * p["min_risk_pct"] / 100.0), entry * p["max_risk_pct"] / 100.0)
        if risk <= 0:
            continue
        sl = entry - risk if is_buy else entry + risk
        t1 = entry + p["rr"] * risk if is_buy else entry - p["rr"] * risk

        qty = int(min((p["capital"] * p["risk_pct"] / 100.0) // risk, (p["capital"] * p["max_position_pct"] / 100.0) // entry))
        if qty <= 0:
            continue

        # bars strictly after the entry bar, up to square-off
        path = bars.iloc[i + 2:]
        path = path[[t < SQUARE_OFF for t in path.index.time]]
        if path.empty:
            continue
        path_dicts = [{"open": float(r.Open), "high": float(r.High), "low": float(r.Low), "close": float(r.Close)}
                      for r in path.itertuples()]
        out = resolve_barrier_outcome(is_buy, path_dicts, sl, t1)
        reason = out["reason"] if out["reason"] != "NONE" else "TIME"
        raw_exit = out["exit_px"] if out["reason"] != "NONE" else float(path["Close"].iloc[-1])
        exit_px = apply_slippage(raw_exit, not is_buy, p["slippage_pct"])

        gross = (exit_px - entry) * qty if is_buy else (entry - exit_px) * qty
        charges = nse_intraday_charges(side, qty, entry, exit_px)
        net = gross - charges
        return {
            "date": str(bars.index[0].date()), "entry_time": str(bars.index[i + 1].time()), "side": side,
            "entry": round(entry, 2), "exit": round(exit_px, 2), "qty": qty, "reason": reason,
            "gross_inr": round(gross, 2), "charges_inr": charges, "net_inr": round(net, 2),
            "r_multiple": round(net / (risk * qty), 3), "risk_inr": round(risk * qty, 2),
        }
    return None


def run_backtest(data, params):
    trades = []
    for sym, df in data.items():
        for _d, day_df in df.groupby(df.index.date):
            tr = find_signal_and_trade(day_df, params)
            if tr:
                tr["symbol"] = sym
                trades.append(tr)
    if not trades:
        return pd.DataFrame()
    tdf = pd.DataFrame(trades).sort_values(["date", "entry_time"]).reset_index(drop=True)
    # portfolio rule: at most N entries per day, earliest signals first (no look-ahead ranking)
    tdf = tdf.groupby("date", group_keys=False).head(params["max_trades_per_day"]).reset_index(drop=True)
    return tdf


# ------------------------------------------------------------------ metrics
def summarize(tdf, capital, seed=11, n_boot=2000):
    if tdf is None or tdf.empty:
        return {"trades": 0, "note": "no trades generated"}
    net = tdf["net_inr"].to_numpy(float)
    r = tdf["r_multiple"].to_numpy(float)
    wins, losses = net[net > 0], net[net <= 0]
    daily = tdf.groupby("date")["net_inr"].sum()
    equity = capital + daily.cumsum()
    peak = equity.cummax()
    max_dd = float((peak - equity).max())
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 2 and daily.std(ddof=1) > 0 else 0.0

    rng = np.random.default_rng(seed)
    boots = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    def half(df):
        return {"trades": int(len(df)), "net_inr": round(float(df["net_inr"].sum()), 2),
                "expectancy_inr": round(float(df["net_inr"].mean()), 2)} if len(df) else {"trades": 0}

    dates = sorted(tdf["date"].unique())
    cut = dates[int(len(dates) * 0.6)] if len(dates) >= 5 else dates[-1]
    return {
        "trades": int(len(tdf)),
        "win_rate_pct": round(float((net > 0).mean() * 100), 1),
        "avg_win_inr": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_inr": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "expectancy_inr": round(float(net.mean()), 2),
        "expectancy_R": round(float(r.mean()), 3),
        "expectancy_R_95ci": [round(float(lo), 3), round(float(hi), 3)],
        "prob_expectancy_positive": round(float((boots > 0).mean()), 3),
        "total_net_inr": round(float(net.sum()), 2),
        "total_charges_inr": round(float(tdf["charges_inr"].sum()), 2),
        "max_drawdown_inr": round(max_dd, 2),
        "daily_sharpe_annualised": round(sharpe, 2),
        "by_side": {k: half(v) for k, v in tdf.groupby("side")},
        "by_exit": {k: int(v) for k, v in tdf["reason"].value_counts().items()},
        "first_60pct_of_days": half(tdf[tdf["date"] < cut]),
        "last_40pct_of_days": half(tdf[tdf["date"] >= cut]),
        "reading_guide": ("If the 95% CI of expectancy_R straddles 0, the sample cannot distinguish skill from luck. "
                          "Need positive expectancy in BOTH time halves before trusting it."),
    }


def main():
    ap = argparse.ArgumentParser(description="PulseAI baseline ORB/VWAP back-tester")
    ap.add_argument("--symbols", default="RELIANCE,TCS,HDFCBANK,ICICIBANK,INFY,SBIN,BHARTIARTL,ITC,LT,AXISBANK,"
                                         "KOTAKBANK,SUNPHARMA,MARUTI,NTPC,BAJFINANCE,TITAN,TATASTEEL,HINDALCO,M&M,WIPRO")
    ap.add_argument("--days", type=int, default=55)
    ap.add_argument("--csv-dir")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--capital", type=float, default=150000.0)
    ap.add_argument("--risk-pct", type=float, default=0.5)
    ap.add_argument("--rr", type=float, default=1.5)
    ap.add_argument("--slippage-pct", type=float, default=0.05)
    ap.add_argument("--max-trades-per-day", type=int, default=3)
    ap.add_argument("--out", default="backtest_output")
    a = ap.parse_args()

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    data = synthetic_data(symbols, a.days) if a.synthetic else (load_csv(a.csv_dir) if a.csv_dir else load_yf(symbols, a.days))
    if not data:
        print("No data loaded (Yahoo unreachable or empty). Try --csv-dir or --synthetic.")
        return

    params = {"capital": a.capital, "risk_pct": a.risk_pct, "rr": a.rr, "slippage_pct": a.slippage_pct,
              "max_trades_per_day": a.max_trades_per_day, "min_risk_pct": 0.4, "max_risk_pct": 1.5,
              "max_position_pct": 25.0}
    tdf = run_backtest(data, params)
    summary = summarize(tdf, a.capital)
    os.makedirs(a.out, exist_ok=True)
    if not tdf.empty:
        tdf.to_csv(os.path.join(a.out, "trades.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"params": params, "summary": summary, "synthetic": a.synthetic}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    