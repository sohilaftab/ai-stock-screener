import yfinance as yf
import pandas as pd
import numpy as np
from nse_client import NSEClient


class MarketDataScanner:
    def __init__(self, tickers):
        self.tickers = tickers
        self.nse = NSEClient()

    @staticmethod
    def _compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    def scan_volume_breakouts(self, vol_multiplier=1.3, min_price_change_pct=1.0, top_n=10, allow_fallback=True):
        """
        Combines 3-month historical chart indicators (EMA20/50, 14d ATR, 20d Avg Vol)
        with ZERO-DELAY live NSE tick prices for instant entry accuracy.
        """
        # 1. Fetch instant live NSE tick prices first (0.3s single call)
        live_nse_map = self.nse.get_live_nse_quotes()

        # 2. Fetch historical daily bars for 20d Average Volume, ATR, and EMAs
        data = yf.download(
            tickers=self.tickers,
            period="3mo",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )

        all_stocks = []
        for ticker in self.tickers:
            try:
                df = data[ticker].dropna(how="all")
                if len(df) < 25:
                    continue

                clean_symbol = ticker.replace(".NS", "")
                hist_df = df.iloc[:-1]
                live_row = df.iloc[-1]

                avg_daily_vol = float(hist_df["Volume"].tail(20).mean())
                if avg_daily_vol <= 0:
                    continue

                # Overlay zero-delay NSE exchange prices if available; otherwise use latest yfinance bar
                if clean_symbol in live_nse_map:
                    nse_tick = live_nse_map[clean_symbol]
                    live_price = nse_tick["current_price"]
                    day_open = nse_tick["open"]
                    day_high = nse_tick["day_high"]
                    day_low = nse_tick["day_low"]
                    prev_close = nse_tick["prev_close"]
                    price_change_pct = nse_tick["change_pct"]
                    live_vol = float(nse_tick["live_volume"]) if nse_tick["live_volume"] > 0 else float(live_row["Volume"])
                else:
                    live_price = float(live_row["Close"])
                    day_open = float(live_row["Open"])
                    day_high = float(live_row["High"])
                    day_low = float(live_row["Low"])
                    prev_close = float(hist_df["Close"].iloc[-1])
                    live_vol = float(live_row["Volume"])
                    price_change_pct = ((live_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

                if live_price <= 50 or prev_close <= 0:
                    continue

                # Update last close in series with live tick price for accurate real-time EMA & RSI
                close_series = df["Close"].copy()
                close_series.iloc[-1] = live_price

                ema_20 = float(close_series.ewm(span=20, adjust=False).mean().iloc[-1])
                ema_50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])
                if live_price > ema_20 > ema_50:
                    chart_trend = "BULLISH_UPTREND"
                elif live_price < ema_20 < ema_50:
                    chart_trend = "BEARISH_DOWNTREND"
                else:
                    chart_trend = "RANGE_CONSOLIDATION"

                high_low = df["High"] - df["Low"]
                high_close = (df["High"] - df["Close"].shift()).abs()
                low_close = (df["Low"] - df["Close"].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr_14 = float(tr.tail(14).mean())
                if np.isnan(atr_14) or atr_14 <= 0:
                    atr_14 = live_price * 0.015

                rsi_14 = self._compute_rsi(close_series, period=14)
                typical_price = (day_high + day_low + live_price) / 3.0
                vol_ratio = live_vol / avg_daily_vol
                trend_prices = [round(float(x), 2) for x in close_series.tail(15).tolist()]

                all_stocks.append({
                    "symbol": clean_symbol,
                    "ltp": round(live_price, 2),
                    "open": round(day_open, 2),
                    "high": round(day_high, 2),
                    "low": round(day_low, 2),
                    "prev_close": round(prev_close, 2),
                    "change_pct": round(price_change_pct, 2),
                    "abs_change": abs(round(price_change_pct, 2)),
                    "live_volume": int(live_vol),
                    "avg_volume": int(avg_daily_vol),
                    "vol_ratio": round(vol_ratio, 2),
                    "vol_surge_pct": int(vol_ratio * 100),
                    "atr": round(atr_14, 2),
                    "rsi": round(rsi_14, 1),
                    "ema_20": round(ema_20, 2),
                    "ema_50": round(ema_50, 2),
                    "chart_trend": chart_trend,
                    "vwap_proxy": round(typical_price, 2),
                    "trend_30d": trend_prices
                })
            except Exception:
                continue

        full_df = pd.DataFrame(all_stocks)
        if full_df.empty:
            return full_df, False

        strict_df = full_df[
            (full_df["vol_ratio"] >= vol_multiplier) &
            (full_df["abs_change"] >= min_price_change_pct)
        ].copy()

        if not strict_df.empty:
            strict_df = strict_df.sort_values(by="vol_ratio", ascending=False)
            return strict_df.head(top_n), False

        if allow_fallback:
            fallback_df = full_df.sort_values(by=["vol_ratio", "abs_change"], ascending=[False, False]).head(top_n)
            return fallback_df, True

        return pd.DataFrame(), False

    @staticmethod
    def fetch_live_intraday_snapshot(symbols):
        """
        Fetches ZERO-DELAY live tick prices from NSE India first, supplemented by
        1-minute (1m) intraday candles for exact real-time VWAP and SL/Target execution.
        """
        if not symbols:
            return {}

        nse = NSEClient()
        live_nse_quotes = nse.get_live_nse_quotes(symbols=set(symbols))

        ns_tickers = [f"{s}.NS" for s in symbols]
        data = yf.download(
            tickers=ns_tickers,
            period="1d",
            interval="1m",  # 1-minute high-resolution bars instead of 5-minute delayed bars
            group_by="ticker",
            threads=True,
            progress=False
        )

        snapshots = {}
        for sym, tk in zip(symbols, ns_tickers):
            try:
                # 1. Start with 1-minute candle VWAP if available
                df = data[tk].dropna(how="all") if len(ns_tickers) > 1 else data.dropna(how="all")
                if not df.empty:
                    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
                    cum_vol = df["Volume"].sum()
                    vwap_1m = float((tp * df["Volume"]).sum() / cum_vol) if cum_vol > 0 else float(df["Close"].iloc[-1])
                    cur_p = float(df["Close"].iloc[-1])
                    d_high = float(df["High"].max())
                    d_low = float(df["Low"].min())
                    vol_1m = int(df["Volume"].iloc[-1])
                else:
                    cur_p, d_high, d_low, vwap_1m, vol_1m = 0.0, 0.0, 0.0, 0.0, 0

                # 2. Override with direct NSE real-time tick price if available
                if sym in live_nse_quotes:
                    q = live_nse_quotes[sym]
                    cur_p = q["current_price"]
                    d_high = max(d_high, q["day_high"]) if d_high > 0 else q["day_high"]
                    d_low = min(d_low, q["day_low"]) if d_low > 0 else q["day_low"]
                    if vwap_1m <= 0:
                        vwap_1m = q["intraday_vwap"]

                if cur_p > 0:
                    snapshots[sym] = {
                        "current_price": round(cur_p, 2),
                        "day_high": round(d_high, 2),
                        "day_low": round(d_low, 2),
                        "intraday_vwap": round(vwap_1m, 2),
                        "last_1m_vol": vol_1m
                    }
            except Exception:
                continue
        return snapshots