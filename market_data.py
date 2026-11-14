import yfinance as yf
import pandas as pd
import numpy as np


class MarketDataScanner:
    def __init__(self, tickers):
        self.tickers = tickers

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
        Downloads 3 months of daily OHLCV data to compute multi-timeframe chart parameters:
        EMA-20, EMA-50, 14-day ATR, RSI-14, 20-day Volume Surge, and 15-day Sparklines.
        """
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

                hist_df = df.iloc[:-1]
                live_row = df.iloc[-1]

                avg_daily_vol = float(hist_df["Volume"].tail(20).mean())
                live_vol = float(live_row["Volume"])
                prev_close = float(hist_df["Close"].iloc[-1])
                live_price = float(live_row["Close"])
                day_open = float(live_row["Open"])
                day_high = float(live_row["High"])
                day_low = float(live_row["Low"])

                if avg_daily_vol <= 0 or prev_close <= 0 or live_price <= 50:
                    continue

                # Chart Trend Parameters: EMA-20 & EMA-50
                ema_20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_50 = float(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
                if live_price > ema_20 > ema_50:
                    chart_trend = "BULLISH_UPTREND"
                elif live_price < ema_20 < ema_50:
                    chart_trend = "BEARISH_DOWNTREND"
                else:
                    chart_trend = "RANGE_CONSOLIDATION"

                # 14-Day ATR for dynamic Stop-Loss and Target sizing
                high_low = df["High"] - df["Low"]
                high_close = (df["High"] - df["Close"].shift()).abs()
                low_close = (df["Low"] - df["Close"].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr_14 = float(tr.tail(14).mean())
                if np.isnan(atr_14) or atr_14 <= 0:
                    atr_14 = live_price * 0.015

                rsi_14 = self._compute_rsi(df["Close"], period=14)
                typical_price = (day_high + day_low + live_price) / 3.0

                vol_ratio = live_vol / avg_daily_vol
                price_change_pct = ((live_price - prev_close) / prev_close) * 100
                clean_symbol = ticker.replace(".NS", "")
                trend_prices = [round(float(x), 2) for x in df["Close"].tail(15).tolist()]

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
        Fetches 5-minute intraday candles for active positions to compute real-time
        Intraday VWAP, session High/Low since entry, and latest momentum.
        """
        if not symbols:
            return {}
        ns_tickers = [f"{s}.NS" for s in symbols]
        data = yf.download(
            tickers=ns_tickers,
            period="1d",
            interval="5m",
            group_by="ticker",
            threads=True,
            progress=False
        )
        snapshots = {}
        for sym, tk in zip(symbols, ns_tickers):
            try:
                df = data[tk].dropna(how="all") if len(ns_tickers) > 1 else data.dropna(how="all")
                if df.empty:
                    continue
                tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
                cum_vol = df["Volume"].sum()
                intraday_vwap = float((tp * df["Volume"]).sum() / cum_vol) if cum_vol > 0 else float(df["Close"].iloc[-1])
                snapshots[sym] = {
                    "current_price": round(float(df["Close"].iloc[-1]), 2),
                    "day_high": round(float(df["High"].max()), 2),
                    "day_low": round(float(df["Low"].min()), 2),
                    "intraday_vwap": round(intraday_vwap, 2),
                    "last_5m_vol": int(df["Volume"].iloc[-1])
                }
            except Exception:
                continue
        return snapshots