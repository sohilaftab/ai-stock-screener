import yfinance as yf
import pandas as pd

class MarketDataScanner:
    def __init__(self, tickers):
        self.tickers = tickers

    def scan_volume_breakouts(self, vol_multiplier=1.5, min_price_change_pct=1.5, top_n=5):
        """
        Downloads 25 trading days of daily OHLCV data in a single multi-threaded batch.
        Compares today's live intraday volume against the 20-day Average Daily Volume (ADV).
        """
        print(f"[INFO] Scanning {len(self.tickers)} stocks via Yahoo Finance batch API...")
        
        # Batch download: period="1mo" gives ~22 trading days. Latest row is LIVE intraday data.
        data = yf.download(
            tickers=self.tickers,
            period="1mo",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )

        breakouts = []
        for ticker in self.tickers:
            try:
                df = data[ticker].dropna(how="all")
                if len(df) < 10:
                    continue

                # Separate historical completed days from today's live session
                hist_df = df.iloc[:-1]
                live_row = df.iloc[-1]

                avg_daily_vol = hist_df["Volume"].tail(20).mean()
                live_vol = float(live_row["Volume"])
                prev_close = float(hist_df["Close"].iloc[-1])
                live_price = float(live_row["Close"])

                if avg_daily_vol <= 0 or prev_close <= 0:
                    continue

                vol_ratio = live_vol / avg_daily_vol
                price_change_pct = ((live_price - prev_close) / prev_close) * 100

                # Trigger condition: Live Volume > (Average Volume * 1.5) + meaningful price action
                if vol_ratio >= vol_multiplier and abs(price_change_pct) >= min_price_change_pct:
                    clean_symbol = ticker.replace(".NS", "")
                    breakouts.append({
                        "symbol": clean_symbol,
                        "ltp": round(live_price, 2),
                        "change_pct": round(price_change_pct, 2),
                        "live_volume": int(live_vol),
                        "avg_volume": int(avg_daily_vol),
                        "vol_ratio": round(vol_ratio, 2),
                        "vol_surge_pct": int(vol_ratio * 100)
                    })
            except Exception:
                continue

        result_df = pd.DataFrame(breakouts)
        if not result_df.empty:
            result_df.sort_values(by="vol_ratio", ascending=False, inplace=True)
            return result_df.head(top_n)
        return result_df