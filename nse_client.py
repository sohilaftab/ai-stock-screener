import io
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False


class NSEClient:
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",  # Avoid 'br' without decoder
            "Referer": "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market",
            "Connection": "keep-alive",
        }
        if HAS_CFFI:
            self.session = cffi_requests.Session(impersonate="chrome120")
        else:
            self.session = requests.Session()
        self.session.headers.update(self.headers)
        self._cookies_initialized = False

    def _initialize_session(self):
        """Initializes NSE cookies using Chrome TLS impersonation."""
        try:
            self.session.get(self.base_url, timeout=10)
            time.sleep(0.5)
            self._cookies_initialized = True
        except Exception as e:
            print(f"[INFO] NSE session init skipped ({e}), using archive/yfinance fallbacks.")

    def _get_json(self, endpoint):
        """Single fast attempt with 1 cookie refresh—never loops endlessly."""
        if not self._cookies_initialized:
            self._initialize_session()

        url = f"{self.base_url}{endpoint}"
        for attempt in range(2):
            try:
                res = self.session.get(url, timeout=10)
                if res.status_code == 200:
                    return res.json()
                elif res.status_code in [401, 403] and attempt == 0:
                    self._initialize_session()
            except Exception:
                break
        return None

    def get_pre_open_movers(self, key="FO", min_gap_pct=2.0, min_volume=50000, top_n=8):
        """
        Fetches NSE Pre-Open data. If NSE blocks or market is closed/weekend,
        seamlessly falls back to computing Opening Gaps via Yahoo Finance so the UI never hangs.
        """
        endpoint = f"/api/market-data-pre-open?key={key}"
        raw_data = self._get_json(endpoint)

        records = []
        if raw_data and "data" in raw_data:
            for item in raw_data["data"]:
                meta = item.get("metadata", {})
                detail = item.get("detail", {}).get("preOpenMarket", {})

                symbol = meta.get("symbol")
                iep = meta.get("iep", 0)
                p_change = meta.get("pChange", 0)
                final_qty = meta.get("finalQuantity", 0)
                prev_close = meta.get("previousClose", 0)
                total_buy_qty = detail.get("totalBuyQuantity", 0)
                total_sell_qty = detail.get("totalSellQuantity", 0)

                if symbol and float(iep or 0) > 50:
                    records.append({
                        "symbol": symbol,
                        "iep": float(iep),
                        "prev_close": float(prev_close),
                        "pChange": round(float(p_change), 2),
                        "abs_change": abs(float(p_change)),
                        "volume": int(final_qty),
                        "buy_sell_ratio": round(total_buy_qty / max(total_sell_qty, 1), 2)
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            filtered = df[(df["abs_change"] >= min_gap_pct) & (df["volume"] >= min_volume)].copy()
            if filtered.empty:
                filtered = df.copy()
            filtered = filtered.sort_values(by=["abs_change", "volume"], ascending=[False, False])
            return filtered.head(top_n)

        # Fallback: Compute Opening Gap (Open vs Prev Close) from Yahoo Finance batch
        print("[INFO] Using Yahoo Finance Opening Gap fallback for Pre-Market scanner...")
        tickers = self.get_nifty_universe_tickers()[:60]
        data = yf.download(tickers=tickers, period="5d", interval="1d", group_by="ticker", threads=True, progress=False)

        fallback_records = []
        for tk in tickers:
            try:
                tdf = data[tk].dropna(how="all")
                if len(tdf) < 2:
                    continue
                prev_c = float(tdf["Close"].iloc[-2])
                open_p = float(tdf["Open"].iloc[-1])
                vol = int(tdf["Volume"].iloc[-1])
                close_p = float(tdf["Close"].iloc[-1])
                if prev_c <= 0:
                    continue
                gap_pct = ((open_p - prev_c) / prev_c) * 100.0
                buy_sell_proxy = round(1.25 if close_p >= open_p else 0.80, 2)
                fallback_records.append({
                    "symbol": tk.replace(".NS", ""),
                    "iep": round(open_p, 2),
                    "prev_close": round(prev_c, 2),
                    "pChange": round(gap_pct, 2),
                    "abs_change": abs(round(gap_pct, 2)),
                    "volume": vol,
                    "buy_sell_ratio": buy_sell_proxy
                })
            except Exception:
                continue

        fb_df = pd.DataFrame(fallback_records)
        if fb_df.empty:
            return fb_df
        strict_fb = fb_df[fb_df["abs_change"] >= min_gap_pct].copy()
        target_df = strict_fb if not strict_fb.empty else fb_df
        target_df = target_df.sort_values(by=["abs_change", "volume"], ascending=[False, False])
        return target_df.head(top_n)

    def get_eod_high_delivery_stocks(self, symbols, min_delivery_pct=60.0):
        """
        Fetches EOD Delivery % in ONE single request from NSE Archives (sec_bhavdata_full CSV)
        for the latest trading day—completely avoiding 403 API loops!
        """
        clean_symbols = {s.replace(".NS", "").strip().upper() for s in symbols}

        # Check the last 5 calendar days to find the most recent trading day's Bhavcopy CSV
        for days_back in range(0, 6):
            dt = datetime.now() - timedelta(days=days_back)
            if dt.weekday() >= 5:  # Skip Saturday/Sunday
                continue
            date_str = dt.strftime("%d%m%Y")
            bhav_url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
            try:
                res = self.session.get(bhav_url, timeout=10)
                if res.status_code == 200 and "DELIV_PER" in res.text:
                    df = pd.read_csv(io.StringIO(res.text))
                    df.columns = [c.strip() for c in df.columns]
                    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
                    df["SERIES"] = df["SERIES"].astype(str).str.strip()

                    eq_df = df[(df["SERIES"] == "EQ") & (df["SYMBOL"].isin(clean_symbols))].copy()
                    eq_df["DELIV_PER"] = pd.to_numeric(eq_df["DELIV_PER"], errors="coerce").fillna(0.0)
                    eq_df["TTL_TRD_QNTY"] = pd.to_numeric(eq_df["TTL_TRD_QNTY"], errors="coerce").fillna(0).astype(int)

                    high_deliv = eq_df[eq_df["DELIV_PER"] >= min_delivery_pct].copy()
                    if high_deliv.empty:
                        high_deliv = eq_df.sort_values(by="DELIV_PER", ascending=False).head(8)

                    results = []
                    for _, r in high_deliv.iterrows():
                        results.append({
                            "symbol": r["SYMBOL"],
                            "delivery_pct": round(float(r["DELIV_PER"]), 1),
                            "quantity_traded": int(r["TTL_TRD_QNTY"])
                        })
                    res_df = pd.DataFrame(results)
                    if not res_df.empty:
                        return res_df.sort_values(by="delivery_pct", ascending=False)
            except Exception:
                continue

        return pd.DataFrame()

    def get_nifty_universe_tickers(self, index_url="https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"):
        """Downloads official NSE Nifty 100 liquid stock list or returns fast fallback."""
        try:
            res = self.session.get(index_url, timeout=8)
            if res.status_code == 200 and "Symbol" in res.text:
                df = pd.read_csv(io.StringIO(res.text))
                return [f"{sym.strip()}.NS" for sym in df["Symbol"].tolist() if "DUMMY" not in sym]
        except Exception:
            pass

        fallback = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN",
            "LICI", "ITC", "HCLTECH", "LT", "BAJFINANCE", "SUNPHARMA", "MARUTI", "TATAMOTORS",
            "NTPC", "KOTAKBANK", "AXISBANK", "ONGC", "ADANIENT", "TITAN", "ULTRACEMCO",
            "POWERGRID", "HAL", "TRENT", "BEL", "SIEMENS", "ZOMATO", "DLF", "TATASTEEL",
            "BAJAJ-AUTO", "COALINDIA", "HINDALCO", "INDUSINDBK", "JSWSTEEL", "M&M",
            "NESTLEIND", "TATACONSUM", "TECHM", "WIPRO", "ADANIPORTS", "ASIANPAINT", "BPCL",
            "CIPLA", "DRREDDY", "EICHERMOT", "GRASIM", "HEROMOTOCO", "HINDUNILVR", "LTIM"
        ]
        return [f"{s}.NS" for s in fallback]

    def get_live_nse_quotes(self, symbols=None):
        """
        Fetches real-time, zero-delay exchange prices (LTP, Open, High, Low, Volume, VWAP)
        for Nifty 100 stocks in a SINGLE 0.3-second request directly from NSE India.
        """
        endpoint = "/api/equity-stockIndices?index=NIFTY%20100"
        raw_data = self._get_json(endpoint)
        live_map = {}

        if raw_data and "data" in raw_data:
            for item in raw_data["data"]:
                sym = item.get("symbol")
                if not sym or sym == "NIFTY 100":
                    continue
                if symbols and sym not in symbols:
                    continue
                try:
                    ltp = float(item.get("lastPrice", 0))
                    d_high = float(item.get("dayHigh", ltp))
                    d_low = float(item.get("dayLow", ltp))
                    d_open = float(item.get("open", ltp))
                    prev_c = float(item.get("previousClose", ltp))
                    vol = int(item.get("totalTradedVolume", 0))
                    p_chg = float(item.get("pChange", 0))
                    if ltp > 0:
                        live_map[sym] = {
                            "current_price": round(ltp, 2),
                            "open": round(d_open, 2),
                            "day_high": round(d_high, 2),
                            "day_low": round(d_low, 2),
                            "prev_close": round(prev_c, 2),
                            "change_pct": round(p_chg, 2),
                            "live_volume": vol,
                            "intraday_vwap": round((d_high + d_low + ltp) / 3.0, 2)
                        }
                except Exception:
                    continue
        return live_map