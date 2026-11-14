import time
import requests
import pandas as pd
from datetime import datetime
import io

class NSEClient:
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        self.session.headers.update(self.headers)
        self._initialize_session()

    def _initialize_session(self):
        """Establishes a valid browser session and stores NSE cookies."""
        try:
            # Step 1: Hit homepage to get primary cookies
            self.session.get(self.base_url, timeout=15)
            time.sleep(1.5)
            # Step 2: Hit the pre-open HTML page to set Referer context
            self.session.get(
                f"{self.base_url}/market-data/pre-open-market-cm-and-emerge-market",
                timeout=15
            )
            # Update headers for JSON API requests
            self.session.headers.update({
                "Accept": "*/*",
                "Referer": f"{self.base_url}/market-data/pre-open-market-cm-and-emerge-market",
                "X-Requested-With": "XMLHttpRequest"
            })
        except requests.RequestException as e:
            print(f"[ERROR] Failed to initialize NSE session: {e}")

    def _get_json(self, endpoint, retries=3):
        """Fetches JSON from NSE API with automatic cookie refresh on 401/403."""
        url = f"{self.base_url}{endpoint}"
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code in [401, 403]:
                    print(f"[WARN] NSE Cookie expired (HTTP {response.status_code}). Refreshing...")
                    self._initialize_session()
                    time.sleep(2)
            except requests.RequestException as e:
                print(f"[WARN] Request attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        return None

    def get_pre_open_movers(self, key="FO", min_gap_pct=2.0, min_volume=50000, top_n=5):
        """
        Fetches pre-market data and filters for stocks gapping > min_gap_pct with strong volume.
        Keys: 'ALL', 'FO' (Futures & Options - best for liquidity), 'NIFTY'
        """
        endpoint = f"/api/market-data-pre-open?key={key}"
        raw_data = self._get_json(endpoint)
        if not raw_data or "data" not in raw_data:
            print("[ERROR] No pre-open data returned from NSE.")
            return pd.DataFrame()

        records = []
        for item in raw_data["data"]:
            meta = item.get("metadata", {})
            detail = item.get("detail", {}).get("preOpenMarket", {})
            
            symbol = meta.get("symbol")
            iep = meta.get("iep", 0)               # Indicative Equilibrium Price
            p_change = meta.get("pChange", 0)      # Percentage Change
            final_qty = meta.get("finalQuantity", 0) # Pre-open matched volume
            prev_close = meta.get("previousClose", 0)
            total_buy_qty = detail.get("totalBuyQuantity", 0)
            total_sell_qty = detail.get("totalSellQuantity", 0)

            if symbol and iep > 50:  # Ignore penny stocks under Rs 50
                records.append({
                    "symbol": symbol,
                    "iep": float(iep),
                    "prev_close": float(prev_close),
                    "pChange": float(p_change),
                    "abs_change": abs(float(p_change)),
                    "volume": int(final_qty),
                    "buy_sell_ratio": round(total_buy_qty / max(total_sell_qty, 1), 2)
                })

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Filter by gap percentage and minimum pre-open matched volume
        filtered = df[(df["abs_change"] >= min_gap_pct) & (df["volume"] >= min_volume)].copy()
        filtered.sort_values(by=["abs_change", "volume"], ascending=[False, False], inplace=True)
        return filtered.head(top_n)

    def get_eod_high_delivery_stocks(self, symbols, min_delivery_pct=60.0):
        """
        Checks End-of-Day delivery percentage for a list of symbols using NSE quote trade_info API.
        Also supports pulling the daily Security-wise Delivery Position CSV from NSE archives.
        """
        high_delivery = []
        for sym in symbols:
            endpoint = f"/api/quote-equity?symbol={requests.utils.quote(sym)}&section=trade_info"
            data = self._get_json(endpoint)
            if data:
                sec_wise = data.get("securityWiseDP", {})
                deliv_pct = sec_wise.get("deliveryToTradedQuantity", 0)
                qty_traded = sec_wise.get("quantityTraded", 0)
                if float(deliv_pct or 0) >= min_delivery_pct:
                    high_delivery.append({
                        "symbol": sym,
                        "delivery_pct": float(deliv_pct),
                        "quantity_traded": int(qty_traded)
                    })
            time.sleep(0.4)  # Respect NSE rate limits (max 3 req/sec)
        return pd.DataFrame(high_delivery)

    def get_nifty_universe_tickers(self, index_url="https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"):
        """Downloads official NSE index constituent list and converts to Yahoo Finance (.NS) format."""
        try:
            res = self.session.get(index_url, timeout=15)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                return [f"{sym.strip()}.NS" for sym in df["Symbol"].tolist() if "DUMMY" not in sym]
        except Exception as e:
            print(f"[WARN] Could not download NSE index CSV, using fallback Nifty 50 list: {e}")
        
        # Fallback liquid tickers if archive server is slow
        fallback = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN",
            "LICI", "ITC", "HCLTECH", "LT", "BAJFINANCE", "SUNPHARMA", "MARUTI", "TATAMOTORS",
            "NTPC", "KOTAKBANK", "AXISBANK", "ONGC", "ADANIENT", "TITAN", "ULTRACEMCO",
            "POWERGRID", "HAL", "TRENT", "BEL", "SIEMENS", "ZOMATO", "DLF", "TATASTEEL"
        ]
        return [f"{s}.NS" for s in fallback]