import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_NIFTY_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK", "KOTAKBANK", "TATAMOTORS", "SUNPHARMA", "MARUTI",
    "NTPC", "ONGC", "POWERGRID", "BAJFINANCE", "TITAN", "ADANIENT", "ADANIPORTS",
    "HAL", "BEL", "MAZDOCK", "TRENT", "ZOMATO", "JIOFIN", "VBL", "CHOLAFIN",
    "INDHOTEL", "DIXON", "POLYCAB", "PERSISTENT", "COFORGE", "MPHASIS", "TATASTEEL",
    "JSWSTEEL", "HINDALCO", "COALINDIA", "BPCL", "IOC", "GAIL", "DLF", "LODHA",
    "BANKBARODA", "PNB", "CANBK", "PFC", "RECLTD", "IRCTC"
]

# Direct mapping of NSE stocks to their primary NSE Sectoral Index on Yahoo Finance
SYMBOL_TO_SECTOR_INDEX = {
    "HDFCBANK": "^NSEBANK", "ICICIBANK": "^NSEBANK", "AXISBANK": "^NSEBANK",
    "KOTAKBANK": "^NSEBANK", "BAJFINANCE": "^NSEBANK", "JIOFIN": "^NSEBANK",
    "CHOLAFIN": "^NSEBANK", "PFC": "^NSEBANK", "RECLTD": "^NSEBANK",
    "SBIN": "^CNXPSUBANK", "BANKBARODA": "^CNXPSUBANK", "PNB": "^CNXPSUBANK", "CANBK": "^CNXPSUBANK",
    "TCS": "^CNXIT", "INFY": "^CNXIT", "PERSISTENT": "^CNXIT", "COFORGE": "^CNXIT", "MPHASIS": "^CNXIT",
    "TATAMOTORS": "^CNXAUTO", "MARUTI": "^CNXAUTO",
    "SUNPHARMA": "^CNXPHARMA",
    "TATASTEEL": "^CNXMETAL", "JSWSTEEL": "^CNXMETAL", "HINDALCO": "^CNXMETAL", "COALINDIA": "^CNXMETAL",
    "RELIANCE": "^CNXENERGY", "NTPC": "^CNXENERGY", "ONGC": "^CNXENERGY",
    "POWERGRID": "^CNXENERGY", "BPCL": "^CNXENERGY", "IOC": "^CNXENERGY", "GAIL": "^CNXENERGY",
    "ITC": "^CNXFMCG", "VBL": "^CNXFMCG", "TRENT": "^CNXFMCG", "TITAN": "^CNXFMCG",
    "DLF": "^CNXREALTY", "LODHA": "^CNXREALTY",
    "LT": "^CNXINFRA", "ADANIPORTS": "^CNXINFRA", "ADANIENT": "^CNXINFRA",
    "HAL": "^CNXINFRA", "BEL": "^CNXINFRA", "MAZDOCK": "^CNXINFRA",
    "BHARTIARTL": "^CNXINFRA", "POLYCAB": "^CNXINFRA", "DIXON": "^CNXINFRA",
    "IRCTC": "^CNXINFRA", "INDHOTEL": "^CNXINFRA", "ZOMATO": "^CNXINFRA",
}

YAHOO_SECTOR_TO_INDEX = {
    "Financial Services": "^NSEBANK",
    "Technology": "^CNXIT",
    "Communication Services": "^CNXIT",
    "Consumer Cyclical": "^CNXAUTO",
    "Healthcare": "^CNXPHARMA",
    "Basic Materials": "^CNXMETAL",
    "Energy": "^CNXENERGY",
    "Utilities": "^CNXENERGY",
    "Consumer Defensive": "^CNXFMCG",
    "Real Estate": "^CNXREALTY",
    "Industrials": "^CNXINFRA",
}

SECTOR_INDEX_TICKERS = [
    "^NSEI", "^NSEBANK", "^CNXPSUBANK", "^CNXIT", "^CNXAUTO",
    "^CNXPHARMA", "^CNXMETAL", "^CNXENERGY", "^CNXFMCG", "^CNXREALTY", "^CNXINFRA"
]


class NSESessionClient:
    """
    Persistent session client for direct nseindia.com JSON endpoints with automatic
    cookie initialization, rate throttling, and graceful fallback on cloud IPs.
    """

    BASE_URL = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE",
        "Connection": "keep-alive",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._cookie_ts = 0.0
        self._last_req_ts = 0.0

    def _ensure_cookies(self, force=False):
        now = time.time()
        if not force and (now - self._cookie_ts) < 240 and self.session.cookies:
            return True
        try:
            self.session.get(self.BASE_URL, timeout=6)
            time.sleep(0.35)
            self.session.get(f"{self.BASE_URL}/get-quotes/equity?symbol=RELIANCE", timeout=6)
            self._cookie_ts = time.time()
            return True
        except Exception:
            return False

    def get_json(self, path: str):
        if not self._ensure_cookies():
            return None

        elapsed = time.time() - self._last_req_ts
        if elapsed < 0.35:
            time.sleep(0.35 - elapsed)

        url = f"{self.BASE_URL}{path}"
        for attempt in range(2):
            try:
                self._last_req_ts = time.time()
                resp = self.session.get(url, timeout=7)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (401, 403) and attempt == 0:
                    self._ensure_cookies(force=True)
                    continue
            except Exception:
                if attempt == 0:
                    self._ensure_cookies(force=True)
        return None


_NSE_CLIENT = NSESessionClient()


class MarketDataScanner:
    """
    Hybrid NSE Direct + Multi-Timeframe Intraday Scanner providing:
    1. Live NSE quotes, circuit bands (lowerCP/upperCP), Delivery %, & Order-Book Imbalance (OBI)
    2. 5m/15m Multi-Timeframe VWAP, VWAP Slope, EMA9/EMA21 Alignment, ORB & Time-Normalized RVOL
    3. Live NSE Sectoral Index alignment & Stock Option Chain Max Call/Put OI walls
    4. FII / DII daily institutional net flows
    """

    _sector_cache_ts = 0.0
    _sector_cache_data = {}

    def __init__(self, universe=None):
        self.universe = universe or DEFAULT_NIFTY_UNIVERSE

    @staticmethod
    def _clean_sym(symbol: str) -> str:
        return str(symbol).replace(".NS", "").replace(".BO", "").strip().upper()

    @staticmethod
    def _safe_float(val, default=0.0) -> float:
        try:
            if val is None or val == "":
                return float(default)
            return float(str(val).replace(",", "").strip())
        except Exception:
            return float(default)

    @classmethod
    def get_sector_index_for_stock(cls, symbol: str, yahoo_sector: str = "") -> str:
        clean = cls._clean_sym(symbol)
        if clean in SYMBOL_TO_SECTOR_INDEX:
            return SYMBOL_TO_SECTOR_INDEX[clean]
        if yahoo_sector and yahoo_sector in YAHOO_SECTOR_TO_INDEX:
            return YAHOO_SECTOR_TO_INDEX[yahoo_sector]
        return "^NSEI"

    # ------------------------------------------------------------------
    # LIVE NSE SECTORAL INDICES SNAPSHOT (60s TTL CACHE)
    # ------------------------------------------------------------------
    @classmethod
    def fetch_sectoral_indices_snapshot(cls, force_refresh: bool = False) -> dict:
        """
        Batch-downloads live intraday % change for Nifty 50 and all major NSE Sectoral
        Indices so every stock can be checked against its sector's live tape.
        """
        now = time.time()
        if not force_refresh and cls._sector_cache_data and (now - cls._sector_cache_ts) < 60.0:
            return cls._sector_cache_data

        snapshot = {idx: {"ltp": 0.0, "change_pct": 0.0} for idx in SECTOR_INDEX_TICKERS}
        try:
            raw = yf.download(
                tickers=SECTOR_INDEX_TICKERS,
                period="5d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if not raw.empty and isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                for idx in SECTOR_INDEX_TICKERS:
                    if idx not in level0:
                        continue
                    closes = raw[idx]["Close"].dropna()
                    if len(closes) >= 2:
                        ltp = float(closes.iloc[-1])
                        prev = float(closes.iloc[-2])
                        chg = ((ltp - prev) / prev) * 100.0 if prev > 0 else 0.0
                        snapshot[idx] = {"ltp": round(ltp, 2), "change_pct": round(chg, 2)}
                    elif len(closes) == 1:
                        snapshot[idx] = {"ltp": round(float(closes.iloc[-1]), 2), "change_pct": 0.0}
        except Exception as e:
            print(f"[WARN] Sectoral indices batch fetch fallback: {e}")

        cls._sector_cache_data = snapshot
        cls._sector_cache_ts = now
        return snapshot

    # ------------------------------------------------------------------
    # DIRECT NSE QUOTE + CIRCUIT BANDS + DELIVERY % + ORDER-BOOK IMBALANCE
    # ------------------------------------------------------------------
    @classmethod
    def _parse_order_book(cls, ob: dict, ltp: float) -> tuple:
        """
        Computes normalized Order-Book Imbalance (OBI in [-1.0, +1.0]) and
        top-of-book bid-ask spread percentage from NSE marketDeptOrderBook / orderBook.
        """
        if not isinstance(ob, dict):
            return 0.0, 0.0

        tot_buy = cls._safe_float(ob.get("totalBuyQuantity", ob.get("totalBuyQty", 0.0)))
        tot_sell = cls._safe_float(ob.get("totalSellQuantity", ob.get("totalSellQty", 0.0)))
        denom = tot_buy + tot_sell
        obi_ratio = round((tot_buy - tot_sell) / denom, 3) if denom > 0 else 0.0

        bids = ob.get("bid", ob.get("buy", [])) or []
        asks = ob.get("ask", ob.get("sell", [])) or []
        best_bid = cls._safe_float(bids[0].get("price", 0.0)) if bids and isinstance(bids[0], dict) else 0.0
        best_ask = cls._safe_float(asks[0].get("price", 0.0)) if asks and isinstance(asks[0], dict) else 0.0

        spread_pct = 0.0
        if best_bid > 0 and best_ask > best_bid and ltp > 0:
            spread_pct = round(((best_ask - best_bid) / ltp) * 100.0, 3)

        return obi_ratio, spread_pct

    @classmethod
    def fetch_nse_equity_quote(cls, symbol: str) -> dict:
        """
        Fetches live quote, VWAP, circuit bands (lowerCP/upperCP), delivery %, and
        order-book imbalance (OBI + spread). Supports NSE's 2026 GetQuoteApi endpoint
        and falls back cleanly to legacy /api/quote-equity.
        """
        clean = cls._clean_sym(symbol)
        encoded = urllib.parse.quote(clean)

        # 1) Try updated 2026 NSE NextApi GetQuoteApi endpoint first
        next_payload = _NSE_CLIENT.get_json(
            f"/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolData&marketType=N&series=EQ&symbol={encoded}"
        )
        if isinstance(next_payload, dict) and next_payload.get("equityResponse"):
            eq_list = next_payload.get("equityResponse") or []
            if eq_list and isinstance(eq_list[0], dict):
                eq = eq_list[0]
                meta = eq.get("metaData", {}) or {}
                t_info = eq.get("tradeInfo", {}) or {}
                ob = eq.get("orderBook", {}) or t_info.get("marketDeptOrderBook", {}) or {}
                p_info = eq.get("priceInfo", {}) or {}
                sec_dp = eq.get("securityWiseDP", {}) or t_info.get("securityWiseDP", {}) or {}

                ltp = cls._safe_float(meta.get("closePrice") or p_info.get("lastPrice") or t_info.get("lastPrice"))
                if ltp <= 0:
                    ltp = cls._safe_float(meta.get("averagePrice"))
                if ltp > 0:
                    d_high = cls._safe_float(meta.get("dayHigh") or (p_info.get("intraDayHighLow") or {}).get("max"), ltp)
                    d_low = cls._safe_float(meta.get("dayLow") or (p_info.get("intraDayHighLow") or {}).get("min"), ltp)
                    vwap = cls._safe_float(meta.get("averagePrice") or p_info.get("vwap"), ltp)
                    p_change = cls._safe_float(meta.get("pChange") or p_info.get("pChange"), 0.0)
                    lower_cp = cls._safe_float(p_info.get("lowerCP") or meta.get("lowerCP"), 0.0)
                    upper_cp = cls._safe_float(p_info.get("upperCP") or meta.get("upperCP"), 0.0)
                    delivery_pct = cls._safe_float(
                        sec_dp.get("deliveryToTradedQuantity") or t_info.get("deliveryToTradedQuantity"), 0.0
                    )
                    obi_ratio, spread_pct = cls._parse_order_book(ob, ltp)

                    return {
                        "symbol": clean,
                        "current_price": round(ltp, 2),
                        "day_high": round(d_high, 2),
                        "day_low": round(d_low, 2),
                        "intraday_vwap": round(vwap, 2),
                        "change_pct": round(p_change, 2),
                        "lower_cp": round(lower_cp, 2),
                        "upper_cp": round(upper_cp, 2),
                        "delivery_pct": round(delivery_pct, 2),
                        "obi_ratio": obi_ratio,
                        "bid_ask_spread_pct": spread_pct,
                        "source": "NSE_GETQUOTE_API",
                    }

        # 2) Fallback to standard /api/quote-equity route
        quote_data = _NSE_CLIENT.get_json(f"/api/quote-equity?symbol={encoded}")
        if not isinstance(quote_data, dict) or "priceInfo" not in quote_data:
            return {}

        p_info = quote_data.get("priceInfo", {}) or {}
        intra = p_info.get("intraDayHighLow", {}) or {}
        ltp = cls._safe_float(p_info.get("lastPrice"), 0.0)
        if ltp <= 0:
            return {}

        lower_cp = cls._safe_float(p_info.get("lowerCP"), 0.0)
        upper_cp = cls._safe_float(p_info.get("upperCP"), 0.0)
        vwap = cls._safe_float(p_info.get("vwap"), ltp)
        d_high = cls._safe_float(intra.get("max"), ltp)
        d_low = cls._safe_float(intra.get("min"), ltp)
        p_change = cls._safe_float(p_info.get("pChange"), 0.0)

        ob = quote_data.get("marketDeptOrderBook", {}) or {}
        dp_info = quote_data.get("securityWiseDP", {}) or {}
        if not ob and not dp_info:
            trade_data = _NSE_CLIENT.get_json(f"/api/quote-equity?symbol={encoded}&section=trade_info") or {}
            if isinstance(trade_data, dict):
                ob = trade_data.get("marketDeptOrderBook", {}) or {}
                dp_info = trade_data.get("securityWiseDP", {}) or {}

        delivery_pct = cls._safe_float(dp_info.get("deliveryToTradedQuantity"), 0.0)
        obi_ratio, spread_pct = cls._parse_order_book(ob, ltp)

        return {
            "symbol": clean,
            "current_price": round(ltp, 2),
            "day_high": round(d_high, 2),
            "day_low": round(d_low, 2),
            "intraday_vwap": round(vwap, 2),
            "change_pct": round(p_change, 2),
            "lower_cp": round(lower_cp, 2),
            "upper_cp": round(upper_cp, 2),
            "delivery_pct": round(delivery_pct, 2),
            "obi_ratio": obi_ratio,
            "bid_ask_spread_pct": spread_pct,
            "source": "NSE_DIRECT",
        }

    @classmethod
    def fetch_option_chain_walls(cls, symbol: str, spot_price: float) -> dict:
        clean = cls._clean_sym(symbol)
        encoded = urllib.parse.quote(clean)
        oc = _NSE_CLIENT.get_json(f"/api/option-chain-equities?symbol={encoded}")
        if not isinstance(oc, dict) or "records" not in oc:
            return {"max_call_oi_strike": 0.0, "max_put_oi_strike": 0.0, "pcr": 1.0}

        records = oc.get("records", {})
        expiry_dates = records.get("expiryDates", [])
        if not expiry_dates:
            return {"max_call_oi_strike": 0.0, "max_put_oi_strike": 0.0, "pcr": 1.0}

        nearest_exp = expiry_dates[0]
        rows = [r for r in records.get("data", []) if r.get("expiryDate") == nearest_exp]

        max_ce_oi = -1.0
        max_ce_strike = 0.0
        max_pe_oi = -1.0
        max_pe_strike = 0.0
        total_ce = 0.0
        total_pe = 0.0

        for r in rows:
            strike = cls._safe_float(r.get("strikePrice"), 0.0)
            ce_oi = cls._safe_float((r.get("CE") or {}).get("openInterest"), 0.0)
            pe_oi = cls._safe_float((r.get("PE") or {}).get("openInterest"), 0.0)
            total_ce += ce_oi
            total_pe += pe_oi

            if strike >= spot_price and ce_oi > max_ce_oi:
                max_ce_oi = ce_oi
                max_ce_strike = strike
            if strike <= spot_price and pe_oi > max_pe_oi:
                max_pe_oi = pe_oi
                max_pe_strike = strike

        pcr = round(total_pe / max(total_ce, 1.0), 2)
        return {
            "max_call_oi_strike": round(max_ce_strike, 2),
            "max_put_oi_strike": round(max_pe_strike, 2),
            "pcr": pcr,
        }

    @classmethod
    def fetch_nse_microstructure(cls, symbol: str, spot_price: float = 0.0) -> dict:
        """Combines NSE direct quote, circuit bands, delivery %, OBI/spread, and F&O OI walls."""
        q = cls.fetch_nse_equity_quote(symbol)
        ltp = q.get("current_price", spot_price)
        walls = cls.fetch_option_chain_walls(symbol, ltp) if ltp > 0 else {}
        return {
            "delivery_pct": q.get("delivery_pct", 0.0),
            "lower_cp": q.get("lower_cp", 0.0),
            "upper_cp": q.get("upper_cp", 0.0),
            "obi_ratio": q.get("obi_ratio", 0.0),
            "bid_ask_spread_pct": q.get("bid_ask_spread_pct", 0.0),
            "nse_live_vwap": q.get("intraday_vwap", 0.0),
            "max_call_oi_strike": walls.get("max_call_oi_strike", 0.0),
            "max_put_oi_strike": walls.get("max_put_oi_strike", 0.0),
            "pcr": walls.get("pcr", 1.0),
        }

    @staticmethod
    def fetch_fii_dii_activity() -> dict:
        data = _NSE_CLIENT.get_json("/api/fiidiiTradeReact")
        result = {"fii_net_cr": 0.0, "dii_net_cr": 0.0, "date": ""}
        if not isinstance(data, list):
            return result
        for row in data:
            cat = str(row.get("category", "")).upper()
            net_val = float(str(row.get("netValue", "0")).replace(",", "") or 0.0)
            if "FII" in cat or "FPI" in cat:
                result["fii_net_cr"] = round(net_val, 2)
                result["date"] = str(row.get("date", ""))
            elif "DII" in cat:
                result["dii_net_cr"] = round(net_val, 2)
        return result

    # ------------------------------------------------------------------
    # UPGRADE 1: 5m/15m MULTI-TIMEFRAME CHART FLOW, ORB & TIME-NORM RVOL
    # ------------------------------------------------------------------
    @classmethod
    def compute_intraday_chart_flow(cls, symbols: list) -> dict:
        """
        Batch-downloads 10 days of 5-minute intraday candles to compute:
        1. True cumulative intraday VWAP & 15m VWAP slope (% change over last 3 x 15m bars)
        2. 15m EMA-9 vs EMA-21 trend alignment (mtf_15m_trend)
        3. Opening Range Breakout (09:15-09:30 IST High/Low -> ORB_BREAKOUT_UP/DOWN)
        4. Time-Normalized RVOL (today's cumulative volume up to current HH:MM vs
           historical average cumulative volume up to the exact same HH:MM)
        """
        flow_map = {}
        if not symbols:
            return flow_map

        ns_map = {s: (s if s.endswith(".NS") else f"{s}.NS") for s in symbols}
        tickers = list(set(ns_map.values()))

        try:
            raw = yf.download(
                tickers=tickers,
                period="10d",
                interval="5m",
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"[WARN] Intraday 5m batch fetch failed: {e}")
            return flow_map

        if raw.empty:
            return flow_map

        is_multi = isinstance(raw.columns, pd.MultiIndex)
        level0 = set(raw.columns.get_level_values(0)) if is_multi else set()

        for orig_sym, ns_sym in ns_map.items():
            try:
                if is_multi:
                    if ns_sym not in level0:
                        continue
                    df = raw[ns_sym].dropna(subset=["Close", "High", "Low"]).copy()
                else:
                    if len(tickers) > 1:
                        continue
                    df = raw.dropna(subset=["Close", "High", "Low"]).copy()

                if len(df) < 5:
                    continue

                if df.index.tz is None:
                    df.index = df.index.tz_localize(IST)
                else:
                    df.index = df.index.tz_convert(IST)

                df["Volume"] = df["Volume"].fillna(0.0)
                latest_date = df.index[-1].date()
                today_5m = df[df.index.date == latest_date].copy()
                hist_5m = df[df.index.date < latest_date].copy()

                if today_5m.empty:
                    continue

                ltp = float(today_5m["Close"].iloc[-1])
                d_high = float(today_5m["High"].max())
                d_low = float(today_5m["Low"].min())
                last_ts = today_5m.index[-1]
                cutoff_mins = last_ts.hour * 60 + last_ts.minute

                # 1) True cumulative intraday VWAP on 5m bars
                tp_5m = (today_5m["High"] + today_5m["Low"] + today_5m["Close"]) / 3.0
                cum_vol = today_5m["Volume"].cumsum()
                cum_pv = (tp_5m * today_5m["Volume"]).cumsum()
                vwap_series = cum_pv / cum_vol.replace(0, pd.NA)
                vwap_series = vwap_series.ffill().fillna(ltp)
                vwap_intraday = float(vwap_series.iloc[-1])

                # 2) 15-minute Resampling -> VWAP Slope + 15m EMA9 / EMA21 Alignment
                df_15m = df.resample("15min").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }).dropna(subset=["Close"])

                ema9_15m = float(df_15m["Close"].ewm(span=9, adjust=False).mean().iloc[-1])
                ema21_15m = float(df_15m["Close"].ewm(span=21, adjust=False).mean().iloc[-1])

                vwap_15m_today = vwap_series.resample("15min").last().dropna()
                if len(vwap_15m_today) >= 3:
                    v_now = float(vwap_15m_today.iloc[-1])
                    v_prev = float(vwap_15m_today.iloc[-3])
                    vwap_slope_pct = ((v_now - v_prev) / v_prev) * 100.0 if v_prev > 0 else 0.0
                elif len(vwap_series) >= 3:
                    v_now = float(vwap_series.iloc[-1])
                    v_prev = float(vwap_series.iloc[-3])
                    vwap_slope_pct = ((v_now - v_prev) / v_prev) * 100.0 if v_prev > 0 else 0.0
                else:
                    vwap_slope_pct = 0.0

                if ltp > vwap_intraday and ema9_15m > ema21_15m and vwap_slope_pct >= 0.0:
                    mtf_15m_trend = "BULLISH"
                elif ltp < vwap_intraday and ema9_15m < ema21_15m and vwap_slope_pct <= 0.0:
                    mtf_15m_trend = "BEARISH"
                else:
                    mtf_15m_trend = "NEUTRAL"

                # 3) Opening Range Breakout (09:15 - 09:30 AM IST)
                orb_mask = (today_5m.index.hour * 60 + today_5m.index.minute) < (9 * 60 + 30)
                orb_bars = today_5m[orb_mask]
                if orb_bars.empty:
                    orb_bars = today_5m.head(3)

                orb_high = float(orb_bars["High"].max())
                orb_low = float(orb_bars["Low"].min())
                last_5m_close = float(today_5m["Close"].iloc[-1])

                if len(today_5m) > len(orb_bars) and last_5m_close > orb_high:
                    orb_signal = "ORB_BREAKOUT_UP"
                    orb_dist_pct = ((last_5m_close - orb_high) / orb_high) * 100.0 if orb_high > 0 else 0.0
                elif len(today_5m) > len(orb_bars) and last_5m_close < orb_low:
                    orb_signal = "ORB_BREAKDOWN_DOWN"
                    orb_dist_pct = ((orb_low - last_5m_close) / orb_low) * 100.0 if orb_low > 0 else 0.0
                else:
                    orb_signal = "INSIDE_ORB"
                    orb_dist_pct = 0.0

                # 4) Time-Normalized Relative Volume (RVOL up to exact current minute)
                today_cum_vol = float(today_5m["Volume"].sum())
                hist_same_time_vols = []
                if not hist_5m.empty:
                    for _, day_grp in hist_5m.groupby(hist_5m.index.date):
                        day_mins = day_grp.index.hour * 60 + day_grp.index.minute
                        sub_vol = float(day_grp.loc[day_mins <= cutoff_mins, "Volume"].sum())
                        if sub_vol > 0:
                            hist_same_time_vols.append(sub_vol)

                if hist_same_time_vols and today_cum_vol > 0:
                    avg_hist_cum_vol = sum(hist_same_time_vols) / len(hist_same_time_vols)
                    rvol_normalized = round(today_cum_vol / max(avg_hist_cum_vol, 1.0), 2)
                else:
                    rvol_normalized = 1.0

                # Recent 5m bar volume acceleration (last 2 bars vs session average 5m bar)
                avg_5m_vol = float(today_5m["Volume"].mean())
                recent_5m_vol = float(today_5m["Volume"].tail(2).mean())
                vol_accel_5m = round(recent_5m_vol / max(avg_5m_vol, 1.0), 2) if avg_5m_vol > 0 else 1.0

                flow_map[orig_sym] = {
                    "ltp_intraday": round(ltp, 2),
                    "day_high_intraday": round(d_high, 2),
                    "day_low_intraday": round(d_low, 2),
                    "vwap_intraday": round(vwap_intraday, 2),
                    "vwap_slope_pct": round(vwap_slope_pct, 3),
                    "mtf_15m_trend": mtf_15m_trend,
                    "ema9_15m": round(ema9_15m, 2),
                    "ema21_15m": round(ema21_15m, 2),
                    "orb_high": round(orb_high, 2),
                    "orb_low": round(orb_low, 2),
                    "orb_signal": orb_signal,
                    "orb_dist_pct": round(orb_dist_pct, 2),
                    "rvol_normalized": rvol_normalized,
                    "vol_accel_5m": vol_accel_5m,
                }
            except Exception:
                continue

        return flow_map

    @classmethod
    def fetch_live_intraday_snapshot(cls, symbols: list) -> dict:
        snapshots = {}
        if not symbols:
            return snapshots

        missing = []
        for sym in symbols:
            direct = cls.fetch_nse_equity_quote(sym)
            if direct and direct.get("current_price", 0) > 0:
                snapshots[sym] = direct
            else:
                missing.append(sym)

        if not missing:
            return snapshots

        for sym in missing:
            ns_sym = sym if sym.endswith(".NS") else f"{sym}.NS"
            try:
                df = yf.Ticker(ns_sym).history(period="1d", interval="5m")
                if df.empty:
                    df = yf.Ticker(ns_sym).history(period="2d", interval="1d")
                if df.empty:
                    continue

                ltp = float(df["Close"].iloc[-1])
                d_high = float(df["High"].max())
                d_low = float(df["Low"].min())
                tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
                vol_sum = float(df["Volume"].sum())
                vwap = float((tp * df["Volume"]).sum() / vol_sum) if vol_sum > 0 else ltp

                snapshots[sym] = {
                    "symbol": cls._clean_sym(sym),
                    "current_price": round(ltp, 2),
                    "day_high": round(d_high, 2),
                    "day_low": round(d_low, 2),
                    "intraday_vwap": round(vwap, 2),
                    "lower_cp": 0.0,
                    "upper_cp": 0.0,
                    "delivery_pct": 0.0,
                    "obi_ratio": 0.0,
                    "bid_ask_spread_pct": 0.0,
                    "source": "YFINANCE_FALLBACK",
                }
            except Exception:
                continue

        return snapshots

    def fetch_premarket_data(self) -> pd.DataFrame:
        raw = _NSE_CLIENT.get_json("/api/market-data-pre-open?key=NIFTY")
        if not isinstance(raw, dict) or "data" not in raw:
            return pd.DataFrame()

        rows = []
        for item in raw.get("data", []):
            meta = item.get("metadata", {})
            detail = item.get("detail", {}).get("preOpenMarket", {})
            sym = meta.get("symbol")
            if not sym:
                continue
            total_buy = self._safe_float(detail.get("totalBuyQuantity"), 0.0)
            total_sell = self._safe_float(detail.get("totalSellQuantity"), 0.0)
            bs_ratio = round(total_buy / max(total_sell, 1.0), 2)
            rows.append({
                "symbol": sym,
                "iep": self._safe_float(detail.get("IEP"), 0.0),
                "pChange": self._safe_float(meta.get("pChange"), 0.0),
                "buy_sell_ratio": bs_ratio,
            })
        return pd.DataFrame(rows)

    def scan_volume_breakouts(self, top_n: int = 15) -> pd.DataFrame:
        """
        Two-stage institutional scanner:
        Stage 1: Fast daily pre-filter (ATR, RSI, trend) across the Nifty universe.
        Stage 2: Enrich top candidates with 5m/15m Multi-Timeframe Flow (VWAP slope,
                 ORB, Time-Normalized RVOL), Live Sector Index % change, and Direct
                 NSE Microstructure (Delivery %, Circuit Bands, OBI, Options OI Walls).
        """
        ns_tickers = [f"{s}.NS" for s in self.universe]
        raw = yf.download(
            tickers=ns_tickers,
            period="1mo",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()

        candidates = []
        for sym in self.universe:
            ns_sym = f"{sym}.NS"
            try:
                if not isinstance(raw.columns, pd.MultiIndex) or ns_sym not in raw.columns.get_level_values(0):
                    continue
                df = raw[ns_sym].dropna(how="all")
                if len(df) < 10:
                    continue

                ltp = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                d_high = float(df["High"].iloc[-1])
                d_low = float(df["Low"].iloc[-1])
                chg_pct = ((ltp - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

                cur_vol = float(df["Volume"].iloc[-1])
                avg_vol_10 = float(df["Volume"].iloc[-11:-1].mean())
                raw_vol_ratio = cur_vol / max(avg_vol_10, 1.0)

                tr = pd.concat([
                    df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift(1)).abs(),
                    (df["Low"] - df["Close"].shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr = float(tr.tail(14).mean())

                delta = df["Close"].diff()
                gain = delta.clip(lower=0).tail(14).mean()
                loss = (-delta.clip(upper=0)).tail(14).mean()
                rs = gain / max(loss, 1e-6)
                rsi = 100.0 - (100.0 / (1.0 + rs))

                ema_10 = float(df["Close"].tail(10).mean())
                trend = "BULLISH_UPTREND" if ltp > ema_10 and chg_pct > 0 else (
                    "BEARISH_DOWNTREND" if ltp < ema_10 and chg_pct < 0 else "RANGE_CONSOLIDATION"
                )

                candidates.append({
                    "symbol": sym,
                    "ltp": round(ltp, 2),
                    "change_pct": round(chg_pct, 2),
                    "vol_ratio": round(raw_vol_ratio, 2),
                    "atr": round(atr, 2),
                    "rsi": round(rsi, 1),
                    "day_high": round(d_high, 2),
                    "day_low": round(d_low, 2),
                    "vwap_proxy": round((d_high + d_low + ltp) / 3.0, 2),
                    "chart_trend": trend,
                })
            except Exception:
                continue

        if not candidates:
            return pd.DataFrame()

        cand_df = pd.DataFrame(candidates)
        # Pre-select slightly wider pool (top_n + 5) so morning stocks with low daily vol_ratio aren't missed
        cand_df["pre_rank"] = cand_df["change_pct"].abs() * 1.5 + cand_df["vol_ratio"]
        pool_df = cand_df.sort_values("pre_rank", ascending=False).head(max(top_n + 5, 18)).copy()
        pool_syms = pool_df["symbol"].tolist()

        # Stage 2A: Batch-compute 5m/15m Multi-Timeframe Flow, ORB & Time-Normalized RVOL
        intraday_flow_map = self.compute_intraday_chart_flow(pool_syms)

        # Stage 2B: Live NSE Sectoral Indices Snapshot
        sector_snap = self.fetch_sectoral_indices_snapshot()

        # Stage 2C: Enrich with Direct NSE Microstructure (Delivery %, OBI, Circuit Bands, OI Walls)
        def _enrich_row(row_dict):
            sym = row_dict["symbol"]
            flow = intraday_flow_map.get(sym, {})
            if flow:
                row_dict["ltp"] = flow.get("ltp_intraday", row_dict["ltp"])
                row_dict["day_high"] = flow.get("day_high_intraday", row_dict["day_high"])
                row_dict["day_low"] = flow.get("day_low_intraday", row_dict["day_low"])
                row_dict["vwap_proxy"] = flow.get("vwap_intraday", row_dict["vwap_proxy"])
                row_dict["vwap_slope_pct"] = flow.get("vwap_slope_pct", 0.0)
                row_dict["mtf_15m_trend"] = flow.get("mtf_15m_trend", "NEUTRAL")
                row_dict["orb_high"] = flow.get("orb_high", row_dict["day_high"])
                row_dict["orb_low"] = flow.get("orb_low", row_dict["day_low"])
                row_dict["orb_signal"] = flow.get("orb_signal", "INSIDE_ORB")
                row_dict["orb_dist_pct"] = flow.get("orb_dist_pct", 0.0)
                row_dict["rvol_normalized"] = flow.get("rvol_normalized", row_dict["vol_ratio"])
                row_dict["vol_accel_5m"] = flow.get("vol_accel_5m", 1.0)
                # Upgrade vol_ratio to Time-Normalized RVOL so downstream scoring uses true intraday RVOL
                row_dict["vol_ratio"] = row_dict["rvol_normalized"]
            else:
                row_dict["vwap_slope_pct"] = 0.0
                row_dict["mtf_15m_trend"] = "NEUTRAL"
                row_dict["orb_high"] = row_dict["day_high"]
                row_dict["orb_low"] = row_dict["day_low"]
                row_dict["orb_signal"] = "INSIDE_ORB"
                row_dict["orb_dist_pct"] = 0.0
                row_dict["rvol_normalized"] = row_dict["vol_ratio"]
                row_dict["vol_accel_5m"] = 1.0

            sec_idx = self.get_sector_index_for_stock(sym)
            sec_chg = float(sector_snap.get(sec_idx, {}).get("change_pct", 0.0))
            row_dict["sector_index"] = sec_idx
            row_dict["sector_chg_pct"] = sec_chg
            row_dict["sector_alpha"] = round(float(row_dict["change_pct"]) - sec_chg, 2)

            micro = self.fetch_nse_microstructure(sym, row_dict["ltp"])
            row_dict.update(micro)
            if micro.get("nse_live_vwap", 0.0) > 0:
                row_dict["vwap_proxy"] = micro["nse_live_vwap"]

            return row_dict

        with ThreadPoolExecutor(max_workers=4) as pool:
            enriched_records = list(pool.map(_enrich_row, pool_df.to_dict(orient="records")))

        res_df = pd.DataFrame(enriched_records)
        res_df["rank_score"] = res_df["rvol_normalized"] * (1.0 + res_df["change_pct"].abs() / 4.0)
        return res_df.sort_values("rank_score", ascending=False).head(top_n).reset_index(drop=True)

    def get_top_movers(self, top_n: int = 15) -> pd.DataFrame:
        return self.scan_volume_breakouts(top_n=top_n)