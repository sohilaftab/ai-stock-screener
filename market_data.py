import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")

# Minimum average daily cash turnover (₹150 Crore) for clean intraday fills
MIN_DAILY_TURNOVER_INR = 150_00_00_000.0

# Liquid F&O-eligible NSE equity universe for cash intraday screening
DEFAULT_FNO_CASH_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK", "KOTAKBANK", "TMPV", "SUNPHARMA", "MARUTI",
    "NTPC", "ONGC", "POWERGRID", "BAJFINANCE", "BAJAJFINSV", "TITAN",
    "ADANIENT", "ADANIPORTS", "HAL", "BEL", "MAZDOCK", "TRENT", "ETERNAL",
    "JIOFIN", "VBL", "CHOLAFIN", "INDHOTEL", "DIXON", "POLYCAB", "PERSISTENT",
    "COFORGE", "MPHASIS", "WIPRO", "HCLTECH", "TECHM", "TATASTEEL", "JSWSTEEL",
    "HINDALCO", "COALINDIA", "BPCL", "IOC", "GAIL", "DLF", "LODHA",
    "BANKBARODA", "PNB", "CANBK", "PFC", "RECLTD", "IRCTC", "M&M",
    "EICHERMOT", "HEROMOTOCO", "TVSMOTOR", "CIPLA", "DRREDDY", "LUPIN",
    "APOLLOHOSP", "ASIANPAINT", "PIDILITIND", "HAVELLS", "SIEMENS", "ABB"
]

# Official NSE Equity Trading Holidays (YYYY-MM-DD)
NSE_HOLIDAYS = {
    "2026-01-15",
    "2026-01-26",
    "2026-03-03",
    "2026-03-26",
    "2026-03-31",
    "2026-04-03",
    "2026-04-14",
    "2026-05-01",
    "2026-05-28",
    "2026-06-26",
    "2026-09-14",
    "2026-10-02",
    "2026-10-20",
    "2026-11-10",
    "2026-11-24",
    "2026-12-25",
    "2027-01-26",
    "2027-03-08",
    "2027-03-22",
    "2027-03-26",
    "2027-04-14",
    "2027-05-01",
    "2027-08-15",
    "2027-10-02",
}

GLOBAL_MACRO_TICKERS = ["BZ=F", "^TNX", "INR=X", "^IXIC", "^GSPC"]

SYMBOL_TO_SECTOR_INDEX = {
    "HDFCBANK": "^NSEBANK", "ICICIBANK": "^NSEBANK", "AXISBANK": "^NSEBANK",
    "KOTAKBANK": "^NSEBANK", "BAJFINANCE": "^NSEBANK", "BAJAJFINSV": "^NSEBANK",
    "JIOFIN": "^NSEBANK", "CHOLAFIN": "^NSEBANK", "PFC": "^NSEBANK", "RECLTD": "^NSEBANK",
    "SBIN": "^CNXPSUBANK", "BANKBARODA": "^CNXPSUBANK", "PNB": "^CNXPSUBANK", "CANBK": "^CNXPSUBANK",
    "TCS": "^CNXIT", "INFY": "^CNXIT", "PERSISTENT": "^CNXIT", "COFORGE": "^CNXIT",
    "MPHASIS": "^CNXIT", "WIPRO": "^CNXIT", "HCLTECH": "^CNXIT", "TECHM": "^CNXIT",
    "TMPV": "^CNXAUTO", "MARUTI": "^CNXAUTO", "M&M": "^CNXAUTO",
    "EICHERMOT": "^CNXAUTO", "HEROMOTOCO": "^CNXAUTO", "TVSMOTOR": "^CNXAUTO",
    "SUNPHARMA": "^CNXPHARMA", "CIPLA": "^CNXPHARMA", "DRREDDY": "^CNXPHARMA",
    "LUPIN": "^CNXPHARMA", "APOLLOHOSP": "^CNXPHARMA",
    "TATASTEEL": "^CNXMETAL", "JSWSTEEL": "^CNXMETAL", "HINDALCO": "^CNXMETAL", "COALINDIA": "^CNXMETAL",
    "RELIANCE": "^CNXENERGY", "NTPC": "^CNXENERGY", "ONGC": "^CNXENERGY",
    "POWERGRID": "^CNXENERGY", "BPCL": "^CNXENERGY", "IOC": "^CNXENERGY", "GAIL": "^CNXENERGY",
    "ITC": "^CNXFMCG", "VBL": "^CNXFMCG", "TRENT": "^CNXFMCG", "TITAN": "^CNXFMCG",
    "ASIANPAINT": "^CNXFMCG", "PIDILITIND": "^CNXFMCG",
    "DLF": "^CNXREALTY", "LODHA": "^CNXREALTY",
    "LT": "^CNXINFRA", "ADANIPORTS": "^CNXINFRA", "ADANIENT": "^CNXINFRA",
    "HAL": "^CNXINFRA", "BEL": "^CNXINFRA", "MAZDOCK": "^CNXINFRA",
    "BHARTIARTL": "^CNXINFRA", "POLYCAB": "^CNXINFRA", "DIXON": "^CNXINFRA",
    "HAVELLS": "^CNXINFRA", "SIEMENS": "^CNXINFRA", "ABB": "^CNXINFRA",
    "IRCTC": "^CNXINFRA", "INDHOTEL": "^CNXINFRA", "ETERNAL": "^CNXINFRA",
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
    Cash-Intraday Equity Scanner over F&O-Eligible NSE Stocks:
    - Dynamically fetches live F&O equity constituents from NSE with fallback list
    - Enforces >= ₹150 Cr average daily turnover filter & circuit-band guard
    - Computes 5m/15m VWAP, VWAP slope, EMA9/EMA21, ORB & time-normalized RVOL
    - Tracks sectoral index alignment, NSE order-book imbalance, delivery %, & global macro cues
    """

    _sector_cache_ts = 0.0
    _sector_cache_data = {}
    _macro_cache_ts = 0.0
    _macro_cache_data = {}
    _fno_list_cache_ts = 0.0
    _fno_list_cache_set = set()

    def __init__(self, universe=None, min_turnover_inr: float = MIN_DAILY_TURNOVER_INR):
        self.min_turnover_inr = float(min_turnover_inr)
        if universe is not None:
            self.universe = [self._clean_sym(s) for s in universe]
        else:
            self.universe = self.get_dynamic_fno_universe()

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
    def get_dynamic_fno_universe(cls, force_refresh: bool = False) -> list:
        now = time.time()
        if not force_refresh and cls._fno_list_cache_set and (now - cls._fno_list_cache_ts) < 21600.0:
            return sorted(cls._fno_list_cache_set)

        live_syms = set()
        payload = _NSE_CLIENT.get_json(
            "/api/equity-stockIndices?index=" + urllib.parse.quote("SECURITIES IN F&O")
        )
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            for row in payload["data"]:
                raw_sym = str(row.get("symbol", "")).strip().upper()
                if not raw_sym or raw_sym.startswith("NIFTY") or raw_sym.startswith("FINNIFTY") or raw_sym.startswith("MIDCPNIFTY"):
                    continue
                live_syms.add(raw_sym)

        merged = set(DEFAULT_FNO_CASH_UNIVERSE) | live_syms if live_syms else set(DEFAULT_FNO_CASH_UNIVERSE)
        cls._fno_list_cache_set = merged
        cls._fno_list_cache_ts = now
        return sorted(merged)

    @classmethod
    def is_fno_eligible(cls, symbol: str) -> bool:
        clean = cls._clean_sym(symbol)
        if cls._fno_list_cache_set:
            return clean in cls._fno_list_cache_set
        return clean in set(DEFAULT_FNO_CASH_UNIVERSE)

    @classmethod
    def get_sector_index_for_stock(cls, symbol: str, yahoo_sector: str = "") -> str:
        clean = cls._clean_sym(symbol)
        if clean in SYMBOL_TO_SECTOR_INDEX:
            return SYMBOL_TO_SECTOR_INDEX[clean]
        if yahoo_sector and yahoo_sector in YAHOO_SECTOR_TO_INDEX:
            return YAHOO_SECTOR_TO_INDEX[yahoo_sector]
        return "^NSEI"

    @staticmethod
    def is_nse_trading_day(d: date) -> bool:
        if d.weekday() >= 5:
            return False
        return d.strftime("%Y-%m-%d") not in NSE_HOLIDAYS

    @classmethod
    def fetch_global_macro_cues(cls, force_refresh: bool = False) -> dict:
        now = time.time()
        if not force_refresh and cls._macro_cache_data and (now - cls._macro_cache_ts) < 300.0:
            return cls._macro_cache_data

        default_macro = {
            "macro_regime": "NEUTRAL",
            "brent_ltp": 0.0,
            "brent_chg_pct": 0.0,
            "us10y_yield": 0.0,
            "us10y_bps_chg": 0.0,
            "usdinr_ltp": 0.0,
            "usdinr_chg_pct": 0.0,
            "nasdaq_chg_pct": 0.0,
            "sp500_chg_pct": 0.0,
            "macro_buy_adj": 0.0,
            "macro_short_adj": 0.0,
            "macro_size_multiplier": 1.0,
            "sector_macro_adjustments": {},
            "summary": "Macro cues neutral / unavailable",
            "updated_at": datetime.now(IST).strftime("%I:%M %p IST"),
        }

        try:
            raw = yf.download(
                tickers=GLOBAL_MACRO_TICKERS,
                period="5d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
                cls._macro_cache_data = default_macro
                cls._macro_cache_ts = now
                return default_macro

            level0 = set(raw.columns.get_level_values(0))

            def _extract(tkr):
                if tkr not in level0:
                    return 0.0, 0.0, 0.0
                closes = raw[tkr]["Close"].dropna()
                if len(closes) < 2:
                    return (float(closes.iloc[-1]) if len(closes) == 1 else 0.0), 0.0, 0.0
                cur = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                chg_pct = ((cur - prev) / prev) * 100.0 if prev > 0 else 0.0
                return cur, prev, chg_pct

            brent_ltp, _, brent_chg = _extract("BZ=F")
            us10y_cur, us10y_prev, _ = _extract("^TNX")
            usdinr_ltp, _, usdinr_chg = _extract("INR=X")
            _, _, nasdaq_chg = _extract("^IXIC")
            _, _, sp500_chg = _extract("^GSPC")

            if us10y_cur > 15.0:
                us10y_cur /= 10.0
                us10y_prev /= 10.0
            us10y_bps_chg = round((us10y_cur - us10y_prev) * 100.0, 1) if us10y_prev > 0 else 0.0

            inflation_shock = (brent_chg >= 2.2) or (us10y_bps_chg >= 8.0) or (brent_chg >= 1.5 and us10y_bps_chg >= 5.0)
            risk_off = (sp500_chg <= -1.2) or (nasdaq_chg <= -1.6) or (usdinr_chg >= 0.55)
            risk_on = (sp500_chg >= 0.7) and (brent_chg <= 1.5) and (us10y_bps_chg <= 4.0) and (usdinr_chg <= 0.30)

            if inflation_shock and risk_off:
                macro_regime, macro_buy_adj, macro_short_adj, macro_size_mult = "INFLATION_RATE_SHOCK", -5.0, 4.0, 0.80
            elif inflation_shock:
                macro_regime, macro_buy_adj, macro_short_adj, macro_size_mult = "INFLATION_RATE_SHOCK", -3.0, 2.0, 0.90
            elif risk_off:
                macro_regime, macro_buy_adj, macro_short_adj, macro_size_mult = "RISK_OFF", -4.0, 3.5, 0.85
            elif risk_on:
                macro_regime, macro_buy_adj, macro_short_adj, macro_size_mult = "RISK_ON", 3.0, -2.5, 1.0
            else:
                macro_regime, macro_buy_adj, macro_short_adj, macro_size_mult = "NEUTRAL", 0.0, 0.0, 1.0

            sec_adj = {}
            it_adj = (3.0 if nasdaq_chg >= 0.9 else (-3.5 if nasdaq_chg <= -1.1 else 0.0)) + (
                2.0 if usdinr_chg >= 0.25 else (-1.5 if usdinr_chg <= -0.35 else 0.0)
            )
            sec_adj["^CNXIT"] = round(max(-5.5, min(5.5, it_adj)), 1)

            rate_adj = (-3.0 if us10y_bps_chg >= 6.0 else (2.5 if us10y_bps_chg <= -6.0 else 0.0)) + (
                -2.0 if brent_chg >= 2.0 else (1.5 if brent_chg <= -2.0 else 0.0)
            ) + (-1.5 if usdinr_chg >= 0.45 else 0.0)
            sec_adj["^NSEBANK"] = round(max(-5.5, min(5.5, rate_adj)), 1)
            sec_adj["^CNXPSUBANK"] = round(max(-5.5, min(5.5, rate_adj)), 1)
            sec_adj["^CNXREALTY"] = round(max(-6.0, min(6.0, rate_adj * 1.15)), 1)

            crude_adj = -3.5 if brent_chg >= 2.0 else (3.0 if brent_chg <= -2.0 else 0.0)
            sec_adj["^CNXAUTO"] = round(crude_adj, 1)
            sec_adj["^CNXENERGY"] = round(crude_adj * 0.6, 1)
            sec_adj["^CNXPHARMA"] = round(1.5 if (usdinr_chg >= 0.25 or risk_off) else 0.0, 1)
            sec_adj["^CNXMETAL"] = round(2.0 if sp500_chg >= 0.9 else (-2.5 if sp500_chg <= -1.1 else 0.0), 1)

            summary = (
                f"{macro_regime} | Brent ${brent_ltp:.1f} ({brent_chg:+.1f}%) | "
                f"US10Y {us10y_cur:.2f}% ({us10y_bps_chg:+.1f}bps) | "
                f"USD/INR {usdinr_ltp:.2f} ({usdinr_chg:+.2f}%) | Nasdaq {nasdaq_chg:+.2f}%"
            )

            macro_res = {
                "macro_regime": macro_regime,
                "brent_ltp": round(brent_ltp, 2),
                "brent_chg_pct": round(brent_chg, 2),
                "us10y_yield": round(us10y_cur, 2),
                "us10y_bps_chg": us10y_bps_chg,
                "usdinr_ltp": round(usdinr_ltp, 2),
                "usdinr_chg_pct": round(usdinr_chg, 2),
                "nasdaq_chg_pct": round(nasdaq_chg, 2),
                "sp500_chg_pct": round(sp500_chg, 2),
                "macro_buy_adj": macro_buy_adj,
                "macro_short_adj": macro_short_adj,
                "macro_size_multiplier": macro_size_mult,
                "sector_macro_adjustments": sec_adj,
                "summary": summary,
                "updated_at": datetime.now(IST).strftime("%I:%M %p IST"),
            }
            cls._macro_cache_data = macro_res
            cls._macro_cache_ts = now
            return macro_res
        except Exception as e:
            print(f"[WARN] Global macro cues fetch fallback: {e}")
            cls._macro_cache_data = default_macro
            cls._macro_cache_ts = now
            return default_macro

    @classmethod
    def fetch_sectoral_indices_snapshot(cls, force_refresh: bool = False) -> dict:
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

    @classmethod
    def _parse_order_book(cls, ob: dict, ltp: float) -> tuple:
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
        clean = cls._clean_sym(symbol)
        encoded = urllib.parse.quote(clean)

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
    def fetch_nse_microstructure(cls, symbol: str, spot_price: float = 0.0) -> dict:
        q = cls.fetch_nse_equity_quote(symbol)
        return {
            "delivery_pct": q.get("delivery_pct", 0.0),
            "lower_cp": q.get("lower_cp", 0.0),
            "upper_cp": q.get("upper_cp", 0.0),
            "obi_ratio": q.get("obi_ratio", 0.0),
            "bid_ask_spread_pct": q.get("bid_ask_spread_pct", 0.0),
            "nse_live_vwap": q.get("intraday_vwap", 0.0),
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

    @classmethod
    def compute_intraday_chart_flow(cls, symbols: list) -> dict:
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

                tp_5m = (today_5m["High"] + today_5m["Low"] + today_5m["Close"]) / 3.0
                cum_vol = today_5m["Volume"].cumsum()
                cum_pv = (tp_5m * today_5m["Volume"]).cumsum()
                vwap_series = (cum_pv / cum_vol.replace(0, pd.NA)).ffill().fillna(ltp)
                vwap_intraday = float(vwap_series.iloc[-1])

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
                    v_now, v_prev = float(vwap_15m_today.iloc[-1]), float(vwap_15m_today.iloc[-3])
                    vwap_slope_pct = ((v_now - v_prev) / v_prev) * 100.0 if v_prev > 0 else 0.0
                elif len(vwap_series) >= 3:
                    v_now, v_prev = float(vwap_series.iloc[-1]), float(vwap_series.iloc[-3])
                    vwap_slope_pct = ((v_now - v_prev) / v_prev) * 100.0 if v_prev > 0 else 0.0
                else:
                    vwap_slope_pct = 0.0

                if ltp > vwap_intraday and ema9_15m > ema21_15m and vwap_slope_pct >= 0.0:
                    mtf_15m_trend = "BULLISH"
                elif ltp < vwap_intraday and ema9_15m < ema21_15m and vwap_slope_pct <= 0.0:
                    mtf_15m_trend = "BEARISH"
                else:
                    mtf_15m_trend = "NEUTRAL"

                orb_mask = (today_5m.index.hour * 60 + today_5m.index.minute) < (9 * 60 + 30)
                orb_bars = today_5m[orb_mask] if not today_5m[orb_mask].empty else today_5m.head(3)
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
        active_syms = self.universe or self.get_dynamic_fno_universe()
        ns_tickers = [f"{s}.NS" for s in active_syms]
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
        for sym in active_syms:
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

                hist_closes = df["Close"].iloc[-11:-1]
                hist_vols = df["Volume"].iloc[-11:-1]
                avg_turnover_inr = float((hist_closes * hist_vols).mean()) if len(hist_closes) > 0 else (ltp * cur_vol)
                if ltp < 50.0 or avg_turnover_inr < self.min_turnover_inr:
                    continue

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
                    "fno_eligible": self.is_fno_eligible(sym),
                    "avg_daily_turnover_cr": round(avg_turnover_inr / 1_00_00_000.0, 2),
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
        cand_df["pre_rank"] = cand_df["change_pct"].abs() * 1.5 + cand_df["vol_ratio"]
        pool_df = cand_df.sort_values("pre_rank", ascending=False).head(max(top_n + 5, 18)).copy()
        pool_syms = pool_df["symbol"].tolist()

        intraday_flow_map = self.compute_intraday_chart_flow(pool_syms)
        sector_snap = self.fetch_sectoral_indices_snapshot()
        macro_cues = self.fetch_global_macro_cues()
        sec_macro_map = macro_cues.get("sector_macro_adjustments", {})

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
            row_dict["macro_sector_adj"] = float(sec_macro_map.get(sec_idx, 0.0))

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