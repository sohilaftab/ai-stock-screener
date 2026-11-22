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

        # Rate-limit to max ~3 requests/sec to respect NSE servers
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
    Hybrid NSE Direct + Yahoo Finance Market Scanner providing:
    1. Zero-lag NSE live quotes + daily circuit price bands (lowerCP / upperCP)
    2. Security-wise Delivery Percentage (institutional accumulation vs intraday churn)
    3. Stock Option Chain Max Call OI (resistance) & Max Put OI (support) walls
    4. FII / DII daily institutional cash & derivatives net positioning
    """

    def __init__(self, universe=None):
        self.universe = universe or DEFAULT_NIFTY_UNIVERSE

    @staticmethod
    def _clean_sym(symbol: str) -> str:
        return str(symbol).replace(".NS", "").replace(".BO", "").strip().upper()

    @classmethod
    def fetch_nse_equity_quote(cls, symbol: str) -> dict:
        """
        Fetches live quote, intraday VWAP, circuit bands (lowerCP, upperCP),
        and delivery % directly from NSE India APIs.
        """
        clean = cls._clean_sym(symbol)
        encoded = urllib.parse.quote(clean)
        quote_data = _NSE_CLIENT.get_json(f"/api/quote-equity?symbol={encoded}")
        if not quote_data or "priceInfo" not in quote_data:
            return {}

        p_info = quote_data.get("priceInfo", {})
        intra = p_info.get("intraDayHighLow", {})
        ltp = float(p_info.get("lastPrice", 0.0) or 0.0)
        if ltp <= 0:
            return {}

        lower_cp = float(p_info.get("lowerCP", 0.0) or 0.0)
        upper_cp = float(p_info.get("upperCP", 0.0) or 0.0)
        vwap = float(p_info.get("vwap", ltp) or ltp)
        d_high = float(intra.get("max", ltp) or ltp)
        d_low = float(intra.get("min", ltp) or ltp)
        p_change = float(p_info.get("pChange", 0.0) or 0.0)

        # Trade info & Delivery percentage
        trade_data = _NSE_CLIENT.get_json(f"/api/quote-equity?symbol={encoded}&section=trade_info") or {}
        dp_info = trade_data.get("securityWiseDP", {})
        delivery_pct = float(dp_info.get("deliveryToTradedQuantity", 0.0) or 0.0)

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
            "source": "NSE_DIRECT",
        }

    @classmethod
    def fetch_option_chain_walls(cls, symbol: str, spot_price: float) -> dict:
        """
        Reads NSE Stock Option Chain to locate the nearest institutional Call OI
        resistance wall above spot and Put OI support floor below spot.
        """
        clean = cls._clean_sym(symbol)
        encoded = urllib.parse.quote(clean)
        oc = _NSE_CLIENT.get_json(f"/api/option-chain-equities?symbol={encoded}")
        if not oc or "records" not in oc:
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
            strike = float(r.get("strikePrice", 0.0) or 0.0)
            ce_oi = float((r.get("CE") or {}).get("openInterest", 0.0) or 0.0)
            pe_oi = float((r.get("PE") or {}).get("openInterest", 0.0) or 0.0)
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
        """Combines NSE direct quote, circuit bands, delivery %, and F&O OI walls."""
        q = cls.fetch_nse_equity_quote(symbol)
        ltp = q.get("current_price", spot_price)
        walls = cls.fetch_option_chain_walls(symbol, ltp) if ltp > 0 else {}
        return {
            "delivery_pct": q.get("delivery_pct", 0.0),
            "lower_cp": q.get("lower_cp", 0.0),
            "upper_cp": q.get("upper_cp", 0.0),
            "max_call_oi_strike": walls.get("max_call_oi_strike", 0.0),
            "max_put_oi_strike": walls.get("max_put_oi_strike", 0.0),
            "pcr": walls.get("pcr", 1.0),
        }

    @staticmethod
    def fetch_fii_dii_activity() -> dict:
        """Fetches latest NSE FII/DII net institutional cash flow in ₹ Crores."""
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
    def fetch_live_intraday_snapshot(cls, symbols: list) -> dict:
        """
        Primary: Direct NSE India equity quote (includes circuit bands & delivery %).
        Fallback: Yahoo Finance 1m/5m intraday bars if NSE blocks cloud runner IP.
        """
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

        # Fallback to Yahoo Finance for any symbol where direct NSE quote was unavailable
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
                    "source": "YFINANCE_FALLBACK",
                }
            except Exception:
                continue

        return snapshots

    def fetch_premarket_data(self) -> pd.DataFrame:
        """Fetches NSE Pre-Open session movers when available."""
        raw = _NSE_CLIENT.get_json("/api/market-data-pre-open?key=NIFTY")
        if not raw or "data" not in raw:
            return pd.DataFrame()

        rows = []
        for item in raw.get("data", []):
            meta = item.get("metadata", {})
            detail = item.get("detail", {}).get("preOpenMarket", {})
            sym = meta.get("symbol")
            if not sym:
                continue
            total_buy = float(detail.get("totalBuyQuantity", 0) or 0)
            total_sell = float(detail.get("totalSellQuantity", 0) or 0)
            bs_ratio = round(total_buy / max(total_sell, 1.0), 2)
            rows.append({
                "symbol": sym,
                "iep": float(detail.get("IEP", 0) or 0),
                "pChange": float(meta.get("pChange", 0) or 0),
                "buy_sell_ratio": bs_ratio,
            })
        return pd.DataFrame(rows)

    def scan_volume_breakouts(self, top_n: int = 15) -> pd.DataFrame:
        """Scans Nifty universe for volume breakouts, RSI, ATR, VWAP, and NSE microstructure."""
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
                vol_ratio = cur_vol / max(avg_vol_10, 1.0)

                # 14-day ATR
                tr = pd.concat([
                    df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift(1)).abs(),
                    (df["Low"] - df["Close"].shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr = float(tr.tail(14).mean())

                # 14-day RSI
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
                    "vol_ratio": round(vol_ratio, 2),
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
        cand_df["rank_score"] = cand_df["vol_ratio"] * (1.0 + cand_df["change_pct"].abs() / 5.0)
        top_df = cand_df.sort_values("rank_score", ascending=False).head(top_n).copy()

        # Enrich top candidates with direct NSE Delivery %, Circuit Bands & Options OI Walls
        def _enrich_micro(row_dict):
            micro = self.fetch_nse_microstructure(row_dict["symbol"], row_dict["ltp"])
            row_dict.update(micro)
            return row_dict

        with ThreadPoolExecutor(max_workers=4) as pool:
            enriched_records = list(pool.map(_enrich_micro, top_df.to_dict(orient="records")))

        return pd.DataFrame(enriched_records)

    def get_top_movers(self, top_n: int = 15) -> pd.DataFrame:
        return self.scan_volume_breakouts(top_n=top_n)