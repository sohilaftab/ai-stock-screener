import os
import json
import random
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
from market_data import MarketDataScanner
from ml_context_model import ContextualMLPredictor
from telegram_notifier import TelegramNotifier
from state_store import atomic_write_json, safe_load_json
from lexicon import BULLISH_KEYWORDS, BEARISH_KEYWORDS
from calendar_utils import expiry_status
from exec_sim import resolve_barrier_outcome, worse_stop_fill, apply_slippage
from health import is_halted

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

IST = ZoneInfo("Asia/Kolkata")

WEIGHTS_FILE = "model_weights.json"
LEDGER_FILE = "trade_history.json"
SHADOW_LEDGER_FILE = "shadow_ledger.json"
PORTFOLIO_FILE = "portfolio_state.json"
MORNING_WATCHLIST_FILE = "morning_watchlist.json"
SECTOR_CACHE_FILE = "sector_cache.json"

INITIAL_CAPITAL = 150000.0
MAX_POSITIONS_PER_SECTOR = 2
MAX_SECTOR_EXPOSURE_PCT = 0.40

# ---- Risk & learning controls (edit here) -------------------------------------------
EARLIEST_ENTRY_MIN = 9 * 60 + 30          # opening range (first 30 min) must be formed first
LATEST_ENTRY_MIN = 14 * 60 + 30           # no new entries after 14:30 IST (square-off is 15:15)
MAX_ENTRIES_PER_DAY = 8
MAX_ENTRIES_PER_CYCLE = 2                 # stagger entries instead of opening 5 at once
MAX_SAME_SIDE_POSITIONS = 3               # cap one-directional (all-long / all-short) exposure
DAILY_LOSS_LIMIT_PCT = 1.5                # stop NEW entries once the day is down this much
MAX_CONSECUTIVE_LOSSES_PER_DAY = 3
MIN_TRADES_FOR_WEIGHT_LEARNING = 20       # parameters stay frozen below this many closed trades
RECALIBRATE_EVERY_N_TRADES = 5            # intraday recalibration cadence (EOD still recalibrates)
WEIGHT_SHRINKAGE_K = 30.0                 # learned adjustment is scaled by n / (n + K)
MIN_TRADES_FOR_TICKER_MEMORY = 3
REGIME_CACHE_TTL_SEC = 120
# --------------------------------------------------------------------------------------


class AdaptiveTradeEngine:
    def __init__(self):
        self.notifier = TelegramNotifier()
        self.ml_model = ContextualMLPredictor()
        self.weights = self._load_weights()
        self.ledger = self._load_ledger()
        self.shadow_ledger = self._load_shadow_ledger()
        self.portfolio = self._load_portfolio()
        self.sector_cache = self._load_sector_cache()
        self._enforce_warmup_priors()

    def _enforce_warmup_priors(self):
        """
        Until MIN_TRADES_FOR_WEIGHT_LEARNING closed trades exist, every learned parameter is held at its prior.
        This also neutralises values left in model_weights.json by the pre-fix code (e.g. a -6 short bias or a
        0.72 stop multiplier learned from 5 trades) so they cannot leak into the new engine.
        """
        n = len([t for t in self.ledger if t.get("status") != "OPEN" and "factor_scores" in t])
        if n >= MIN_TRADES_FOR_WEIGHT_LEARNING:
            return
        self.weights.update({
            "volume_weight": 0.28, "sentiment_weight": 0.28, "premarket_weight": 0.22, "momentum_weight": 0.22,
            "atr_sl_multiplier": 0.55, "atr_tp_multiplier": 1.60, "max_profit_giveback_pct": 45.0,
            "learned_max_hold_mins": 180.0, "min_conviction_threshold": 52.0, "buy_rsi_ceiling": 72.0,
            "risk_per_trade_pct": 1.0, "buy_bias_bonus": 0.0, "short_bias_bonus": 0.0,
            "blacklisted_tickers": [], "preferred_tickers": [],
        })

    @staticmethod
    def _now_ist():
        return datetime.now(IST)

    @staticmethod
    def _get_session_bucket(dt=None):
        dt = dt or datetime.now(IST)
        mins = dt.hour * 60 + dt.minute
        if mins < 9 * 60 + 45:
            return "OPENING_SURGE (09:15-09:45)"
        elif mins < 11 * 60 + 30:
            return "MORNING_TREND (09:45-11:30)"
        elif mins < 13 * 60 + 30:
            return "MIDDAY_CHOP (11:30-13:30)"
        elif mins < 15 * 60:
            return "AFTERNOON_MOVE (13:30-15:00)"
        else:
            return "CLOSING_SESSION (15:00-15:30)"

    def _calculate_hold_minutes(self, trade):
        now_ist = self._now_ist()
        if trade.get("entry_iso"):
            try:
                entry_dt = datetime.fromisoformat(trade["entry_iso"])
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=IST)
                return max(0.0, (now_ist - entry_dt).total_seconds() / 60.0)
            except Exception:
                pass
        try:
            dt_str = f"{trade.get('date')} {trade.get('timestamp')}"
            entry_dt = datetime.strptime(dt_str, "%Y-%m-%d %I:%M:%S %p").replace(tzinfo=IST)
            return max(0.0, (now_ist - entry_dt).total_seconds() / 60.0)
        except Exception:
            return 0.0

    @staticmethod
    def _get_archetype_slippage_pct(archetype: str) -> float:
        if archetype == "LARGE_CAP_STEADY":
            return 0.03
        if archetype == "HIGH_VOL_MIDCAP":
            return 0.14
        return 0.07

    @staticmethod
    def _calculate_nse_intraday_charges(side, qty, entry_price, exit_price):
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

    @staticmethod
    def _classify_stock_archetype(ltp, atr):
        atr_pct = (atr / ltp) * 100.0 if ltp > 0 else 1.5
        if atr_pct < 1.8:
            return "LARGE_CAP_STEADY"
        elif atr_pct <= 3.5:
            return "MOMENTUM_MOVER"
        return "HIGH_VOL_MIDCAP"

    @staticmethod
    def _compute_clv(ltp, d_high, d_low):
        span = float(d_high) - float(d_low)
        if span <= 1e-6:
            return 0.0
        clv = ((float(ltp) - float(d_low)) - (float(d_high) - float(ltp))) / span
        return round(max(-1.0, min(1.0, clv)), 3)

    @staticmethod
    def _adversarial_bear_critic(
        side, ltp, atr, chg_pct, rsi, clv, intraday_alpha, archetype, regime_name,
        mtf_15m_trend="NEUTRAL", vwap_slope_pct=0.0, obi_ratio=0.0,
        bid_ask_spread_pct=0.0, sector_chg_pct=0.0, macro_regime="NEUTRAL",
        macro_sector_adj=0.0
    ):
        red_flags = []
        atr_pct = (atr / ltp) * 100.0 if ltp > 0 else 1.5

        if side == "BUY":
            if atr_pct > 0 and chg_pct > (2.1 * atr_pct):
                red_flags.append("Chasing exhausted >2.1x ATR spike")
            if clv < -0.35:
                red_flags.append(f"Bearish upper-wick rejection (CLV {clv:+.2f})")
            if rsi > 74.0:
                red_flags.append(f"Overbought RSI ({rsi:.1f})")
            if intraday_alpha < -0.6:
                red_flags.append(f"Lagging Nifty 50 ({intraday_alpha:+.2f}% alpha)")
            if mtf_15m_trend == "BEARISH" and vwap_slope_pct < -0.08:
                red_flags.append(f"15m MTF Bearish & falling VWAP ({vwap_slope_pct:+.2f}%)")
            if obi_ratio < -0.40:
                red_flags.append(f"Heavy NSE ask wall (OBI {obi_ratio:+.2f})")
            if sector_chg_pct < -1.0:
                red_flags.append(f"Fighting falling sector ({sector_chg_pct:+.2f}%)")
            if macro_regime == "INFLATION_RATE_SHOCK" and macro_sector_adj <= -3.0:
                red_flags.append("Global crude/yield shock opposing rate-sensitive BUY")
        else:
            if atr_pct > 0 and chg_pct < -(2.1 * atr_pct):
                red_flags.append("Chasing exhausted >2.1x ATR drop")
            if clv > 0.35:
                red_flags.append(f"Bullish lower-wick bounce (CLV {clv:+.2f})")
            if rsi < 26.0:
                red_flags.append(f"Oversold RSI ({rsi:.1f})")
            if intraday_alpha > 0.6:
                red_flags.append(f"Outperforming Nifty 50 ({intraday_alpha:+.2f}% alpha)")
            if mtf_15m_trend == "BULLISH" and vwap_slope_pct > 0.08:
                red_flags.append(f"15m MTF Bullish & rising VWAP ({vwap_slope_pct:+.2f}%)")
            if obi_ratio > 0.40:
                red_flags.append(f"Heavy NSE bid wall (OBI {obi_ratio:+.2f})")
            if sector_chg_pct > 1.0:
                red_flags.append(f"Shorting strong sector ({sector_chg_pct:+.2f}%)")
            if macro_regime == "RISK_ON" and macro_sector_adj >= 3.0:
                red_flags.append("Global Risk-On tailwind opposing SHORT")

        if bid_ask_spread_pct > 0.35:
            red_flags.append(f"Illiquid spread ({bid_ask_spread_pct:.2f}%)")
        if archetype == "HIGH_VOL_MIDCAP" and regime_name in ("SIDEWAYS_CHOP", "HIGH_VOLATILITY_PANIC"):
            red_flags.append(f"High-vol midcap during {regime_name}")

        vetoed = len(red_flags) >= 2
        penalty = round(len(red_flags) * 4.5, 1)
        return vetoed, penalty, red_flags

    def _gemini_bull_bear_debate(self, candidate: dict, regime_info: dict) -> dict:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or not HAS_GENAI:
            return {"veto": False, "conviction_adj": 0.0, "debate_note": "Quantitative Bear Critic passed"}

        macro = regime_info.get("global_macro", {})
        prompt = (
            "You are an NSE cash-intraday risk committee running a Bull vs. Bear debate "
            "before executing a same-day cash intraday equity trade (squared off before close). Analyze strictly in JSON:\n"
            f"- Stock: {candidate['symbol']} | Action: {candidate['action']} | LTP: ₹{candidate['entry_price']}\n"
            f"- Nifty Regime: {regime_info.get('regime')} (Nifty {regime_info.get('nifty_chg_pct')}%, "
            f"VIX {regime_info.get('vix_ltp')}, FII Net ₹{regime_info.get('fii_net_cr', 0)} Cr)\n"
            f"- Global Macro: {macro.get('summary', 'Neutral')} | Sector Bias={candidate.get('macro_sector_adj', 0):+.1f}\n"
            f"- 5m/15m Chart Flow: 15m Trend={candidate.get('mtf_15m_trend')}, VWAP Slope={candidate.get('vwap_slope_pct')}%, "
            f"ORB={candidate.get('orb_signal')}, Time-Norm RVOL={candidate.get('rvol_normalized')}x\n"
            f"- Sector Alignment: {candidate.get('sector_index')} ({candidate.get('sector_chg_pct'):+.2f}%), "
            f"Sector Alpha={candidate.get('sector_alpha'):+.2f}%\n"
            f"- Microstructure: Delivery={candidate.get('delivery_pct', 0)}%, OBI={candidate.get('obi_ratio', 0):+.2f}, "
            f"Spread={candidate.get('bid_ask_spread_pct', 0):.2f}%, CLV={candidate.get('clv', 0)}, RSI={candidate.get('rsi', 50)}\n"
            f"- News Headline: {candidate.get('headline', '')}\n\n"
            "Return ONLY valid JSON with keys: "
            '{"veto": boolean, "conviction_adj": number (-12 to +8), "bear_risk": "short 1-line summary"}'
        )

        client = genai.Client(api_key=api_key)
        for model_name in ("gemini-2.5-flash", "gemini-3.5-flash-lite"):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                parsed = json.loads(resp.text.strip())
                veto = bool(parsed.get("veto", False))
                adj = round(max(-12.0, min(8.0, float(parsed.get("conviction_adj", 0.0)))), 1)
                note = str(parsed.get("bear_risk", "LLM debate completed"))[:120]
                return {"veto": veto, "conviction_adj": adj, "debate_note": note}
            except Exception:
                continue

        return {"veto": False, "conviction_adj": 0.0, "debate_note": "Quantitative Bear Critic passed"}

    def _check_page_hinkley_drift(self):
        settled = [t for t in self.ledger if t.get("status") != "OPEN" and "pnl_inr" in t]
        if len(settled) < 8:
            return False

        recent = settled[-12:]
        errors = [1.0 if float(t.get("pnl_inr", 0.0)) <= 0 else 0.0 for t in recent]
        mean_err = sum(errors) / len(errors)

        ph_sum, min_ph, delta, threshold = 0.0, 0.0, 0.15, 2.6
        for e in errors:
            ph_sum += (e - mean_err - delta)
            min_ph = min(min_ph, ph_sum)

        if (ph_sum - min_ph) > threshold and mean_err >= 0.65:
            for arm_stats in self.weights.get("bandit_arms", {}).values():
                arm_stats["alpha"] = round(max(2.0, float(arm_stats.get("alpha", 2.0)) * 0.5), 2)
                arm_stats["beta"] = round(max(2.0, float(arm_stats.get("beta", 2.0)) * 0.5), 2)

            self.weights["drift_alert"] = (
                f"Concept drift detected on {self._now_ist().strftime('%Y-%m-%d %I:%M %p IST')} "
                f"(rolling loss rate {mean_err:.0%}); reset Bandit priors for rapid adaptation."
            )
            return True
        return False

    _REGIME_CACHE = {"ts": 0.0, "data": None}

    def _detect_market_regime(self):
        """Cached wrapper (a cycle used to hit Yahoo + NSE for the same index data several times)."""
        cache = AdaptiveTradeEngine._REGIME_CACHE
        if cache["data"] is not None and (time.time() - cache["ts"]) < REGIME_CACHE_TTL_SEC:
            self.weights["last_market_regime"] = cache["data"]
            return cache["data"]
        info = self._detect_market_regime_uncached()
        if float(info.get("nifty_ltp", 0.0) or 0.0) > 0:
            cache["ts"], cache["data"] = time.time(), info
        return info

    def _detect_market_regime_uncached(self):
        now_ist = self._now_ist()
        expiry_info = self._expiry_today()
        macro_cues = MarketDataScanner.fetch_global_macro_cues()
        macro_buy_adj = float(macro_cues.get("macro_buy_adj", 0.0))
        macro_short_adj = float(macro_cues.get("macro_short_adj", 0.0))
        macro_size_mult = float(macro_cues.get("macro_size_multiplier", 1.0))

        default_regime = {
            "regime": "NEUTRAL", "nifty_ltp": 0.0, "nifty_chg_pct": 0.0,
            "vix_ltp": 14.0, "vix_chg_pct": 0.0, "fii_net_cr": 0.0, "dii_net_cr": 0.0,
            "buy_adj": macro_buy_adj, "short_adj": macro_short_adj,
            "size_multiplier": round(macro_size_mult, 2), "sl_multiplier_adj": 1.0,
            "global_macro": macro_cues, "updated_at": now_ist.strftime("%I:%M %p IST"),
            "expiry_status": expiry_info,
        }
        try:
            nifty_df = yf.Ticker("^NSEI").history(period="5d", interval="15m")
            vix_df = yf.Ticker("^INDIAVIX").history(period="5d", interval="1d")
            fii_dii = MarketDataScanner.fetch_fii_dii_activity()
            fii_net = float(fii_dii.get("fii_net_cr", 0.0))
            dii_net = float(fii_dii.get("dii_net_cr", 0.0))
            fii_age = fii_dii.get("age_days")
            if fii_age is not None and fii_age > 4:      # stale NSE print: do not trade on it
                fii_net, dii_net = 0.0, 0.0

            if nifty_df.empty:
                self.weights["last_market_regime"] = default_regime
                return default_regime

            if nifty_df.index.tz is None:
                nifty_df.index = nifty_df.index.tz_localize(IST)
            else:
                nifty_df.index = nifty_df.index.tz_convert(IST)

            today_date = nifty_df.index[-1].date()
            today_bars = nifty_df[nifty_df.index.date == today_date]
            prev_bars = nifty_df[nifty_df.index.date < today_date]

            nifty_ltp = float(nifty_df["Close"].iloc[-1])
            prev_close = float(prev_bars["Close"].iloc[-1]) if not prev_bars.empty else float(today_bars["Open"].iloc[0])
            nifty_chg_pct = ((nifty_ltp - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
            intraday_mid = float(((today_bars["High"] + today_bars["Low"] + today_bars["Close"]) / 3.0).mean()) if not today_bars.empty else nifty_ltp
            ema_20 = float(nifty_df["Close"].tail(20).mean())

            vix_ltp, vix_chg_pct = 14.0, 0.0
            if not vix_df.empty:
                vix_ltp = float(vix_df["Close"].iloc[-1])
                if len(vix_df) >= 2:
                    prev_vix = float(vix_df["Close"].iloc[-2])
                    vix_chg_pct = ((vix_ltp - prev_vix) / prev_vix) * 100.0 if prev_vix > 0 else 0.0

            fii_bias = 3.0 if fii_net > 1500 else (-3.0 if fii_net < -1500 else 0.0)

            if vix_ltp >= 18.5 or vix_chg_pct >= 8.0:
                regime = "HIGH_VOLATILITY_PANIC"
                buy_adj, short_adj = -8.0 + fii_bias + macro_buy_adj, -3.0 - fii_bias + macro_short_adj
                base_size_mult, sl_adj = 0.50, 1.25
            elif nifty_chg_pct <= -0.45 and nifty_ltp < intraday_mid and nifty_ltp < ema_20:
                regime = "BEAR_TREND"
                buy_adj, short_adj = -8.0 + fii_bias + macro_buy_adj, 6.0 - fii_bias + macro_short_adj
                base_size_mult, sl_adj = 0.85, 1.05
            elif nifty_chg_pct >= 0.40 and nifty_ltp > intraday_mid and nifty_ltp > ema_20:
                regime = "BULL_TREND"
                buy_adj, short_adj = 6.0 + fii_bias + macro_buy_adj, -6.0 - fii_bias + macro_short_adj
                base_size_mult, sl_adj = 1.0, 1.0
            else:
                regime = "SIDEWAYS_CHOP"
                buy_adj, short_adj = -2.0 + fii_bias + macro_buy_adj, -2.0 - fii_bias + macro_short_adj
                base_size_mult, sl_adj = 0.80, 0.95

            info = {
                "regime": regime, "nifty_ltp": round(nifty_ltp, 2), "nifty_chg_pct": round(nifty_chg_pct, 2),
                "vix_ltp": round(vix_ltp, 2), "vix_chg_pct": round(vix_chg_pct, 2),
                "fii_net_cr": fii_net, "dii_net_cr": dii_net,
                "buy_adj": round(buy_adj, 1), "short_adj": round(short_adj, 1),
                "size_multiplier": round(max(0.35, min(1.0, base_size_mult * macro_size_mult)), 2),
                "sl_multiplier_adj": sl_adj, "global_macro": macro_cues,
                "updated_at": now_ist.strftime("%I:%M %p IST"),
                "expiry_status": expiry_info, "fii_data_age_days": fii_age,
            }
            self.weights["last_market_regime"] = info
            return info
        except Exception as e:
            print(f"[WARN] Market regime detection fallback: {e}")
            self.weights["last_market_regime"] = default_regime
            return default_regime

    def _batch_compute_mansfield_rs(self, symbols):
        rs_map = {s: 0.0 for s in symbols}
        if not symbols:
            return rs_map
        try:
            ns_map = {s: (s if s.endswith(".NS") else f"{s}.NS") for s in symbols}
            tickers = ["^NSEI"] + list(set(ns_map.values()))
            raw = yf.download(tickers=tickers, period="1mo", interval="1d", group_by="ticker", threads=True, progress=False)
            if raw.empty or not isinstance(raw.columns, pd.MultiIndex) or "^NSEI" not in raw.columns.get_level_values(0):
                return rs_map
            nifty_close = raw["^NSEI"]["Close"].dropna()
            if len(nifty_close) < 10:
                return rs_map
            for orig_sym, ns_sym in ns_map.items():
                try:
                    if ns_sym not in raw.columns.get_level_values(0):
                        continue
                    stk_close = raw[ns_sym]["Close"].dropna()
                    aligned = pd.concat([stk_close, nifty_close], axis=1, join="inner").dropna()
                    aligned.columns = ["stk", "nifty"]
                    if len(aligned) >= 10:
                        rs_series = aligned["stk"] / aligned["nifty"]
                        sma_10 = float(rs_series.tail(10).mean())
                        if sma_10 > 0:
                            rs_map[orig_sym] = round(((float(rs_series.iloc[-1]) / sma_10) - 1.0) * 100.0, 2)
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] Mansfield RS batch computation fallback: {e}")
        return rs_map

    def _default_bandit_arms(self):
        return {
            "MOMENTUM_BREAKOUT": {"alpha": 2.0, "beta": 2.0, "trades": 0, "wins": 0, "net_pnl_inr": 0.0},
            "NEWS_CATALYST": {"alpha": 2.0, "beta": 2.0, "trades": 0, "wins": 0, "net_pnl_inr": 0.0},
            "PREMARKET_GAP": {"alpha": 2.0, "beta": 2.0, "trades": 0, "wins": 0, "net_pnl_inr": 0.0},
            "RELATIVE_STRENGTH_LEADER": {"alpha": 2.0, "beta": 2.0, "trades": 0, "wins": 0, "net_pnl_inr": 0.0},
        }

    @staticmethod
    def _classify_strategy_arm(vol_score, sent_score, pre_score, mom_score, mansfield_rs, side):
        rs_aligned = (side == "BUY" and mansfield_rs >= 1.2) or (side.startswith("SELL") and mansfield_rs <= -1.2)
        if rs_aligned and abs(mansfield_rs) >= 1.5:
            return "RELATIVE_STRENGTH_LEADER"
        pillar_scores = {
            "NEWS_CATALYST": sent_score,
            "PREMARKET_GAP": pre_score,
            "MOMENTUM_BREAKOUT": max(vol_score, mom_score),
        }
        return max(pillar_scores.items(), key=lambda x: x[1])[0]

    def _sample_bandit_arm(self, arm_name):
        arms = self.weights.setdefault("bandit_arms", self._default_bandit_arms())
        arm = arms.setdefault(arm_name, {"alpha": 2.0, "beta": 2.0, "trades": 0, "wins": 0, "net_pnl_inr": 0.0})
        alpha, beta = max(0.5, float(arm.get("alpha", 2.0))), max(0.5, float(arm.get("beta", 2.0)))
        sampled_theta = random.betavariate(alpha, beta)
        posterior_prob = alpha / (alpha + beta)
        bandit_adj = round(max(-8.0, min(10.0, (sampled_theta - 0.50) * 22.0)), 2)
        return sampled_theta, posterior_prob, bandit_adj

    def _update_bandit_arm(self, arm_name, pnl_inr, risk_inr, weight_scale=1.0, count_as_real=True):
        arms = self.weights.setdefault("bandit_arms", self._default_bandit_arms())
        arm = arms.setdefault(arm_name, {"alpha": 2.0, "beta": 2.0, "trades": 0, "wins": 0, "net_pnl_inr": 0.0})
        alpha, beta = float(arm.get("alpha", 2.0)), float(arm.get("beta", 2.0))
        r_mag = min(2.0, max(0.4, abs(pnl_inr) / max(risk_inr, 100.0))) * float(weight_scale)

        if pnl_inr > 0:
            alpha += r_mag
            if count_as_real:
                arm["wins"] = int(arm.get("wins", 0)) + 1
        else:
            beta += r_mag

        if (alpha + beta) > 60.0:
            scale = 50.0 / (alpha + beta)
            alpha, beta = max(1.0, alpha * scale), max(1.0, beta * scale)

        arm["alpha"], arm["beta"] = round(alpha, 3), round(beta, 3)
        if count_as_real:
            arm["trades"] = int(arm.get("trades", 0)) + 1
            arm["net_pnl_inr"] = round(float(arm.get("net_pnl_inr", 0.0)) + pnl_inr, 2)

    def _compute_half_kelly_fraction(self, win_prob, fallback_rr):
        settled = [t for t in self.ledger if t.get("status") != "OPEN" and float(t.get("pnl_inr", 0.0)) != 0.0]
        wins = [float(t["pnl_inr"]) for t in settled if float(t["pnl_inr"]) > 0]
        losses = [abs(float(t["pnl_inr"])) for t in settled if float(t["pnl_inr"]) < 0]

        if len(wins) >= 3 and len(losses) >= 3:
            avg_win = sum(wins) / len(wins)
            avg_loss = max(1.0, sum(losses) / len(losses))
            payoff_b = max(0.6, min(4.0, avg_win / avg_loss))
        else:
            payoff_b = max(1.0, min(3.5, float(fallback_rr)))

        p = max(0.35, min(0.85, float(win_prob)))
        full_kelly = p - ((1.0 - p) / payoff_b)
        half_kelly = 0.5 * max(0.0, full_kelly)
        return round(max(0.08, min(0.25, half_kelly)), 4), round(payoff_b, 2)

    def _load_shadow_ledger(self):
        return safe_load_json(SHADOW_LEDGER_FILE, [], list, critical=False)

    def _save_shadow_ledger(self):
        try:
            atomic_write_json(SHADOW_LEDGER_FILE, self.shadow_ledger[-150:])
        except Exception as e:
            print(f"[WARN] Could not save shadow ledger: {e}")

    def _record_shadow_candidates(self, skipped_candidates):
        if not skipped_candidates:
            return
        now_ist = self._now_ist()
        today_str = now_ist.strftime("%Y-%m-%d")
        session_bucket = self._get_session_bucket(now_ist)
        existing_today = {s["symbol"] for s in self.shadow_ledger if s.get("date") == today_str}
        today_count = len(existing_today)

        added = 0
        for item in skipped_candidates:
            if today_count + added >= 15:
                break
            sym = item["symbol"]
            if sym in existing_today or float(item.get("conviction", 0)) < 40.0:
                continue

            entry_iso = now_ist.isoformat()
            shadow_id = item.get("shadow_id") or f"SHADOW_{today_str}_{sym}_{entry_iso}"

            rec = {
                "shadow_id": shadow_id, "date": today_str, "entry_iso": entry_iso,
                "entry_session": session_bucket, "symbol": sym, "action": item["action"],
                "is_expiry_day": bool(item.get("is_expiry_day", False)),
                "strategy_arm": item.get("strategy_arm", "MOMENTUM_BREAKOUT"),
                "archetype": item.get("archetype", "MOMENTUM_MOVER"),
                "rvol_normalized": float(item.get("rvol_normalized", 1.0)),
                "vol_accel_5m": float(item.get("vol_accel_5m", 1.0)),
                "vwap_slope_pct": float(item.get("vwap_slope_pct", 0.0)),
                "mtf_15m_trend": item.get("mtf_15m_trend", "NEUTRAL"),
                "orb_signal": item.get("orb_signal", "INSIDE_ORB"),
                "clv": float(item.get("clv", 0.0)), "rsi": float(item.get("rsi", 50.0)),
                "mansfield_rs": float(item.get("mansfield_rs", 0.0)),
                "sector_chg_pct": float(item.get("sector_chg_pct", 0.0)),
                "obi_ratio": float(item.get("obi_ratio", 0.0)),
                "bid_ask_spread_pct": float(item.get("bid_ask_spread_pct", 0.0)),
                "delivery_pct": float(item.get("delivery_pct", 0.0)),
                "vix_ltp": float(item.get("vix_ltp", 14.0)), "fii_net_cr": float(item.get("fii_net_cr", 0.0)),
                "sector_index": item.get("sector_index", "^NSEI"),
                "sector_alpha": float(item.get("sector_alpha", 0.0)),
                "macro_regime": item.get("macro_regime", "NEUTRAL"),
                "macro_sector_adj": float(item.get("macro_sector_adj", 0.0)),
                "brent_chg_pct": float(item.get("brent_chg_pct", 0.0)),
                "us10y_bps_chg": float(item.get("us10y_bps_chg", 0.0)),
                "usdinr_chg_pct": float(item.get("usdinr_chg_pct", 0.0)),
                "nasdaq_chg_pct": float(item.get("nasdaq_chg_pct", 0.0)),
                "entry_price": float(item["entry_price"]), "target_1": float(item["target_1"]),
                "stop_loss": float(item["stop_loss"]), "conviction": float(item["conviction"]),
                "skip_reason": item.get("skip_reason", "Below conviction threshold"),
                "status": "SHADOW_OPEN", "shadow_pnl_pct": 0.0
            }
            self.shadow_ledger.append(rec)
            self.ml_model.log_setup_observation(shadow_id, "SHADOW_TRADE", rec)
            existing_today.add(sym)
            added += 1

        if added > 0:
            self._save_shadow_ledger()

    def _evaluate_shadow_trades(self):
        """
        Labels shadow (skipped) setups using the 5-minute bars AFTER the hypothetical entry,
        walking bar by bar so stop-vs-target ordering is respected, and charging real costs.
        (Previously: whole-day high/low, which included price action before the entry.)
        """
        now_ist = self._now_ist()
        mins_now = now_ist.hour * 60 + now_ist.minute
        near_close = mins_now >= (15 * 60 + 10)
        today = now_ist.strftime("%Y-%m-%d")

        due = [
            s for s in self.shadow_ledger
            if s.get("status") == "SHADOW_OPEN"
            and (self._calculate_hold_minutes(s) >= 75.0 or near_close or s.get("date") != today)
        ]
        if not due:
            return 0

        items = []
        for s in due:
            try:
                items.append((s["shadow_id"], s["symbol"], self._next_bar_start_iso(s["entry_iso"])))
            except Exception:
                continue
        bars_map = MarketDataScanner.fetch_bars_since(items)
        evaluated = 0

        for s in due:
            try:
                entry_dt = datetime.fromisoformat(str(s["entry_iso"]))
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=IST)
                horizon_end = entry_dt + timedelta(minutes=75)
                bars = [
                    b for b in (bars_map.get(s["shadow_id"]) or [])
                    if b["date"] == s.get("date") and datetime.fromisoformat(b["ts"]) < horizon_end
                ]
            except Exception:
                bars = []

            if not bars:
                # Unknown is not "no touch": retry next cycle, and void very old ones so they never linger.
                try:
                    age_days = (now_ist.date() - datetime.strptime(s["date"], "%Y-%m-%d").date()).days
                except Exception:
                    age_days = 99
                if age_days >= 3:
                    s["status"] = "SHADOW_VOID"
                continue

            entry, t1, sl = float(s["entry_price"]), float(s["target_1"]), float(s["stop_loss"])
            side = s["action"]
            is_buy = str(side).upper().startswith("BUY")
            outcome = resolve_barrier_outcome(is_buy, bars, sl, t1)
            exit_px = outcome["exit_px"] if outcome["reason"] != "NONE" else float(bars[-1]["close"])

            slip = self._get_archetype_slippage_pct(s.get("archetype", "MOMENTUM_MOVER"))
            fill_in = apply_slippage(entry, is_buy, slip)
            fill_out = apply_slippage(exit_px, not is_buy, slip)
            pseudo_qty = max(1, int(20000.0 // max(entry, 1.0)))
            gross = (fill_out - fill_in) * pseudo_qty if is_buy else (fill_in - fill_out) * pseudo_qty
            charges = self._calculate_nse_intraday_charges("BUY" if is_buy else "SELL", pseudo_qty, fill_in, fill_out)
            pseudo_pnl_inr = round(gross - charges, 2)
            net_pnl_pct = round((pseudo_pnl_inr / max(pseudo_qty * entry, 1.0)) * 100.0, 3)

            s["status"] = "SHADOW_WIN" if pseudo_pnl_inr > 0 else "SHADOW_LOSS"
            s["shadow_pnl_pct"] = net_pnl_pct
            s["shadow_exit_reason"] = outcome["reason"]

            self._update_bandit_arm(s.get("strategy_arm", "MOMENTUM_BREAKOUT"), pseudo_pnl_inr, 300.0, weight_scale=0.35, count_as_real=False)
            self._update_archetype_memory(s.get("archetype", "MOMENTUM_MOVER"), pseudo_pnl_inr, hit_sl=(outcome["reason"] == "SL"), weight_scale=0.35)
            evaluated += 1

        if evaluated > 0:
            self.weights["shadow_samples_learned"] = int(self.weights.get("shadow_samples_learned", 0)) + evaluated
            self._save_weights()
        self._save_shadow_ledger()
        return evaluated

    def _update_archetype_memory(self, archetype, pnl_inr, hit_sl=False, weight_scale=1.0):
        arch_mem = self.weights.setdefault("archetype_memory", {})
        rec = arch_mem.setdefault(archetype, {
            "weighted_wins": 1.0, "weighted_losses": 1.0, "sl_hits": 0, "trades": 0, "sl_mult_adj": 1.0, "score_adj": 0.0
        })
        if pnl_inr > 0:
            rec["weighted_wins"] = round(float(rec["weighted_wins"]) + weight_scale, 2)
        else:
            rec["weighted_losses"] = round(float(rec["weighted_losses"]) + weight_scale, 2)
            if hit_sl:
                rec["sl_hits"] = int(rec.get("sl_hits", 0)) + 1
        if weight_scale >= 1.0:
            rec["trades"] = int(rec.get("trades", 0)) + 1

        total_w = float(rec["weighted_wins"]) + float(rec["weighted_losses"])
        win_prob = float(rec["weighted_wins"]) / max(total_w, 1.0)
        rec["score_adj"] = round(max(-8.0, min(8.0, (win_prob - 0.50) * 20.0)), 1)
        if rec["trades"] >= 3 and (rec["sl_hits"] / rec["trades"]) > 0.45:
            rec["sl_mult_adj"] = 1.20
        elif win_prob >= 0.60:
            rec["sl_mult_adj"] = 0.95

    def _load_sector_cache(self):
        return safe_load_json(SECTOR_CACHE_FILE, {}, dict, critical=False)

    def _save_sector_cache(self):
        try:
            atomic_write_json(SECTOR_CACHE_FILE, self.sector_cache)
        except Exception:
            pass

    def _get_stock_sector(self, sym):
        clean_sym = sym.replace(".NS", "").strip().upper()
        if clean_sym in self.sector_cache and self.sector_cache[clean_sym] != "Unknown":
            return self.sector_cache[clean_sym]
        try:
            info = yf.Ticker(f"{clean_sym}.NS").info or {}
            sector = info.get("sector") or info.get("industry") or "General"
            self.sector_cache[clean_sym] = sector
            self._save_sector_cache()
            return sector
        except Exception:
            return self.sector_cache.get(clean_sym, "General")

    def _get_open_sector_exposure(self):
        counts, capital = {}, {}
        for t in self.ledger:
            if t.get("status") != "OPEN":
                continue
            sec = t.get("sector") or self._get_stock_sector(t["symbol"])
            counts[sec] = counts.get(sec, 0) + 1
            capital[sec] = capital.get(sec, 0.0) + float(t.get("capital_used", 0.0))
        return counts, capital

    def _should_square_off_intraday(self, trade) -> tuple:
        now_ist = self._now_ist()
        mins = now_ist.hour * 60 + now_ist.minute
        if trade.get("date") != now_ist.strftime("%Y-%m-%d"):
            return True, "Prior-day intraday position squared off automatically."
        if mins >= (15 * 60 + 15):
            return True, "Mandatory intraday square-off window (15:15+ IST)."
        return False, ""

    def _analyze_live_news_emotion(self, sym):
        clean_sym = sym.replace(".NS", "").strip()
        query = urllib.parse.quote(f"{clean_sym} NSE stock news")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")[:5]
            if not items:
                return {"emotion": "NEUTRAL", "score": 0.0, "top_headline": "No breaking news"}

            bull_hits, bear_hits, headlines = 0, 0, []
            for item in items:
                title = (item.findtext("title") or "").strip()
                headlines.append(title)
                words = set(w.strip(".,:;!?()[]\"'").lower() for w in title.split())
                bull_hits += len(words & BULLISH_KEYWORDS)
                bear_hits += len(words & BEARISH_KEYWORDS)

            net = bull_hits - bear_hits
            emotion = "POSITIVE" if net >= 2 else ("NEGATIVE" if net <= -2 else "NEUTRAL")
            return {"emotion": emotion, "score": round(float(net), 2), "top_headline": headlines[0], "all_headlines": headlines[:3]}
        except Exception:
            return {"emotion": "NEUTRAL", "score": 0.0, "top_headline": "News feed unavailable", "all_headlines": []}

    def _expiry_today(self, dt=None):
        d = (dt or self._now_ist()).date()
        return expiry_status(d, MarketDataScanner.is_nse_trading_day)

    @staticmethod
    def _next_bar_start_iso(iso_str):
        """Start of the first 5-minute bar strictly AFTER the timestamp (the entry bar is skipped,
        so price action before the fill can never count as a stop/target touch)."""
        dt = datetime.fromisoformat(str(iso_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        dt = dt.astimezone(IST)
        floored = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
        return (floored + timedelta(minutes=5)).isoformat()

    def entries_allowed(self, now_ist=None):
        """Portfolio-level circuit breakers. Returns (allowed: bool, reason: str)."""
        now_ist = now_ist or self._now_ist()
        mins = now_ist.hour * 60 + now_ist.minute
        if is_halted():
            return False, "Kill switch active (HALT_TRADING file / PULSEAI_HALT env)"
        if mins < EARLIEST_ENTRY_MIN:
            return False, "Before 09:30 IST - opening range still forming"
        if mins >= LATEST_ENTRY_MIN:
            return False, "Past the 14:30 IST new-entry cutoff"

        today = now_ist.strftime("%Y-%m-%d")
        todays = [t for t in self.ledger if t.get("date") == today]
        if len(todays) >= MAX_ENTRIES_PER_DAY:
            return False, f"Daily entry cap reached ({MAX_ENTRIES_PER_DAY})"

        closed = [t for t in todays if t.get("status") != "OPEN"]
        open_t = [t for t in todays if t.get("status") == "OPEN"]
        day_pnl = sum(float(t.get("pnl_inr", 0.0) or 0.0) for t in closed) + \
            sum(float(t.get("unrealized_pnl_inr", 0.0) or 0.0) for t in open_t)
        equity = float(self.portfolio.get("current_equity", INITIAL_CAPITAL) or INITIAL_CAPITAL)
        if day_pnl <= -(equity * DAILY_LOSS_LIMIT_PCT / 100.0):
            return False, f"Daily loss limit hit ({day_pnl:,.0f} INR <= -{DAILY_LOSS_LIMIT_PCT}% of equity)"

        n = MAX_CONSECUTIVE_LOSSES_PER_DAY
        if len(closed) >= n and all(float(t.get("pnl_inr", 0.0) or 0.0) <= 0 for t in closed[-n:]):
            return False, f"{n} consecutive losses today - standing down"
        return True, ""

    def _load_weights(self):
        default_weights = {
            "volume_weight": 0.28, "sentiment_weight": 0.28, "premarket_weight": 0.22, "momentum_weight": 0.22,
            "risk_per_trade_pct": 1.0, "min_conviction_threshold": 52.0,
            "atr_sl_multiplier": 0.55, "atr_tp_multiplier": 1.60,
            "buy_rsi_ceiling": 72.0, "short_rsi_floor": 28.0,
            "buy_bias_bonus": 0.0, "short_bias_bonus": 0.0,
            "max_profit_giveback_pct": 45.0, "learned_max_hold_mins": 180.0,
            "bandit_arms": self._default_bandit_arms(), "archetype_memory": {},
            "shadow_samples_learned": 0, "ml_model_summary": "Collecting walk-forward samples",
            "session_memory": {}, "ticker_memory": {}, "blacklisted_tickers": [], "preferred_tickers": [],
            "trade_lessons_log": [], "drift_alert": "",
            "last_market_regime": {"regime": "NEUTRAL", "nifty_chg_pct": 0.0, "vix_ltp": 14.0, "fii_net_cr": 0.0},
            "generation": 1, "last_evaluated_count": 0,
            "last_updated": self._now_ist().strftime("%Y-%m-%d %H:%M"),
            "learning_notes": "Initialized Strict Cash Intraday PulseAI."
        }
        loaded = safe_load_json(WEIGHTS_FILE, {}, dict, critical=False)
        if loaded:
            default_weights.update(loaded)
            if "bandit_arms" not in loaded:
                default_weights["bandit_arms"] = self._default_bandit_arms()
        return default_weights

    def _save_weights(self):
        atomic_write_json(WEIGHTS_FILE, self.weights)

    def _load_ledger(self):
        # critical: an unreadable ledger must STOP the run, never silently become [] and get overwritten
        return safe_load_json(LEDGER_FILE, [], list, critical=True)

    def _save_ledger(self):
        atomic_write_json(LEDGER_FILE, self.ledger)

    def _load_portfolio(self):
        today_str = self._now_ist().strftime("%Y-%m-%d")
        default_portfolio = {
            "initial_capital": INITIAL_CAPITAL, "manual_deposits": 0.0,
            "total_capital_contributed": INITIAL_CAPITAL, "deposit_history": [],
            "current_equity": INITIAL_CAPITAL, "available_cash": INITIAL_CAPITAL,
            "allocated_capital": 0.0, "total_realized_pnl": 0.0, "total_taxes_paid": 0.0,
            "win_streak": 0, "loss_streak": 0, "mistakes_identified": [],
            "daily_equity_log": [{"date": today_str, "equity": INITIAL_CAPITAL, "pnl": 0.0}]
        }
        default_portfolio.update(safe_load_json(PORTFOLIO_FILE, {}, dict, critical=True))
        self._recalculate_cash(default_portfolio)
        return default_portfolio

    def _recalculate_cash(self, port_dict):
        deposits = port_dict.get("deposit_history", [])
        manual_total = round(sum(float(d.get("amount", 0.0)) for d in deposits), 2)
        total_contributed = round(INITIAL_CAPITAL + manual_total, 2)

        port_dict["initial_capital"] = INITIAL_CAPITAL
        port_dict["manual_deposits"] = manual_total
        port_dict["total_capital_contributed"] = total_contributed

        open_allocated = sum(float(t.get("capital_used", 0.0)) for t in self.ledger if t.get("status") == "OPEN")
        realized = sum(float(t.get("pnl_inr", 0.0)) for t in self.ledger if t.get("status") != "OPEN")
        taxes_paid = sum(float(t.get("charges_inr", 0.0)) for t in self.ledger if t.get("status") != "OPEN")

        port_dict["total_realized_pnl"] = round(realized, 2)
        port_dict["total_taxes_paid"] = round(taxes_paid, 2)
        port_dict["current_equity"] = round(max(0.0, total_contributed + realized), 2)
        port_dict["allocated_capital"] = round(open_allocated, 2)
        port_dict["available_cash"] = round(max(0.0, port_dict["current_equity"] - open_allocated), 2)

    def _save_portfolio(self):
        self._recalculate_cash(self.portfolio)
        atomic_write_json(PORTFOLIO_FILE, self.portfolio)

    def add_manual_funds(self, amount, note="Manual user deposit", notify_telegram=True):
        amt = round(float(amount), 2)
        if amt <= 0:
            raise ValueError("Deposit amount must be greater than ₹0.")
        now_str = self._now_ist().strftime("%Y-%m-%d %I:%M:%S %p IST")
        self.portfolio.setdefault("deposit_history", []).append({
            "timestamp": now_str, "amount": amt, "note": str(note).strip() or "Manual user deposit"
        })
        self._save_portfolio()
        if notify_telegram:
            self.notifier.send_message(
                f"💵 <b>MANUAL FUNDS ADDED BY USER</b>\n━━━━━━━━━━━━━━━━━━━━\n"
                f"➕ <b>Amount Added:</b> +₹{amt:,.2f}\n📝 <b>Note:</b> {note}\n"
                f"🏦 <b>New Total Equity:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>\n"
                f"💰 <b>Available Cash to Trade:</b> <b>₹{self.portfolio['available_cash']:,.2f}</b>"
            )
        return self.portfolio

    def save_morning_watchlist(self, rec_df):
        records = rec_df.to_dict(orient="records") if not rec_df.empty else []
        atomic_write_json(MORNING_WATCHLIST_FILE, {"date": self._now_ist().strftime("%Y-%m-%d"), "items": records})

    def compare_with_morning_watchlist(self, new_rec_df):
        if not os.path.exists(MORNING_WATCHLIST_FILE) or new_rec_df.empty:
            return []
        try:
            with open(MORNING_WATCHLIST_FILE, "r") as f:
                saved = json.load(f)
            old_map = {item["symbol"]: item for item in saved.get("items", [])}
        except Exception:
            return []

        changes = []
        new_symbols = set(new_rec_df["symbol"].tolist())
        for _, row in new_rec_df.iterrows():
            sym = row["symbol"]
            if sym not in old_map:
                changes.append(f"🆕 <b>NEW SETUP DETECTED: {sym}</b> ({row['action']} @ ₹{row['entry_price']}, Conv {row['conviction']}/100).")
            else:
                old_item = old_map[sym]
                if old_item["action"] != row["action"]:
                    changes.append(f"🔄 <b>DIRECTION FLIP: {sym}</b> from {old_item['action']} to <b>{row['action']}</b>!")
                elif abs(float(row["entry_price"]) - float(old_item["entry_price"])) / float(old_item["entry_price"]) > 0.01:
                    changes.append(f"⚡ <b>LEVEL UPDATE: {sym}</b> Entry ₹{row['entry_price']} | Target ₹{row['target_1']} | SL ₹{row['stop_loss']}.")
        for old_sym in old_map:
            if old_sym not in new_symbols:
                changes.append(f"❌ <b>DROPPED: {old_sym}</b> removed.")
        return changes

    def generate_intraday_recommendations(self, enriched_df, premarket_df=None, top_k=5):
        if enriched_df.empty:
            return pd.DataFrame()

        mins_now = self._now_ist().hour * 60 + self._now_ist().minute
        expiry_flag = bool(self._expiry_today()["is_expiry_day"])
        if mins_now >= LATEST_ENTRY_MIN:
            return pd.DataFrame()

        self._recalculate_cash(self.portfolio)
        current_equity = float(self.portfolio["current_equity"])
        avail_cash = float(self.portfolio["available_cash"])

        regime_info = self._detect_market_regime()
        regime_name = regime_info["regime"]
        nifty_chg_pct = float(regime_info.get("nifty_chg_pct", 0.0))
        vix_ltp = float(regime_info.get("vix_ltp", 14.0))
        fii_net_cr = float(regime_info.get("fii_net_cr", 0.0))
        regime_buy_adj = float(regime_info["buy_adj"])
        regime_short_adj = float(regime_info["short_adj"])
        regime_size_mult = float(regime_info["size_multiplier"])
        regime_sl_adj = float(regime_info["sl_multiplier_adj"])

        macro_cues = regime_info.get("global_macro", {})
        macro_regime = macro_cues.get("macro_regime", "NEUTRAL")
        sec_macro_map = macro_cues.get("sector_macro_adjustments", {})

        current_session = self._get_session_bucket()
        session_mem = self.weights.get("session_memory", {}).get(current_session, {})
        session_adj = float(session_mem.get("score_adj", 0.0))
        session_blocked = session_mem.get("trades", 0) >= 4 and session_mem.get("win_rate", 50.0) < 30.0

        candidate_syms = enriched_df["symbol"].tolist()
        mansfield_rs_map = self._batch_compute_mansfield_rs(candidate_syms)
        sector_snap = MarketDataScanner.fetch_sectoral_indices_snapshot()

        intraday_flow_fallback = {}
        if "rvol_normalized" not in enriched_df.columns or "orb_signal" not in enriched_df.columns:
            intraday_flow_fallback = MarketDataScanner.compute_intraday_chart_flow(candidate_syms)

        pre_map = {}
        if premarket_df is not None and not premarket_df.empty:
            for _, r in premarket_df.iterrows():
                pre_map[r["symbol"]] = {
                    "iep": float(r.get("iep", 0)),
                    "pChange": float(r.get("pChange", 0)),
                    "buy_sell_ratio": float(r.get("buy_sell_ratio", 1.0))
                }

        w_vol = float(self.weights["volume_weight"])
        w_sent = float(self.weights["sentiment_weight"])
        w_pre = float(self.weights["premarket_weight"])
        w_mom = float(self.weights["momentum_weight"])

        base_atr_sl_mult = float(self.weights.get("atr_sl_multiplier", 0.55)) * regime_sl_adj
        atr_tp_mult = float(self.weights.get("atr_tp_multiplier", 1.60))
        buy_rsi_max = float(self.weights.get("buy_rsi_ceiling", 72.0))
        short_rsi_min = float(self.weights.get("short_rsi_floor", 28.0))
        min_conviction = float(self.weights.get("min_conviction_threshold", 52.0))
        buy_bonus = float(self.weights.get("buy_bias_bonus", 0.0))
        short_bonus = float(self.weights.get("short_bias_bonus", 0.0))

        ticker_memory = self.weights.get("ticker_memory", {})
        archetype_memory = self.weights.get("archetype_memory", {})
        blacklist = set(self.weights.get("blacklisted_tickers", []))

        scored_candidates, skipped_for_shadow = [], []

        for _, row in enriched_df.iterrows():
            sym = row["symbol"]
            if sym in blacklist and len(enriched_df) > 3:
                continue

            flow = intraday_flow_fallback.get(sym, {})
            ltp = float(flow.get("ltp_intraday", row["ltp"]))
            chg = float(row["change_pct"])
            rvol_norm = float(row.get("rvol_normalized", flow.get("rvol_normalized", row["vol_ratio"])))
            vol_accel_5m = float(row.get("vol_accel_5m", flow.get("vol_accel_5m", 1.0)))
            atr = float(row.get("atr", ltp * 0.015))
            rsi = float(row.get("rsi", 50.0))
            vwap = float(row.get("vwap_proxy", flow.get("vwap_intraday", ltp)))
            vwap_slope_pct = float(row.get("vwap_slope_pct", flow.get("vwap_slope_pct", 0.0)))
            mtf_15m_trend = str(row.get("mtf_15m_trend", flow.get("mtf_15m_trend", "NEUTRAL")))
            orb_signal = str(row.get("orb_signal", flow.get("orb_signal", "INSIDE_ORB")))
            orb_high = float(row.get("orb_high", flow.get("orb_high", ltp)))
            orb_low = float(row.get("orb_low", flow.get("orb_low", ltp)))
            d_high = float(row.get("day_high", flow.get("day_high_intraday", ltp)))
            d_low = float(row.get("day_low", flow.get("day_low_intraday", ltp)))
            chart_trend = row.get("chart_trend", "RANGE_CONSOLIDATION")

            sec_idx = str(row.get("sector_index", "") or MarketDataScanner.get_sector_index_for_stock(sym))
            sec_chg_pct = float(row.get("sector_chg_pct", sector_snap.get(sec_idx, {}).get("change_pct", nifty_chg_pct)))
            sector_alpha = round(chg - sec_chg_pct, 2)
            raw_macro_sec_adj = float(row.get("macro_sector_adj", sec_macro_map.get(sec_idx, 0.0)))

            delivery_pct = float(row.get("delivery_pct", 0.0) or 0.0)
            obi_ratio = float(row.get("obi_ratio", 0.0) or 0.0)
            bid_ask_spread_pct = float(row.get("bid_ask_spread_pct", 0.0) or 0.0)
            lower_cp = float(row.get("lower_cp", 0.0) or 0.0)
            upper_cp = float(row.get("upper_cp", 0.0) or 0.0)

            if sym in pre_map and pre_map[sym]["iep"] > 50:
                chg = pre_map[sym]["pChange"]

            raw_sent = row.get("raw_sentiment", "NEUTRAL")
            ai_conf = float(row.get("AI Confidence", 50)) / 100.0
            headline = row.get("Why Is It Moving? (Latest News)", "Technical chart & volume setup")

            if (chg >= 0 and raw_sent != "NEGATIVE") or (mtf_15m_trend == "BULLISH" and orb_signal == "ORB_BREAKOUT_UP"):
                side = "BUY"
            elif (chg < 0 and raw_sent != "POSITIVE") or (mtf_15m_trend == "BEARISH" and orb_signal == "ORB_BREAKDOWN_DOWN"):
                side = "SELL (SHORT)"
            else:
                side = "BUY" if chg >= 0 else "SELL (SHORT)"

            if side == "BUY" and upper_cp > 0 and ltp >= (upper_cp * 0.985):
                continue
            if side.startswith("SELL") and lower_cp > 0 and ltp <= (lower_cp * 1.015):
                continue

            archetype = self._classify_stock_archetype(ltp, atr)
            arch_stats = archetype_memory.get(archetype, {})
            arch_score_adj = float(arch_stats.get("score_adj", 0.0))
            arch_sl_adj = float(arch_stats.get("sl_mult_adj", 1.0))
            clv = self._compute_clv(ltp, d_high, d_low)

            mansfield_rs = float(mansfield_rs_map.get(sym, 0.0))
            intraday_alpha = round(chg - nifty_chg_pct, 2)

            if side == "BUY" and raw_sent == "POSITIVE" and ltp < vwap and intraday_alpha < -0.5:
                continue

            vetoed, critic_penalty, red_flags = self._adversarial_bear_critic(
                side, ltp, atr, chg, rsi, clv, intraday_alpha, archetype, regime_name,
                mtf_15m_trend=mtf_15m_trend, vwap_slope_pct=vwap_slope_pct, obi_ratio=obi_ratio,
                bid_ask_spread_pct=bid_ask_spread_pct, sector_chg_pct=sec_chg_pct,
                macro_regime=macro_regime, macro_sector_adj=raw_macro_sec_adj,
            )

            deliv_mult = 1.20 if delivery_pct >= 50.0 else (0.75 if (0 < delivery_pct < 22.0) else 1.0)
            accel_bonus = 8.0 if vol_accel_5m >= 1.5 else 0.0
            vol_score = min(100.0, ((rvol_norm / 2.4) * 100.0 * deliv_mult) + accel_bonus)

            if (side == "BUY" and raw_sent == "POSITIVE") or (side.startswith("SELL") and raw_sent == "NEGATIVE"):
                sent_score = 65.0 + (ai_conf * 35.0)
            elif raw_sent == "NEUTRAL":
                sent_score = 52.0
            else:
                sent_score = 20.0

            if sym in pre_map:
                pm = pre_map[sym]
                if side == "BUY" and pm["pChange"] > 0 and pm["buy_sell_ratio"] >= 1.1:
                    pre_score = min(100.0, 65.0 + pm["buy_sell_ratio"] * 15.0)
                elif side.startswith("SELL") and pm["pChange"] < 0 and pm["buy_sell_ratio"] <= 0.9:
                    pre_score = 88.0
                else:
                    pre_score = 45.0
            else:
                pre_score = 55.0

            if side == "BUY":
                trend_bonus = 18.0 if chart_trend == "BULLISH_UPTREND" else 0.0
                mtf_bonus = 14.0 if mtf_15m_trend == "BULLISH" else (-12.0 if mtf_15m_trend == "BEARISH" else 0.0)
                orb_bonus = 12.0 if orb_signal == "ORB_BREAKOUT_UP" else (-8.0 if orb_signal == "ORB_BREAKDOWN_DOWN" else 0.0)
                vwap_bonus = 10.0 if (ltp >= vwap and vwap_slope_pct >= 0.0) else (-10.0 if ltp < vwap else 2.0)
                clv_bonus = 6.0 * clv
                rsi_bonus = 10.0 if (50.0 <= rsi <= buy_rsi_max) else (-14.0 if rsi > buy_rsi_max else 0.0)
                mom_score = max(10.0, min(100.0, 40.0 + trend_bonus + mtf_bonus + orb_bonus + vwap_bonus + rsi_bonus + clv_bonus))
                rs_adj = max(-6.0, min(8.0, (mansfield_rs * 1.4) + (intraday_alpha * 1.0)))
                sector_adj = max(-7.0, min(7.0, (sec_chg_pct * 2.5) + (sector_alpha * 1.2)))
                obi_adj = max(-5.0, min(5.0, obi_ratio * 8.0))
                macro_sec_applied = max(-6.0, min(6.0, raw_macro_sec_adj))
            else:
                trend_bonus = 18.0 if chart_trend == "BEARISH_DOWNTREND" else 0.0
                mtf_bonus = 14.0 if mtf_15m_trend == "BEARISH" else (-12.0 if mtf_15m_trend == "BULLISH" else 0.0)
                orb_bonus = 12.0 if orb_signal == "ORB_BREAKDOWN_DOWN" else (-8.0 if orb_signal == "ORB_BREAKOUT_UP" else 0.0)
                vwap_bonus = 10.0 if (ltp <= vwap and vwap_slope_pct <= 0.0) else (-10.0 if ltp > vwap else 2.0)
                clv_bonus = -6.0 * clv
                rsi_bonus = 10.0 if (short_rsi_min <= rsi <= 50.0) else (-14.0 if rsi < short_rsi_min else 0.0)
                mom_score = max(10.0, min(100.0, 40.0 + trend_bonus + mtf_bonus + orb_bonus + vwap_bonus + rsi_bonus + clv_bonus))
                rs_adj = max(-6.0, min(8.0, (-mansfield_rs * 1.4) + (-intraday_alpha * 1.0)))
                sector_adj = max(-7.0, min(7.0, (-sec_chg_pct * 2.5) + (-sector_alpha * 1.2)))
                obi_adj = max(-5.0, min(5.0, -obi_ratio * 8.0))
                macro_sec_applied = max(-6.0, min(6.0, -raw_macro_sec_adj))

            base_conviction = (w_vol * vol_score) + (w_sent * sent_score) + (w_pre * pre_score) + (w_mom * mom_score)
            strategy_arm = self._classify_strategy_arm(vol_score, sent_score, pre_score, mom_score, mansfield_rs, side)
            _, posterior_prob, bandit_adj = self._sample_bandit_arm(strategy_arm)

            ticker_adj = 0.0
            if sym in ticker_memory:
                tm = ticker_memory[sym]
                trades_n, win_r, net_pnl = tm.get("trades", 0), tm.get("win_rate", 50.0), tm.get("net_pnl_inr", 0.0)
                if trades_n >= MIN_TRADES_FOR_TICKER_MEMORY:
                    if win_r >= 60.0 and net_pnl > 0:
                        ticker_adj = min(8.0, 3.0 + (trades_n * 1.5))
                    elif win_r < 40.0 and net_pnl < 0:
                        ticker_adj = -min(15.0, 5.0 + (trades_n * 3.0))

            chop_adj = -8.0 if (regime_name == "SIDEWAYS_CHOP" and abs(chg) > 3.5) else 0.0
            dir_adj = (buy_bonus + regime_buy_adj) if side == "BUY" else (short_bonus + regime_short_adj)

            conviction = round(
                max(1.0, min(99.0, base_conviction + ticker_adj + dir_adj + session_adj + chop_adj + rs_adj + sector_adj + obi_adj + macro_sec_applied + bandit_adj + arch_score_adj - critic_penalty)),
                1
            )

            effective_sl_mult = base_atr_sl_mult * arch_sl_adj
            intraday_risk_per_share = max(round(atr * effective_sl_mult, 2), round(ltp * 0.006, 2))
            reward_1_per_share = round(intraday_risk_per_share * (atr_tp_mult / max(effective_sl_mult, 0.3)), 2)
            reward_2_per_share = round(intraday_risk_per_share * ((atr_tp_mult + 1.0) / max(effective_sl_mult, 0.3)), 2)

            entry_price = round(ltp, 2)
            if side == "BUY":
                stop_loss = round(entry_price - intraday_risk_per_share, 2)
                target_1 = round(entry_price + reward_1_per_share, 2)
                target_2 = round(entry_price + reward_2_per_share, 2)
                if upper_cp > 0 and target_1 > upper_cp:
                    target_1 = upper_cp
            else:
                stop_loss = round(entry_price + intraday_risk_per_share, 2)
                target_1 = round(entry_price - reward_1_per_share, 2)
                target_2 = round(entry_price - reward_2_per_share, 2)
                if lower_cp > 0 and target_1 < lower_cp:
                    target_1 = lower_cp

            rr_val = round(abs(target_1 - entry_price) / max(intraday_risk_per_share, 0.01), 2)
            calibrated_p = round(0.5 * posterior_prob + 0.5 * (conviction / 100.0), 4)

            cand_obj = {
                "is_expiry_day": expiry_flag,
                "symbol": sym, "action": side, "conviction": conviction, "calibrated_win_prob": calibrated_p,
                "strategy_arm": strategy_arm, "archetype": archetype, "entry_session": current_session,
                "clv": clv, "rvol_normalized": round(rvol_norm, 2), "vol_accel_5m": round(vol_accel_5m, 2),
                "vwap_slope_pct": round(vwap_slope_pct, 3), "mtf_15m_trend": mtf_15m_trend,
                "orb_signal": orb_signal, "orb_high": orb_high, "orb_low": orb_low,
                "obi_ratio": obi_ratio, "bid_ask_spread_pct": bid_ask_spread_pct,
                "sector_index": sec_idx, "sector_chg_pct": sec_chg_pct, "sector_alpha": sector_alpha,
                "vix_ltp": vix_ltp, "fii_net_cr": fii_net_cr, "macro_regime": macro_regime,
                "macro_sector_adj": round(macro_sec_applied, 1),
                "brent_chg_pct": float(macro_cues.get("brent_chg_pct", 0.0)),
                "us10y_bps_chg": float(macro_cues.get("us10y_bps_chg", 0.0)),
                "usdinr_chg_pct": float(macro_cues.get("usdinr_chg_pct", 0.0)),
                "nasdaq_chg_pct": float(macro_cues.get("nasdaq_chg_pct", 0.0)),
                "delivery_pct": delivery_pct, "lower_cp": lower_cp, "upper_cp": upper_cp,
                "mansfield_rs": mansfield_rs, "intraday_alpha": intraday_alpha,
                "entry_price": entry_price, "target_1": target_1, "target_2": target_2, "stop_loss": stop_loss,
                "intraday_risk_per_share": intraday_risk_per_share, "rr_val": rr_val, "rr_ratio": f"1 : {rr_val}",
                "chart_trend": chart_trend, "vol_ratio": round(rvol_norm, 2), "rsi": rsi,
                "sentiment": raw_sent, "headline": headline,
                "vol_score": round(vol_score, 1), "sent_score": round(sent_score, 1),
                "pre_score": round(pre_score, 1), "mom_score": round(mom_score, 1),
                "debate_note": "Quantitative Bear Critic passed",
            }

            ml_res = self.ml_model.predict_setup_edge(cand_obj)
            cand_obj["ml_prob"] = ml_res["ml_prob"]
            cand_obj["ml_conv_adj"] = ml_res["ml_conv_adj"]
            cand_obj["ml_status"] = ml_res["ml_status"]
            if ml_res["ml_active"]:
                cand_obj["conviction"] = round(max(1.0, min(99.0, cand_obj["conviction"] + ml_res["ml_conv_adj"])), 1)
                cand_obj["calibrated_win_prob"] = round(0.35 * posterior_prob + 0.35 * (cand_obj["conviction"] / 100.0) + 0.30 * ml_res["ml_prob"], 4)

            if vetoed:
                cand_obj["skip_reason"] = f"Bear Critic Veto: {', '.join(red_flags)}"
                skipped_for_shadow.append(cand_obj)
                continue
            if session_blocked or cand_obj["conviction"] < min_conviction or avail_cash < 500:
                cand_obj["skip_reason"] = "Low-win-rate session window" if session_blocked else f"Conviction {cand_obj['conviction']} < {min_conviction}"
                skipped_for_shadow.append(cand_obj)
                continue

            scored_candidates.append(cand_obj)

        self._record_shadow_candidates(skipped_for_shadow)
        if not scored_candidates:
            return pd.DataFrame()

        scored_candidates.sort(key=lambda x: x["conviction"], reverse=True)
        debate_approved = []
        for idx, item in enumerate(scored_candidates):
            if idx < 2:
                debate = self._gemini_bull_bear_debate(item, regime_info)
                item["debate_note"] = debate["debate_note"]
                if debate["veto"]:
                    item["skip_reason"] = f"Gemini Bear Debate Veto: {debate['debate_note']}"
                    skipped_for_shadow.append(item)
                    continue
                item["conviction"] = round(max(1.0, min(99.0, item["conviction"] + debate["conviction_adj"])), 1)
                if item["conviction"] < min_conviction:
                    item["skip_reason"] = f"Post-debate conviction {item['conviction']} < {min_conviction}"
                    skipped_for_shadow.append(item)
                    continue
            debate_approved.append(item)

        self._record_shadow_candidates(skipped_for_shadow)
        if not debate_approved:
            return pd.DataFrame()

        sector_counts, sector_capital = self._get_open_sector_exposure()
        max_sector_cap = current_equity * MAX_SECTOR_EXPOSURE_PCT
        filtered_by_sector = []
        for item in debate_approved:
            if len(filtered_by_sector) >= top_k:
                break
            sec = self._get_stock_sector(item["symbol"])
            if sector_counts.get(sec, 0) >= MAX_POSITIONS_PER_SECTOR or sector_capital.get(sec, 0.0) >= max_sector_cap:
                continue
            item["sector"] = sec
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            filtered_by_sector.append(item)

        if not filtered_by_sector:
            return pd.DataFrame()

        remaining_wallet_cash = avail_cash
        risk_pct_allowed = (float(self.weights.get("risk_per_trade_pct", 1.0)) / 100.0) * regime_size_mult
        max_risk_inr_per_trade = current_equity * risk_pct_allowed

        final_suggestions = []
        for item in filtered_by_sector:
            entry_price = item["entry_price"]
            risk_per_share = item["intraday_risk_per_share"]
            sec = item["sector"]
            if remaining_wallet_cash < entry_price:
                continue

            kelly_frac, _ = self._compute_half_kelly_fraction(item["calibrated_win_prob"], item["rr_val"])
            kelly_budget = current_equity * kelly_frac * regime_size_mult
            sec_headroom = max(0.0, max_sector_cap - sector_capital.get(sec, 0.0))
            slot_budget = min(remaining_wallet_cash, kelly_budget, sec_headroom)

            qty_by_kelly = int(slot_budget // entry_price)
            qty_by_risk = int(max_risk_inr_per_trade // max(risk_per_share, 0.5))
            qty = max(1, min(qty_by_kelly, max(qty_by_risk, int((slot_budget * 0.85) // entry_price))))
            if qty * entry_price > remaining_wallet_cash or qty * entry_price > sec_headroom:
                qty = int(min(remaining_wallet_cash, sec_headroom) // entry_price)
            if qty <= 0:
                continue

            capital_used = round(qty * entry_price, 2)
            remaining_wallet_cash = round(remaining_wallet_cash - capital_used, 2)
            sector_capital[sec] = sector_capital.get(sec, 0.0) + capital_used

            final_suggestions.append({
                "symbol": item["symbol"], "sector": sec, "sector_index": item["sector_index"],
                "sector_chg_pct": item["sector_chg_pct"], "sector_alpha": item["sector_alpha"],
                "market_regime": regime_name, "vix_ltp": item["vix_ltp"], "fii_net_cr": item["fii_net_cr"],
                "macro_regime": item["macro_regime"], "macro_sector_adj": item["macro_sector_adj"],
                "brent_chg_pct": item["brent_chg_pct"], "us10y_bps_chg": item["us10y_bps_chg"],
                "usdinr_chg_pct": item["usdinr_chg_pct"], "nasdaq_chg_pct": item["nasdaq_chg_pct"],
                "strategy_arm": item["strategy_arm"], "archetype": item["archetype"],
                "entry_session": item["entry_session"], "clv": item["clv"],
                "rvol_normalized": item["rvol_normalized"], "vol_accel_5m": item["vol_accel_5m"],
                "vwap_slope_pct": item["vwap_slope_pct"], "mtf_15m_trend": item["mtf_15m_trend"],
                "orb_signal": item["orb_signal"], "obi_ratio": item["obi_ratio"],
                "bid_ask_spread_pct": item["bid_ask_spread_pct"], "delivery_pct": item["delivery_pct"],
                "lower_cp": item["lower_cp"], "upper_cp": item["upper_cp"],
                "debate_note": item["debate_note"], "ml_prob": item["ml_prob"],
                "ml_conv_adj": item["ml_conv_adj"], "ml_status": item["ml_status"],
                "mansfield_rs": item["mansfield_rs"], "intraday_alpha": item["intraday_alpha"],
                "kelly_pct": round(kelly_frac * regime_size_mult * 100.0, 1),
                "action": item["action"], "conviction": item["conviction"],
                "qty": qty, "capital_used": capital_used,
                "max_loss_inr": round(qty * risk_per_share, 2),
                "expected_profit_inr": round(qty * abs(item["target_1"] - entry_price), 2),
                "entry_price": entry_price, "target_1": item["target_1"],
                "target_2": item["target_2"], "stop_loss": item["stop_loss"],
                "risk_pct": round((risk_per_share / entry_price) * 100, 2),
                "reward_pct": round((abs(item["target_1"] - entry_price) / entry_price) * 100, 2),
                "rr_ratio": item["rr_ratio"], "chart_trend": item["chart_trend"],
                "vol_ratio": item["vol_ratio"], "rsi": item["rsi"],
                "sentiment": item["sentiment"], "headline": item["headline"],
                "vol_score": item["vol_score"], "sent_score": item["sent_score"],
                "pre_score": item["pre_score"], "mom_score": item["mom_score"]
            })

        self._save_weights()
        return pd.DataFrame(final_suggestions)

    def record_recommendations(self, rec_df, notify_telegram=True):
        if rec_df.empty:
            return 0

        allowed, why = self.entries_allowed()
        if not allowed:
            print(f"[RISK GATE] No new entries: {why}")
            return 0

        self._recalculate_cash(self.portfolio)
        avail_cash = float(self.portfolio["available_cash"])
        now_ist = self._now_ist()
        today_str, now_time = now_ist.strftime("%Y-%m-%d"), now_ist.strftime("%I:%M:%S %p")
        session_bucket = self._get_session_bucket(now_ist)

        open_symbols = {item["symbol"] for item in self.ledger if item.get("status") == "OPEN"}
        sector_counts, _ = self._get_open_sector_exposure()
        fresh_ticks = MarketDataScanner.fetch_live_intraday_snapshot(rec_df["symbol"].tolist())

        is_expiry_today = bool(self._expiry_today()["is_expiry_day"])
        todays_count = sum(1 for t in self.ledger if t.get("date") == today_str)
        entry_budget = max(0, min(MAX_ENTRIES_PER_CYCLE, MAX_ENTRIES_PER_DAY - todays_count))
        side_counts = {"BUY": 0, "SELL": 0}
        for t in self.ledger:
            if t.get("status") == "OPEN":
                side_counts["BUY" if str(t.get("action", "")).upper().startswith("BUY") else "SELL"] += 1

        added = 0
        for _, row in rec_df.iterrows():
            if added >= entry_budget:
                break
            sym = row["symbol"]
            if sym in open_symbols:
                continue
            side_key = "BUY" if str(row["action"]).upper().startswith("BUY") else "SELL"
            if side_counts[side_key] >= MAX_SAME_SIDE_POSITIONS:
                continue
            sec = row.get("sector") or self._get_stock_sector(sym)
            if sector_counts.get(sec, 0) >= MAX_POSITIONS_PER_SECTOR:
                continue

            old_entry = float(row["entry_price"])
            snap = fresh_ticks.get(sym, {})
            live_tick_p = snap.get("current_price", old_entry) or old_entry
            price_shift = live_tick_p - old_entry
            slip_pct = self._get_archetype_slippage_pct(str(row.get("archetype", "MOMENTUM_MOVER")))
            entry_p = round(apply_slippage(live_tick_p, side_key == "BUY", slip_pct), 2)   # realistic fill

            if avail_cash < entry_p:
                continue
            qty = int(row["qty"])
            cap_needed = round(qty * entry_p, 2)
            if cap_needed > avail_cash:
                qty = int(avail_cash // entry_p)
                cap_needed = round(qty * entry_p, 2)
            if qty <= 0 or cap_needed <= 0 or cap_needed > avail_cash:
                continue

            avail_cash = round(avail_cash - cap_needed, 2)
            t1 = round(float(row["target_1"]) + price_shift, 2)
            t2 = round(float(row["target_2"]) + price_shift, 2)
            sl = round(float(row["stop_loss"]) + price_shift, 2)
            max_loss = round(abs(entry_p - sl) * qty, 2)
            exp_profit = round(abs(t1 - entry_p) * qty, 2)

            regime_tag = row.get("market_regime", self.weights.get("last_market_regime", {}).get("regime", "NEUTRAL"))
            macro_tag = str(row.get("macro_regime", "NEUTRAL"))
            macro_sec_adj = float(row.get("macro_sector_adj", 0.0) or 0.0)
            arm_tag = row.get("strategy_arm", "MOMENTUM_BREAKOUT")
            arch_tag = row.get("archetype", "MOMENTUM_MOVER")
            mrs_val = float(row.get("mansfield_rs", 0.0))
            kelly_pct = float(row.get("kelly_pct", 15.0))
            deliv_pct = float(row.get("delivery_pct", snap.get("delivery_pct", 0.0)) or 0.0)
            rvol_norm = float(row.get("rvol_normalized", row.get("vol_ratio", 1.0)) or 1.0)
            vol_accel = float(row.get("vol_accel_5m", 1.0) or 1.0)
            vwap_slope = float(row.get("vwap_slope_pct", 0.0) or 0.0)
            mtf_trend = str(row.get("mtf_15m_trend", "NEUTRAL"))
            orb_sig = str(row.get("orb_signal", "INSIDE_ORB"))
            obi_val = float(row.get("obi_ratio", snap.get("obi_ratio", 0.0)) or 0.0)
            spread_val = float(row.get("bid_ask_spread_pct", snap.get("bid_ask_spread_pct", 0.0)) or 0.0)
            sec_idx = str(row.get("sector_index", "^NSEI"))
            sec_chg = float(row.get("sector_chg_pct", 0.0) or 0.0)
            sec_alpha = float(row.get("sector_alpha", 0.0) or 0.0)
            lower_cp = float(row.get("lower_cp", snap.get("lower_cp", 0.0)) or 0.0)
            upper_cp = float(row.get("upper_cp", snap.get("upper_cp", 0.0)) or 0.0)
            debate_note = str(row.get("debate_note", "Passed risk debate"))
            ml_status = str(row.get("ml_status", "Collecting samples"))

            trade_record = {
                "trade_id": f"{today_str}_{sym}_{now_ist.strftime('%H%M')}",
                "date": today_str, "timestamp": now_time, "entry_iso": now_ist.isoformat(),
                "entry_session": session_bucket, "market_regime": regime_tag,
                "is_expiry_day": is_expiry_today, "entry_slippage_pct": slip_pct,
                "signal_price": round(float(live_tick_p), 2),
                "last_bar_check_iso": self._next_bar_start_iso(now_ist.isoformat()),
                "vix_ltp": float(row.get("vix_ltp", 14.0) or 14.0),
                "fii_net_cr": float(row.get("fii_net_cr", 0.0) or 0.0),
                "macro_regime": macro_tag, "macro_sector_adj": macro_sec_adj,
                "brent_chg_pct": float(row.get("brent_chg_pct", 0.0) or 0.0),
                "us10y_bps_chg": float(row.get("us10y_bps_chg", 0.0) or 0.0),
                "usdinr_chg_pct": float(row.get("usdinr_chg_pct", 0.0) or 0.0),
                "nasdaq_chg_pct": float(row.get("nasdaq_chg_pct", 0.0) or 0.0),
                "strategy_arm": arm_tag, "archetype": arch_tag, "mansfield_rs": mrs_val,
                "rvol_normalized": rvol_norm, "vol_accel_5m": vol_accel, "vwap_slope_pct": vwap_slope,
                "mtf_15m_trend": mtf_trend, "orb_signal": orb_sig, "clv": float(row.get("clv", 0.0) or 0.0),
                "obi_ratio": obi_val, "bid_ask_spread_pct": spread_val,
                "sector_index": sec_idx, "sector_chg_pct": sec_chg, "sector_alpha": sec_alpha,
                "delivery_pct": deliv_pct, "lower_cp": lower_cp, "upper_cp": upper_cp,
                "debate_note": debate_note, "ml_status": ml_status, "kelly_pct": kelly_pct,
                "sector": sec, "symbol": sym, "action": row["action"], "conviction": float(row["conviction"]),
                "qty": qty, "capital_used": cap_needed, "entry_price": entry_p,
                "initial_target": t1, "target_1": t1, "target_2": t2, "initial_sl": sl, "stop_loss": sl,
                "current_price": entry_p, "peak_pnl_inr": 0.0, "unrealized_pnl_inr": 0.0,
                "entry_rsi": float(row.get("rsi", 50.0)), "rsi": float(row.get("rsi", 50.0)),
                "entry_sentiment": row.get("sentiment", "NEUTRAL"),
                "chart_trend": row.get("chart_trend", "RANGE_CONSOLIDATION"),
                "status": "OPEN",
                "trade_management_note": f"15m:{mtf_trend} | {orb_sig} | RVOL:{rvol_norm:.1f}x | Macro:{macro_tag} ({macro_sec_adj:+.1f}) | ML:{ml_status}",
                "exit_price": None, "exit_time": None, "hold_minutes": 0.0,
                "day_high": None, "day_low": None, "gross_pnl_inr": 0.0, "charges_inr": 0.0,
                "pnl_pct": 0.0, "pnl_inr": 0.0, "mistake_diagnosis": None,
                "factor_scores": {
                    "volume": float(row["vol_score"]), "sentiment": float(row["sent_score"]),
                    "premarket": float(row["pre_score"]), "momentum": float(row["mom_score"])
                },
                "headline": row["headline"]
            }
            self.ledger.append(trade_record)
            self.ml_model.log_setup_observation(trade_record["trade_id"], "REAL_TRADE", trade_record)
            open_symbols.add(sym)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            side_counts[side_key] += 1
            added += 1

            if notify_telegram:
                side_header = "🟢 <b>INTRADAY BUY ENTRY</b>" if row["action"] == "BUY" else "🔴 <b>INTRADAY SHORT ENTRY</b>"
                self.notifier.send_message(
                    f"{side_header}\n━━━━━━━━━━━━━━━━━━━━\n"
                    f"📌 <b>Stock:</b> <code>{sym}</code> ({row['action']}) | <b>Sector:</b> {sec} ({sec_idx})\n"
                    f"🌍 <b>Macro:</b> <code>{macro_tag}</code> (Sec Adj {macro_sec_adj:+.1f})\n"
                    f"📈 <b>Chart Flow:</b> 15m <code>{mtf_trend}</code> | <code>{orb_sig}</code> | VWAP Slope {vwap_slope:+.2f}%\n"
                    f"📊 <b>RVOL:</b> {rvol_norm:.1f}x | <b>OBI:</b> {obi_val:+.2f} | <b>Deliv:</b> {deliv_pct:.1f}% | <b>Sec Alpha:</b> {sec_alpha:+.2f}%\n"
                    f"🎰 <b>Arm:</b> <code>{arm_tag}</code> | <b>TRA:</b> <code>{arch_tag}</code> | <b>ML:</b> <code>{ml_status}</code>\n"
                    f"📦 <b>Quantity:</b> <b>{qty} Shares</b> @ <b>₹{entry_p}</b> (Half-Kelly: {kelly_pct:.1f}%)\n"
                    f"💰 <b>Wallet Used:</b> ₹{cap_needed:,.2f} <i>(Cash Left: ₹{avail_cash:,.2f})</i>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>Target 1:</b> ₹{t1} <i>(+₹{exp_profit:,.0f})</i> | 🚀 <b>Target 2:</b> ₹{t2}\n"
                    f"🛑 <b>Initial Stop-Loss:</b> ₹{sl} <i>(-₹{max_loss:,.0f} risk)</i>\n"
                    f"⚖️ <b>Risk Debate:</b> <i>{debate_note}</i>\n"
                    f"📰 <b>Catalyst ({row['sentiment']}):</b> {row['headline']}"
                )

        if added > 0:
            self._save_ledger()
            self._save_portfolio()
        return added

    def manage_live_positions(self, notify_telegram=True):
        self._evaluate_shadow_trades()
        self._check_page_hinkley_drift()

        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        if not open_trades:
            return []

        regime_info = self._detect_market_regime()
        nifty_chg_pct = float(regime_info.get("nifty_chg_pct", 0.0))
        snapshots = MarketDataScanner.fetch_live_intraday_snapshot([t["symbol"] for t in open_trades])
        actions_taken = []
        now_ist = self._now_ist()
        today_str_live = now_ist.strftime("%Y-%m-%d")

        # 5-minute bars AFTER each entry: lets us catch a stop/target touched between polling cycles
        # and judge them in order, instead of using the whole day's high/low (which includes pre-entry prices).
        bar_items = []
        for t in open_trades:
            since = t.get("last_bar_check_iso") or (self._next_bar_start_iso(t["entry_iso"]) if t.get("entry_iso") else None)
            if since:
                bar_items.append((t.get("trade_id") or t["symbol"], t["symbol"], since))
        bars_map = MarketDataScanner.fetch_bars_since(bar_items) if bar_items else {}
        now_time = now_ist.strftime("%I:%M:%S %p")

        giveback_limit_pct = float(self.weights.get("max_profit_giveback_pct", 45.0))
        max_hold_mins = float(self.weights.get("learned_max_hold_mins", 180.0))

        for trade in open_trades:
            sym = trade["symbol"]
            snap = snapshots.get(sym)
            if not snap:
                continue

            ltp, d_high, d_low, vwap = snap["current_price"], snap["day_high"], snap["day_low"], snap["intraday_vwap"]
            lower_cp = float(snap.get("lower_cp", trade.get("lower_cp", 0.0)) or 0.0)
            upper_cp = float(snap.get("upper_cp", trade.get("upper_cp", 0.0)) or 0.0)

            entry, t1, t2, sl = float(trade["entry_price"]), float(trade["target_1"]), float(trade["target_2"]), float(trade["stop_loss"])
            init_sl = float(trade.get("initial_sl", sl))
            qty, side = int(trade.get("qty", 1)), trade["action"]
            arch_used = trade.get("archetype", "MOMENTUM_MOVER")

            risk_unit = max(abs(entry - init_sl), entry * 0.005)
            hold_mins = round(self._calculate_hold_minutes(trade), 1)
            trade["hold_minutes"], trade["current_price"], trade["day_high"], trade["day_low"] = hold_mins, ltp, d_high, d_low

            unrealized_per_share = (ltp - entry) if side == "BUY" else (entry - ltp)
            unrealized_inr = round(unrealized_per_share * qty, 2)
            trade["unrealized_pnl_inr"] = unrealized_inr
            r_multiple = unrealized_per_share / risk_unit

            peak_pnl = max(float(trade.get("peak_pnl_inr", 0.0)), unrealized_inr)
            trade["peak_pnl_inr"] = round(peak_pnl, 2)

            if side == "BUY" and lower_cp > 0 and ltp <= lower_cp:
                trade["trade_management_note"] = f"LOWER CIRCUIT LOCKED @ ₹{lower_cp}"
                continue
            if side.startswith("SELL") and upper_cp > 0 and ltp >= upper_cp:
                trade["trade_management_note"] = f"UPPER CIRCUIT LOCKED @ ₹{upper_cp}"
                continue

            stk_chg_pct = ((ltp - entry) / entry) * 100.0 if entry > 0 else 0.0
            live_alpha = stk_chg_pct - nifty_chg_pct

            live_news, emotion_flipped = None, False
            if unrealized_inr < 0 or (peak_pnl > 0 and unrealized_inr < peak_pnl * 0.6):
                live_news = self._analyze_live_news_emotion(sym)
                if (side == "BUY" and live_news["emotion"] == "NEGATIVE") or (side.startswith("SELL") and live_news["emotion"] == "POSITIVE"):
                    emotion_flipped = True

            force_eod_close, eod_reason = self._should_square_off_intraday(trade)

            is_buy_side = (side == "BUY")
            stale_trade = trade.get("date") != today_str_live
            t_bars = bars_map.get(trade.get("trade_id") or sym) or []
            day_bars = [b for b in t_bars if b["date"] == trade.get("date")]
            barrier = resolve_barrier_outcome(is_buy_side, day_bars, sl, t1) if day_bars else {"reason": "NONE", "exit_px": None}
            bar_hit = barrier["reason"]

            exit_triggered, exit_status, raw_exit_price, exit_note = False, "", ltp, ""

            if bar_hit == "SL":
                is_be = (sl >= entry) if is_buy_side else (sl <= entry)
                exit_triggered, exit_status, raw_exit_price = True, ("BREAKEVEN EXIT" if is_be else "LOSS (SL HIT)"), float(barrier["exit_px"])
                exit_note = "Stop touched on 5m bars after entry" + (" (gapped through the stop)." if barrier.get("gap") else ".")
            elif stale_trade and force_eod_close and bar_hit != "T1":
                stale_px = float(day_bars[-1]["close"]) if day_bars else ltp
                exit_triggered, exit_status, raw_exit_price = True, "INTRADAY SQUARE-OFF", stale_px
                exit_note = f"{eod_reason} (marked at the {trade.get('date')} session close, not today's price)"
            elif emotion_flipped:
                exit_triggered, exit_status, raw_exit_price = True, "EMOTION SHIFT EXIT", ltp
                exit_note = f"News emotion flipped to {live_news['emotion']}: {live_news['top_headline']}"
            elif force_eod_close:
                exit_triggered, exit_status, raw_exit_price, exit_note = True, "INTRADAY SQUARE-OFF", ltp, eod_reason
            elif (
                side == "BUY" and trade.get("entry_sentiment") == "POSITIVE"
                and hold_mins >= 20.0 and ltp < vwap and live_alpha < -0.8 and unrealized_inr < 0
            ):
                exit_triggered, exit_status, raw_exit_price = True, "DISTRIBUTION TRAP EXIT", ltp
                exit_note = f"Bullish news failed to hold VWAP (₹{vwap}) with {live_alpha:+.2f}% Nifty alpha."
            elif (side == "BUY" and (ltp >= t1 or bar_hit == "T1")) or (side.startswith("SELL") and (ltp <= t1 or bar_hit == "T1")):
                strong_momentum = ((ltp >= t1 and ltp > vwap and t1 < t2) if side == "BUY" else (ltp <= t1 and ltp < vwap and t1 > t2))
                if strong_momentum:
                    new_sl = round(entry + (0.85 * risk_unit), 2) if side == "BUY" else round(entry - (0.85 * risk_unit), 2)
                    trade["stop_loss"] = max(sl, new_sl) if side == "BUY" else min(sl, new_sl)
                    trade["target_1"] = t2
                    trade["trade_management_note"] = f"AI extended target to ₹{t2} & locked SL at ₹{trade['stop_loss']}."
                    actions_taken.append(f"🚀 <b>TARGET EXTENDED ({sym}):</b> New Target ₹{t2} | Locked SL ₹{trade['stop_loss']}")
                else:
                    exit_triggered, exit_status, raw_exit_price, exit_note = True, "WIN (TARGET HIT)", t1, "AI booked full target profit."
            elif (side == "BUY" and ltp <= sl) or (side.startswith("SELL") and ltp >= sl):
                is_be = (sl >= entry) if side == "BUY" else (sl <= entry)
                exit_triggered, exit_status, raw_exit_price, exit_note = True, ("BREAKEVEN EXIT" if is_be else "LOSS (SL HIT)"), worse_stop_fill(side == "BUY", sl, ltp), "Exited at dynamic Stop-Loss (filled at stop or worse)."
            elif peak_pnl >= (risk_unit * qty * 0.75) and unrealized_inr > 0:
                giveback_pct = ((peak_pnl - unrealized_inr) / peak_pnl) * 100.0
                if giveback_pct >= giveback_limit_pct:
                    exit_triggered, exit_status, raw_exit_price = True, "WIN (PROFIT LOCKED)", ltp
                    exit_note = f"AI locked profit after {giveback_pct:.0f}% pullback from peak ₹{peak_pnl:,.0f}."
            elif r_multiple <= -0.65 and ((side == "BUY" and ltp < vwap) or (side.startswith("SELL") and ltp > vwap)):
                exit_triggered, exit_status, raw_exit_price, exit_note = True, "EARLY CUT (LOSS MINIMIZED)", ltp, f"Cut early against VWAP (₹{vwap})."
            elif hold_mins >= max_hold_mins and unrealized_inr <= 0:
                exit_triggered, exit_status, raw_exit_price, exit_note = True, "STAGNATION EXIT (AI TIMING)", ltp, f"Exited after {hold_mins:.0f}m without follow-through."
            elif r_multiple >= 0.80 and ((side == "BUY" and sl < entry) or (side.startswith("SELL") and sl > entry)):
                trade["stop_loss"] = entry
                trade["trade_management_note"] = f"SL trailed to breakeven ₹{entry}."

            if not exit_triggered:
                # next cycle only needs bars from here on (skip the partial bar, avoid re-judging with moved levels)
                trade["last_bar_check_iso"] = self._next_bar_start_iso(now_ist.isoformat())

            if exit_triggered:
                slip_pct = self._get_archetype_slippage_pct(arch_used) / 100.0
                exit_price = round(raw_exit_price * (1.0 - slip_pct), 2) if side == "BUY" else round(raw_exit_price * (1.0 + slip_pct), 2)
                gross_pnl_inr = ((exit_price - entry) * qty) if side == "BUY" else ((entry - exit_price) * qty)
                charges_inr = self._calculate_nse_intraday_charges(side, qty, entry, exit_price)
                net_pnl_inr = round(gross_pnl_inr - charges_inr, 2)
                cap_used = max(float(trade.get("capital_used", entry * qty)), 1.0)
                net_pnl_pct = round((net_pnl_inr / cap_used) * 100.0, 2)

                trade.update({
                    "status": exit_status, "exit_price": exit_price, "exit_time": f"{now_time} IST",
                    "unrealized_pnl_inr": 0.0, "gross_pnl_inr": round(gross_pnl_inr, 2),
                    "charges_inr": charges_inr, "pnl_pct": net_pnl_pct, "pnl_inr": net_pnl_inr,
                    "trade_management_note": f"{exit_note} (After ₹{charges_inr:.2f} taxes & slippage)"
                })

                arm_used = trade.get("strategy_arm", "MOMENTUM_BREAKOUT")
                self._update_bandit_arm(arm_used, net_pnl_inr, risk_unit * qty, weight_scale=1.0, count_as_real=True)
                self._update_archetype_memory(arch_used, net_pnl_inr, hit_sl=("SL HIT" in exit_status), weight_scale=1.0)
                self._recalculate_cash(self.portfolio)
                lesson = self._research_and_learn_from_trade(trade, live_news)

                actions_taken.append(f"{'✅' if net_pnl_inr >= 0 else '🛑'} <b>AI CLOSED {sym} ({exit_status}):</b> ₹{exit_price} | Net P&L: <b>₹{net_pnl_inr:+,.0f}</b>")
                if notify_telegram:
                    header = "✅ <b>INTRADAY TRADE CLOSED — NET PROFIT</b>" if net_pnl_inr >= 0 else "🔴 <b>INTRADAY TRADE CLOSED — CUT & RESEARCHED</b>"
                    self.notifier.send_message(
                        f"{header}\n━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> ({side}) | <b>Arm:</b> <code>{arm_used}</code>\n"
                        f"🏷️ <b>AI Decision:</b> {exit_status} ({hold_mins:.0f} mins)\n"
                        f"🔢 <b>Entry:</b> ₹{entry} ➔ <b>Exit:</b> <b>₹{exit_price}</b>\n"
                        f"🧾 <b>Gross P&L:</b> ₹{gross_pnl_inr:+,.2f} | <b>Intraday Taxes/Fees:</b> -₹{charges_inr:.2f}\n"
                        f"💰 <b>Net Realized P&L:</b> <b>₹{net_pnl_inr:+,.2f} ({net_pnl_pct:+.2f}%)</b>\n"
                        f"🏦 <b>Updated Wallet Equity:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>\n"
                        f"🧠 <b>Post-Trade AI Research & Lesson:</b>\n<i>{lesson}</i>"
                    )

        self._save_ledger()
        self._save_portfolio()
        self._save_weights()
        return actions_taken

    def _research_and_learn_from_trade(self, trade, pre_fetched_news=None):
        sym = trade["symbol"]
        pnl_inr = float(trade.get("pnl_inr", 0.0))
        peak_pnl = float(trade.get("peak_pnl_inr", 0.0))
        side = trade.get("action", "BUY")
        session_used = trade.get("entry_session", self._get_session_bucket())
        regime_used = trade.get("market_regime", "NEUTRAL")
        arm_used = trade.get("strategy_arm", "MOMENTUM_BREAKOUT")
        arch_used = trade.get("archetype", "MOMENTUM_MOVER")
        hold_mins = float(trade.get("hold_minutes", 0.0))

        news_ctx = pre_fetched_news or self._analyze_live_news_emotion(sym)
        latest_headline = news_ctx.get("top_headline", "No new headline")
        latest_emotion = news_ctx.get("emotion", "NEUTRAL")

        if pnl_inr < 0:
            entry_rsi = float(trade.get("entry_rsi", 50.0))
            reasons = []
            if (side == "BUY" and latest_emotion == "NEGATIVE") or (side.startswith("SELL") and latest_emotion == "POSITIVE"):
                reasons.append(f"sudden {latest_emotion} news ('{latest_headline[:65]}')")
            if peak_pnl > abs(pnl_inr) * 0.5 and peak_pnl > 200:
                reasons.append(f"gave back +₹{peak_pnl:,.0f} peak profit")
            if side == "BUY" and entry_rsi > 68:
                reasons.append(f"overbought entry RSI {entry_rsi}")
            if not reasons:
                reasons.append(f"{arm_used} on {arch_used} underperformed in {session_used} [{regime_used}]")

            lesson = f"MISTAKE AUTOPSY ({sym} {side}, ₹{pnl_inr:+,.0f} net): Failed due to {', '.join(reasons)}."
            trade["mistake_diagnosis"] = lesson
            self.portfolio["mistakes_identified"] = ([lesson] + self.portfolio.get("mistakes_identified", []))[:20]
        else:
            lesson = f"WINNING EDGE ({sym} {side}, +₹{pnl_inr:,.0f} net in {hold_mins:.0f}m): Reinforced {arm_used} on {arch_used} [{regime_used}]."
            trade["mistake_diagnosis"] = lesson

        self.weights.setdefault("trade_lessons_log", []).insert(0, {
            "timestamp": self._now_ist().strftime("%Y-%m-%d %I:%M %p IST"),
            "symbol": sym, "pnl_inr": round(pnl_inr, 2), "session": session_used,
            "regime": regime_used, "arm": arm_used, "archetype": arch_used,
            "emotion_at_exit": latest_emotion, "lesson": lesson
        })
        self.weights["trade_lessons_log"] = self.weights["trade_lessons_log"][:50]
        self._optimize_weights_from_history(force_recalibrate=False)   # cadence-limited; EOD forces it
        return lesson

    def evaluate_open_trades_and_learn(self, notify_telegram=True):
        self.manage_live_positions(notify_telegram=notify_telegram)
        learning_summary = self._optimize_weights_from_history(force_recalibrate=True)

        ml_info = self.ml_model.train_walk_forward(self.ledger, self.shadow_ledger)
        if ml_info.get("trained"):
            self.weights["ml_model_summary"] = (
                f"{ml_info['backend']} | n={ml_info['samples_count']} | "
                f"WF AUC={ml_info['wf_auc']:.2f} | WF WR={ml_info['wf_win_rate']}%"
            )
        else:
            self.weights["ml_model_summary"] = (
                f"Collecting samples ({ml_info.get('samples_count', 0)}/{ml_info.get('min_required', 60)})"
            )
        self._save_weights()

        today_str = self._now_ist().strftime("%Y-%m-%d")
        today_pnl = sum(float(t.get("pnl_inr", 0)) for t in self.ledger if t.get("date") == today_str and t.get("status") != "OPEN")
        self._recalculate_cash(self.portfolio)
        eq_log = self.portfolio.get("daily_equity_log", [])
        if eq_log and eq_log[-1].get("date") == today_str:
            eq_log[-1]["equity"] = self.portfolio["current_equity"]
            eq_log[-1]["pnl"] = round(today_pnl, 2)
        else:
            eq_log.append({"date": today_str, "equity": self.portfolio["current_equity"], "pnl": round(today_pnl, 2)})
        self.portfolio["daily_equity_log"] = eq_log[-60:]
        self._save_portfolio()

        open_count = sum(1 for t in self.ledger if t.get("status") == "OPEN")
        return {"evaluated": len(self.ledger) - open_count, "open_positions": open_count, "message": learning_summary}

    def _optimize_weights_from_history(self, force_recalibrate=False):
        settled = [t for t in self.ledger if t.get("status") != "OPEN" and "factor_scores" in t]
        if not settled:
            return "No settled trades yet to train the AI."
        n_settled = len(settled)
        last_eval = int(self.weights.get("last_evaluated_count", 0))
        if not force_recalibrate and (n_settled - last_eval) < RECALIBRATE_EVERY_N_TRADES:
            return self.weights.get("learning_notes", "Model up to date.")
        learning_active = n_settled >= MIN_TRADES_FOR_WEIGHT_LEARNING
        shrink = n_settled / (n_settled + WEIGHT_SHRINKAGE_K)

        base = {"volume": 0.28, "sentiment": 0.28, "premarket": 0.22, "momentum": 0.22}
        adjustments = {"volume": 0.0, "sentiment": 0.0, "premarket": 0.0, "momentum": 0.0}
        recent = settled[-50:]

        wins_count = sl_hits = target_hits = giveback_losses = 0
        winning_convictions, losing_convictions, losing_buy_rsis, winning_hold_times = [], [], [], []

        for t in recent:
            pnl, pnl_inr, peak_inr = float(t.get("pnl_pct", 0.0)), float(t.get("pnl_inr", 0.0)), float(t.get("peak_pnl_inr", 0.0))
            status_str, conv, hold_m = str(t.get("status", "")), float(t.get("conviction", 60.0)), float(t.get("hold_minutes", 0.0))
            if pnl_inr > 0:
                wins_count += 1
                winning_convictions.append(conv)
                if hold_m > 0:
                    winning_hold_times.append(hold_m)
            else:
                losing_convictions.append(conv)
                if peak_inr > 250 and pnl_inr < 0:
                    giveback_losses += 1
                if t.get("action") == "BUY" and "entry_rsi" in t:
                    losing_buy_rsis.append(float(t["entry_rsi"]))

            if "SL HIT" in status_str:
                sl_hits += 1
            elif "TARGET" in status_str or "PROFIT LOCKED" in status_str:
                target_hits += 1

            direction = 1.0 if pnl_inr > 0 else (-1.25 if pnl_inr < 0 else 0.0)
            magnitude = min(2.5, max(0.5, abs(pnl)))
            for factor_key in adjustments:
                factor_strength = (float(t["factor_scores"].get(factor_key, 50.0)) - 50.0) / 100.0
                adjustments[factor_key] += direction * magnitude * factor_strength * 0.018

        # Statistical guard-rail: below MIN_TRADES the weights stay at their priors; above it the learned
        # adjustment is shrunk toward the prior by n / (n + K).  (5 trades must not re-weight a model.)
        if learning_active:
            adjustments = {k: v * shrink for k, v in adjustments.items()}
        else:
            adjustments = {k: 0.0 for k in adjustments}

        raw_vol = max(0.12, min(0.50, base["volume"] + adjustments["volume"]))
        raw_sent = max(0.12, min(0.50, base["sentiment"] + adjustments["sentiment"]))
        raw_pre = max(0.12, min(0.50, base["premarket"] + adjustments["premarket"]))
        raw_mom = max(0.12, min(0.50, base["momentum"] + adjustments["momentum"]))
        total = raw_vol + raw_sent + raw_pre + raw_mom

        self.weights["volume_weight"] = round(raw_vol / total, 3)
        self.weights["sentiment_weight"] = round(raw_sent / total, 3)
        self.weights["premarket_weight"] = round(raw_pre / total, 3)
        self.weights["momentum_weight"] = round(1.0 - (self.weights["volume_weight"] + self.weights["sentiment_weight"] + self.weights["premarket_weight"]), 3)

        ticker_stats = {}
        for t in settled:
            sym, pnl_inr = t["symbol"], float(t.get("pnl_inr", 0.0))
            st = ticker_stats.setdefault(sym, {"trades": 0, "wins": 0, "losses": 0, "net_pnl_inr": 0.0})
            st["trades"] += 1
            if pnl_inr > 0:
                st["wins"] += 1
            else:
                st["losses"] += 1
            st["net_pnl_inr"] = round(st["net_pnl_inr"] + pnl_inr, 2)

        blacklisted, preferred = [], []
        for sym, st_data in ticker_stats.items():
            st_data["win_rate"] = round((st_data["wins"] / st_data["trades"]) * 100.0, 1)
            if st_data["trades"] >= 4 and st_data["win_rate"] < 35.0 and st_data["net_pnl_inr"] < -500:
                blacklisted.append(sym)
            elif st_data["trades"] >= 4 and st_data["win_rate"] >= 65.0 and st_data["net_pnl_inr"] > 300:
                preferred.append(sym)

        self.weights["ticker_memory"], self.weights["blacklisted_tickers"], self.weights["preferred_tickers"] = ticker_stats, blacklisted, preferred

        session_stats = {}
        for t in settled:
            sess, pnl_inr = t.get("entry_session", "MORNING_TREND (09:45-11:30)"), float(t.get("pnl_inr", 0.0))
            s_d = session_stats.setdefault(sess, {"trades": 0, "wins": 0, "net_pnl_inr": 0.0, "score_adj": 0.0})
            s_d["trades"] += 1
            if pnl_inr > 0:
                s_d["wins"] += 1
            s_d["net_pnl_inr"] = round(s_d["net_pnl_inr"] + pnl_inr, 2)

        for sess, s_data in session_stats.items():
            wr = (s_data["wins"] / s_data["trades"]) * 100.0
            s_data["win_rate"] = round(wr, 1)
            s_data["score_adj"] = round(max(-12.0, min(8.0, (wr - 50.0) * 0.25)), 1) if s_data["trades"] >= 5 else 0.0
        self.weights["session_memory"] = session_stats

        if learning_active:
            sl_rate = (sl_hits / len(recent)) * 100.0
            tp_rate = (target_hits / len(recent)) * 100.0
            self.weights["atr_sl_multiplier"] = round(min(0.72, 0.55 + ((sl_rate - 40.0) * 0.004)), 2) if sl_rate > 40.0 else (0.52 if sl_rate < 20.0 else 0.55)
            self.weights["atr_tp_multiplier"] = 1.80 if tp_rate >= 45.0 else (1.40 if (tp_rate < 20.0 and len(recent) >= 5) else 1.60)
            if giveback_losses >= 2:
                self.weights["max_profit_giveback_pct"] = max(25.0, round(45.0 - (giveback_losses * 5.0), 1))
            if winning_hold_times:
                self.weights["learned_max_hold_mins"] = round(max(45.0, min(240.0, (sum(winning_hold_times) / len(winning_hold_times)) * 1.5)), 1)
            if losing_convictions and winning_convictions:
                avg_loss_conv, avg_win_conv = sum(losing_convictions) / len(losing_convictions), sum(winning_convictions) / len(winning_convictions)
                if avg_win_conv > avg_loss_conv:
                    self.weights["min_conviction_threshold"] = round(max(50.0, min(68.0, (avg_loss_conv + avg_win_conv) / 2.0)), 1)
            if losing_buy_rsis and len([r for r in losing_buy_rsis if r >= 67.0]) >= 2:
                self.weights["buy_rsi_ceiling"] = 66.0

            recent_win_rate = (wins_count / len(recent)) * 100.0
            total_contrib = float(self.portfolio.get("total_capital_contributed", INITIAL_CAPITAL))
            if recent_win_rate >= 60.0 and self.portfolio["current_equity"] >= total_contrib:
                self.weights["risk_per_trade_pct"] = min(1.5, round(1.0 + ((recent_win_rate - 50.0) * 0.015), 2))
            elif recent_win_rate < 45.0 or self.portfolio["current_equity"] < total_contrib:
                self.weights["risk_per_trade_pct"] = max(0.6, round(1.0 - ((50.0 - recent_win_rate) * 0.015), 2))

        else:
            self.weights.update({
                "atr_sl_multiplier": 0.55, "atr_tp_multiplier": 1.60, "max_profit_giveback_pct": 45.0,
                "learned_max_hold_mins": 180.0, "min_conviction_threshold": 52.0,
                "buy_rsi_ceiling": 72.0, "risk_per_trade_pct": 1.0,
            })

        self.weights["generation"] = int(self.weights.get("generation", 1)) + 1
        self.weights["last_evaluated_count"] = len(settled)
        self.weights["last_updated"] = self._now_ist().strftime("%Y-%m-%d %I:%M %p IST")

        arms = self.weights.get("bandit_arms", self._default_bandit_arms())
        top_arm = max(arms.items(), key=lambda kv: float(kv[1].get("alpha", 2.0)) / (float(kv[1].get("alpha", 2.0)) + float(kv[1].get("beta", 2.0))))
        top_arm_prob = float(top_arm[1]["alpha"]) / (float(top_arm[1]["alpha"]) + float(top_arm[1]["beta"]))
        shadow_n = int(self.weights.get("shadow_samples_learned", 0))

        mode = "LEARNING" if learning_active else f"WARM-UP {n_settled}/{MIN_TRADES_FOR_WEIGHT_LEARNING} (parameters frozen at priors)"
        note = f"Gen #{self.weights['generation']} [{mode}] ({n_settled} real + {shadow_n} shadow): Top Arm={top_arm[0]} ({top_arm_prob:.0%}), Min Conv={self.weights['min_conviction_threshold']}"
        self.weights["learning_notes"] = note
        self._save_weights()
        return note

    def get_performance_metrics(self):
        self._recalculate_cash(self.portfolio)
        settled = [t for t in self.ledger if t.get("status") != "OPEN"]
        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        unrealized_total = sum(float(t.get("unrealized_pnl_inr", 0.0)) for t in open_trades)
        total_contributed = float(self.portfolio.get("total_capital_contributed", INITIAL_CAPITAL))
        wins = [t for t in settled if float(t.get("pnl_inr", 0)) > 0]
        win_rate = round((len(wins) / len(settled)) * 100.0, 1) if settled else 0.0
        total_pnl_inr = round(sum(float(t.get("pnl_inr", 0)) for t in settled), 2)
        roi_pct = round((total_pnl_inr / total_contributed) * 100.0, 2) if total_contributed > 0 else 0.0

        pnls = [float(t.get("pnl_inr", 0.0)) for t in settled]
        gross_win = sum(x for x in pnls if x > 0)
        gross_loss = -sum(x for x in pnls if x < 0)
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (None if gross_win == 0 else 999.0)
        expectancy = round(sum(pnls) / len(pnls), 2) if pnls else 0.0
        peak = run = max_dd = 0.0
        for x in pnls:
            run += x
            peak = max(peak, run)
            max_dd = max(max_dd, peak - run)
        max_dd_pct = round((max_dd / total_contributed) * 100.0, 2) if total_contributed > 0 else 0.0
        if len(pnls) < 30:
            confidence = f"Too few trades ({len(pnls)}) to say anything about edge"
        elif len(pnls) < 100:
            confidence = f"Early read only ({len(pnls)} trades) - aim for 100+ before trusting results"
        else:
            confidence = f"{len(pnls)} trades - sample is large enough to start judging expectancy"

        return {
            "profit_factor": profit_factor, "expectancy_inr": expectancy,
            "max_drawdown_inr": round(max_dd, 2), "max_drawdown_pct": max_dd_pct, "edge_confidence": confidence,
            "initial_capital": INITIAL_CAPITAL,
            "manual_deposits": self.portfolio.get("manual_deposits", 0.0),
            "total_capital_contributed": total_contributed,
            "current_equity": self.portfolio["current_equity"],
            "available_cash": self.portfolio["available_cash"],
            "allocated_capital": self.portfolio["allocated_capital"],
            "unrealized_pnl_inr": round(unrealized_total, 2),
            "total_trades": len(self.ledger),
            "settled_trades": len(settled),
            "open_trades": len(open_trades),
            "win_rate": win_rate,
            "total_pnl_pct": roi_pct,
            "total_pnl_inr": total_pnl_inr,
            "total_taxes_paid": self.portfolio.get("total_taxes_paid", 0.0),
            "shadow_samples_learned": int(self.weights.get("shadow_samples_learned", 0)),
        }