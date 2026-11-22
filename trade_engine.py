import os
import json
import random
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
from market_data import MarketDataScanner
from telegram_notifier import TelegramNotifier

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
MAX_SECTOR_EXPOSURE_PCT = 0.40   # Max 40% of portfolio equity in one sector
MIN_OVERNIGHT_CUSHION_PCT = 1.0  # Require >= +1.0% unrealized gain to hold overnight

BULLISH_KEYWORDS = {
    "surge", "soar", "jump", "gain", "profit", "beat", "order", "win", "won",
    "contract", "approval", "upgrade", "buy", "bullish", "growth", "expansion",
    "dividend", "bonus", "acquisition", "partnership", "record", "strong", "rally"
}
BEARISH_KEYWORDS = {
    "fall", "drop", "plunge", "slump", "loss", "miss", "downgrade", "sell",
    "bearish", "fraud", "probe", "raid", "penalty", "fine", "ban", "recall",
    "default", "debt", "resign", "cut", "weak", "crash", "warning", "lawsuit"
}


class AdaptiveTradeEngine:
    def __init__(self):
        self.notifier = TelegramNotifier()
        self.weights = self._load_weights()
        self.ledger = self._load_ledger()
        self.shadow_ledger = self._load_shadow_ledger()
        self.portfolio = self._load_portfolio()
        self.sector_cache = self._load_sector_cache()

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

    # ------------------------------------------------------------------
    # REALISTIC 2026 INDIAN NSE STATUTORY TAXES & ARCHETYPE SLIPPAGE
    # ------------------------------------------------------------------
    @staticmethod
    def _get_archetype_slippage_pct(archetype: str) -> float:
        """Realistic NSE bid-ask impact slippage by stock liquidity archetype."""
        if archetype == "LARGE_CAP_STEADY":
            return 0.03
        if archetype == "HIGH_VOL_MIDCAP":
            return 0.14
        return 0.07  # MOMENTUM_MOVER

    @staticmethod
    def _calculate_nse_charges(side, qty, entry_price, exit_price, is_overnight=False):
        qty = max(1, int(qty))
        buy_val = (entry_price * qty) if side == "BUY" else (exit_price * qty)
        sell_val = (exit_price * qty) if side == "BUY" else (entry_price * qty)
        turnover = buy_val + sell_val

        if is_overnight:
            stt = round(0.001 * turnover)
            stamp = round(0.00015 * buy_val)
            brokerage = 0.0 if side == "BUY" else min(20.0, 0.0003 * buy_val)
            dp_charges = 15.93 if side == "BUY" else 0.0
        else:
            stt = round(0.00025 * sell_val)
            stamp = round(0.00003 * buy_val)
            brokerage = min(20.0, 0.0003 * buy_val) + min(20.0, 0.0003 * sell_val)
            dp_charges = 0.0

        nse_txn = 0.0000307 * turnover
        sebi_fee = 0.000001 * turnover
        gst = 0.18 * (brokerage + nse_txn + sebi_fee)
        return round(stt + stamp + brokerage + nse_txn + sebi_fee + gst + dp_charges, 2)

    # ------------------------------------------------------------------
    # QLIB TRA: 3-ARCHETYPE STOCK ROUTING & ALPHA158 MICROSTRUCTURE
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ADVERSARIAL CRITIC + FREE-TIER GEMINI FLASH BULL-VS-BEAR DEBATE
    # ------------------------------------------------------------------
    @staticmethod
    def _adversarial_bear_critic(side, ltp, atr, chg_pct, rsi, clv, intraday_alpha, archetype, regime_name):
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
        else:
            if atr_pct > 0 and chg_pct < -(2.1 * atr_pct):
                red_flags.append("Chasing exhausted >2.1x ATR drop")
            if clv > 0.35:
                red_flags.append(f"Bullish lower-wick bounce (CLV {clv:+.2f})")
            if rsi < 26.0:
                red_flags.append(f"Oversold RSI ({rsi:.1f})")
            if intraday_alpha > 0.6:
                red_flags.append(f"Outperforming Nifty 50 ({intraday_alpha:+.2f}% alpha)")

        if archetype == "HIGH_VOL_MIDCAP" and regime_name in ("SIDEWAYS_CHOP", "HIGH_VOLATILITY_PANIC"):
            red_flags.append(f"High-vol midcap during {regime_name}")

        vetoed = len(red_flags) >= 2
        penalty = round(len(red_flags) * 4.5, 1)
        return vetoed, penalty, red_flags

    def _gemini_bull_bear_debate(self, candidate: dict, regime_info: dict) -> dict:
        """
        FactorMAD Multi-Agent Debate using Google GenAI free tier (gemini-2.5-flash /
        gemini-3.5-flash-lite). Runs ONLY on top 2 finalists (~10-15 calls/day) to critique
        full news context, delivery %, OI walls, and priced-in rumor risks.
        """
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or not HAS_GENAI:
            return {"veto": False, "conviction_adj": 0.0, "debate_note": "Quantitative Bear Critic passed"}

        prompt = (
            "You are an institutional NSE risk committee running a Bull vs. Bear adversarial debate "
            "before executing an Indian equity trade. Analyze this candidate strictly in JSON:\n"
            f"- Stock: {candidate['symbol']} | Action: {candidate['action']} | LTP: ₹{candidate['entry_price']}\n"
            f"- Nifty Regime: {regime_info.get('regime')} (Nifty {regime_info.get('nifty_chg_pct')}%, "
            f"VIX {regime_info.get('vix_ltp')}, FII Net ₹{regime_info.get('fii_net_cr', 0)} Cr)\n"
            f"- Delivery %: {candidate.get('delivery_pct', 0)}% | Mansfield RS: {candidate.get('mansfield_rs', 0)}% | "
            f"CLV: {candidate.get('clv', 0)} | RSI: {candidate.get('rsi', 50)}\n"
            f"- Option OI Walls: Call Resistance ₹{candidate.get('max_call_oi_strike', 0)}, "
            f"Put Support ₹{candidate.get('max_put_oi_strike', 0)}\n"
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

    # ------------------------------------------------------------------
    # PAGE-HINKLEY ONLINE CONCEPT-DRIFT DETECTOR
    # ------------------------------------------------------------------
    def _check_page_hinkley_drift(self):
        """
        Monitors rolling prediction residual errors across recent settled trades.
        If a sudden macro shock (Budget, RBI policy, tariff shock) causes structural drift,
        automatically softens Bandit alpha/beta priors toward neutral so the AI adapts immediately.
        """
        settled = [t for t in self.ledger if t.get("status") != "OPEN" and "pnl_inr" in t]
        if len(settled) < 8:
            return False

        recent = settled[-12:]
        errors = [1.0 if float(t.get("pnl_inr", 0.0)) <= 0 else 0.0 for t in recent]
        mean_err = sum(errors) / len(errors)

        ph_sum = 0.0
        min_ph = 0.0
        delta = 0.15
        threshold = 2.6

        for e in errors:
            ph_sum += (e - mean_err - delta)
            min_ph = min(min_ph, ph_sum)

        if (ph_sum - min_ph) > threshold and mean_err >= 0.65:
            arms = self.weights.get("bandit_arms", {})
            for arm_stats in arms.values():
                arm_stats["alpha"] = round(max(2.0, float(arm_stats.get("alpha", 2.0)) * 0.5), 2)
                arm_stats["beta"] = round(max(2.0, float(arm_stats.get("beta", 2.0)) * 0.5), 2)

            self.weights["drift_alert"] = (
                f"Concept drift detected on {self._now_ist().strftime('%Y-%m-%d %I:%M %p IST')} "
                f"(rolling loss rate {mean_err:.0%}); reset Bandit priors for rapid adaptation."
            )
            print(f"[DRIFT DETECTOR] {self.weights['drift_alert']}")
            return True
        return False

    # ------------------------------------------------------------------
    # NIFTY 50, INDIA VIX & FII/DII MARKET REGIME ENGINE
    # ------------------------------------------------------------------
    def _detect_market_regime(self):
        default_regime = {
            "regime": "NEUTRAL",
            "nifty_ltp": 0.0,
            "nifty_chg_pct": 0.0,
            "vix_ltp": 14.0,
            "vix_chg_pct": 0.0,
            "fii_net_cr": 0.0,
            "dii_net_cr": 0.0,
            "buy_adj": 0.0,
            "short_adj": 0.0,
            "size_multiplier": 1.0,
            "sl_multiplier_adj": 1.0,
            "updated_at": self._now_ist().strftime("%I:%M %p IST")
        }
        try:
            nifty_df = yf.Ticker("^NSEI").history(period="5d", interval="15m")
            vix_df = yf.Ticker("^INDIAVIX").history(period="5d", interval="1d")
            fii_dii = MarketDataScanner.fetch_fii_dii_activity()
            fii_net = float(fii_dii.get("fii_net_cr", 0.0))
            dii_net = float(fii_dii.get("dii_net_cr", 0.0))

            if nifty_df.empty:
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

            vix_ltp = 14.0
            vix_chg_pct = 0.0
            if not vix_df.empty:
                vix_ltp = float(vix_df["Close"].iloc[-1])
                if len(vix_df) >= 2:
                    prev_vix = float(vix_df["Close"].iloc[-2])
                    vix_chg_pct = ((vix_ltp - prev_vix) / prev_vix) * 100.0 if prev_vix > 0 else 0.0

            # FII institutional flow bias (-3 to +3 pts)
            fii_bias = 3.0 if fii_net > 1500 else (-3.0 if fii_net < -1500 else 0.0)

            if vix_ltp >= 18.5 or vix_chg_pct >= 8.0:
                regime = "HIGH_VOLATILITY_PANIC"
                buy_adj = -8.0 + fii_bias
                short_adj = -3.0 - fii_bias
                size_mult = 0.50
                sl_adj = 1.25
            elif nifty_chg_pct <= -0.45 and nifty_ltp < intraday_mid and nifty_ltp < ema_20:
                regime = "BEAR_TREND"
                buy_adj = -8.0 + fii_bias
                short_adj = 6.0 - fii_bias
                size_mult = 0.85
                sl_adj = 1.05
            elif nifty_chg_pct >= 0.40 and nifty_ltp > intraday_mid and nifty_ltp > ema_20:
                regime = "BULL_TREND"
                buy_adj = 6.0 + fii_bias
                short_adj = -6.0 - fii_bias
                size_mult = 1.0
                sl_adj = 1.0
            else:
                regime = "SIDEWAYS_CHOP"
                buy_adj = -2.0 + fii_bias
                short_adj = -2.0 - fii_bias
                size_mult = 0.80
                sl_adj = 0.95

            info = {
                "regime": regime,
                "nifty_ltp": round(nifty_ltp, 2),
                "nifty_chg_pct": round(nifty_chg_pct, 2),
                "vix_ltp": round(vix_ltp, 2),
                "vix_chg_pct": round(vix_chg_pct, 2),
                "fii_net_cr": fii_net,
                "dii_net_cr": dii_net,
                "buy_adj": round(buy_adj, 1),
                "short_adj": round(short_adj, 1),
                "size_multiplier": size_mult,
                "sl_multiplier_adj": sl_adj,
                "updated_at": self._now_ist().strftime("%I:%M %p IST")
            }
            self.weights["last_market_regime"] = info
            return info
        except Exception as e:
            print(f"[WARN] Market regime detection fallback: {e}")
            return default_regime

    # ------------------------------------------------------------------
    # MANSFIELD RELATIVE STRENGTH VS NIFTY 50
    # ------------------------------------------------------------------
    def _batch_compute_mansfield_rs(self, symbols):
        rs_map = {s: 0.0 for s in symbols}
        if not symbols:
            return rs_map

        try:
            ns_map = {s: (s if s.endswith(".NS") else f"{s}.NS") for s in symbols}
            tickers = ["^NSEI"] + list(set(ns_map.values()))
            raw = yf.download(
                tickers=tickers,
                period="1mo",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False
            )
            if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
                return rs_map

            if "^NSEI" not in raw.columns.get_level_values(0):
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
                    if len(aligned) < 10:
                        continue

                    rs_series = aligned["stk"] / aligned["nifty"]
                    sma_10 = float(rs_series.tail(10).mean())
                    if sma_10 > 0:
                        mrs = ((float(rs_series.iloc[-1]) / sma_10) - 1.0) * 100.0
                        rs_map[orig_sym] = round(mrs, 2)
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] Mansfield RS batch computation fallback: {e}")

        return rs_map

    # ------------------------------------------------------------------
    # THOMPSON SAMPLING MULTI-ARMED BANDIT & HALF-KELLY SIZING
    # ------------------------------------------------------------------
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
        alpha = max(0.5, float(arm.get("alpha", 2.0)))
        beta = max(0.5, float(arm.get("beta", 2.0)))

        sampled_theta = random.betavariate(alpha, beta)
        posterior_prob = alpha / (alpha + beta)
        bandit_adj = round(max(-8.0, min(10.0, (sampled_theta - 0.50) * 22.0)), 2)
        return sampled_theta, posterior_prob, bandit_adj

    def _update_bandit_arm(self, arm_name, pnl_inr, risk_inr, weight_scale=1.0, count_as_real=True):
        arms = self.weights.setdefault("bandit_arms", self._default_bandit_arms())
        arm = arms.setdefault(arm_name, {"alpha": 2.0, "beta": 2.0, "trades": 0, "wins": 0, "net_pnl_inr": 0.0})

        alpha = float(arm.get("alpha", 2.0))
        beta = float(arm.get("beta", 2.0))
        r_mag = min(2.0, max(0.4, abs(pnl_inr) / max(risk_inr, 100.0))) * float(weight_scale)

        if pnl_inr > 0:
            alpha += r_mag
            if count_as_real:
                arm["wins"] = int(arm.get("wins", 0)) + 1
        else:
            beta += r_mag

        if (alpha + beta) > 60.0:
            scale = 50.0 / (alpha + beta)
            alpha = max(1.0, alpha * scale)
            beta = max(1.0, beta * scale)

        arm["alpha"] = round(alpha, 3)
        arm["beta"] = round(beta, 3)
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
        clamped_fraction = round(max(0.08, min(0.25, half_kelly)), 4)
        return clamped_fraction, round(payoff_b, 2)

    # ------------------------------------------------------------------
    # COUNTERFACTUAL SHADOW-TRADE LEDGER
    # ------------------------------------------------------------------
    def _load_shadow_ledger(self):
        if os.path.exists(SHADOW_LEDGER_FILE):
            try:
                with open(SHADOW_LEDGER_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_shadow_ledger(self):
        try:
            with open(SHADOW_LEDGER_FILE, "w") as f:
                json.dump(self.shadow_ledger[-150:], f, indent=2)
        except Exception:
            pass

    def _record_shadow_candidates(self, skipped_candidates):
        if not skipped_candidates:
            return
        now_ist = self._now_ist()
        today_str = now_ist.strftime("%Y-%m-%d")
        session_bucket = self._get_session_bucket(now_ist)

        existing_today = {
            s["symbol"] for s in self.shadow_ledger if s.get("date") == today_str
        }
        today_count = len(existing_today)

        added = 0
        for item in skipped_candidates:
            if today_count + added >= 15:
                break
            sym = item["symbol"]
            if sym in existing_today or float(item.get("conviction", 0)) < 40.0:
                continue

            self.shadow_ledger.append({
                "date": today_str,
                "entry_iso": now_ist.isoformat(),
                "entry_session": session_bucket,
                "symbol": sym,
                "action": item["action"],
                "strategy_arm": item.get("strategy_arm", "MOMENTUM_BREAKOUT"),
                "archetype": item.get("archetype", "MOMENTUM_MOVER"),
                "entry_price": float(item["entry_price"]),
                "target_1": float(item["target_1"]),
                "stop_loss": float(item["stop_loss"]),
                "conviction": float(item["conviction"]),
                "skip_reason": item.get("skip_reason", "Below conviction threshold"),
                "status": "SHADOW_OPEN",
                "shadow_pnl_pct": 0.0
            })
            existing_today.add(sym)
            added += 1

        if added > 0:
            self._save_shadow_ledger()

    def _evaluate_shadow_trades(self):
        now_ist = self._now_ist()
        mins_now = now_ist.hour * 60 + now_ist.minute
        near_close = mins_now >= (15 * 60 + 10)

        due_shadows = []
        for s in self.shadow_ledger:
            if s.get("status") != "SHADOW_OPEN":
                continue
            age_mins = self._calculate_hold_minutes(s)
            if age_mins >= 75.0 or near_close or s.get("date") != now_ist.strftime("%Y-%m-%d"):
                due_shadows.append(s)

        if not due_shadows:
            return 0

        syms = list({s["symbol"] for s in due_shadows})
        snaps = MarketDataScanner.fetch_live_intraday_snapshot(syms)
        evaluated = 0

        for s in due_shadows:
            sym = s["symbol"]
            snap = snaps.get(sym)
            if not snap:
                continue

            ltp = float(snap["current_price"])
            d_high = float(snap["day_high"])
            d_low = float(snap["day_low"])
            entry = float(s["entry_price"])
            t1 = float(s["target_1"])
            sl = float(s["stop_loss"])
            side = s["action"]

            if side == "BUY":
                if d_high >= t1 and d_low > sl:
                    exit_px = t1
                elif d_low <= sl and d_high < t1:
                    exit_px = sl
                else:
                    exit_px = ltp
                pnl_pct = ((exit_px - entry) / entry) * 100.0
            else:
                if d_low <= t1 and d_high < sl:
                    exit_px = t1
                elif d_high >= sl and d_low > t1:
                    exit_px = sl
                else:
                    exit_px = ltp
                pnl_pct = ((entry - exit_px) / entry) * 100.0

            net_pnl_pct = round(pnl_pct - 0.08, 2)
            pseudo_pnl_inr = round((net_pnl_pct / 100.0) * 20000.0, 2)

            s["status"] = "SHADOW_WIN" if net_pnl_pct > 0 else "SHADOW_LOSS"
            s["shadow_pnl_pct"] = net_pnl_pct

            self._update_bandit_arm(
                s.get("strategy_arm", "MOMENTUM_BREAKOUT"),
                pseudo_pnl_inr,
                300.0,
                weight_scale=0.35,
                count_as_real=False
            )
            self._update_archetype_memory(
                s.get("archetype", "MOMENTUM_MOVER"),
                pseudo_pnl_inr,
                hit_sl=(exit_px == sl),
                weight_scale=0.35
            )
            evaluated += 1

        if evaluated > 0:
            self.weights["shadow_samples_learned"] = int(self.weights.get("shadow_samples_learned", 0)) + evaluated
            self._save_shadow_ledger()
            self._save_weights()
        return evaluated

    def _update_archetype_memory(self, archetype, pnl_inr, hit_sl=False, weight_scale=1.0):
        arch_mem = self.weights.setdefault("archetype_memory", {})
        rec = arch_mem.setdefault(archetype, {
            "weighted_wins": 1.0,
            "weighted_losses": 1.0,
            "sl_hits": 0,
            "trades": 0,
            "sl_mult_adj": 1.0,
            "score_adj": 0.0
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

    # ------------------------------------------------------------------
    # SECTOR CORRELATION & OVERNIGHT GAP GUARDS
    # ------------------------------------------------------------------
    def _load_sector_cache(self):
        if os.path.exists(SECTOR_CACHE_FILE):
            try:
                with open(SECTOR_CACHE_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_sector_cache(self):
        try:
            with open(SECTOR_CACHE_FILE, "w") as f:
                json.dump(self.sector_cache, f, indent=2)
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
        counts = {}
        capital = {}
        for t in self.ledger:
            if t.get("status") != "OPEN":
                continue
            sec = t.get("sector") or self._get_stock_sector(t["symbol"])
            counts[sec] = counts.get(sec, 0) + 1
            capital[sec] = capital.get(sec, 0.0) + float(t.get("capital_used", 0.0))
        return counts, capital

    def _has_upcoming_corporate_event(self, sym):
        clean_sym = sym.replace(".NS", "").strip().upper()
        try:
            cal = yf.Ticker(f"{clean_sym}.NS").calendar
            now_date = self._now_ist().date()
            cutoff_date = now_date + timedelta(days=2)

            if isinstance(cal, dict):
                edates = cal.get("Earnings Date") or []
                for d in edates:
                    d_obj = d.date() if hasattr(d, "date") else d
                    if now_date <= d_obj <= cutoff_date:
                        return True
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for val in cal.values.flatten():
                    if hasattr(val, "date"):
                        d_obj = val.date()
                        if now_date <= d_obj <= cutoff_date:
                            return True
        except Exception:
            pass
        return False

    def _check_overnight_hold_eligibility(self, trade, ltp, market_regime_info):
        now_ist = self._now_ist()
        mins = now_ist.hour * 60 + now_ist.minute
        if not ((15 * 60 + 5) <= mins <= (15 * 60 + 29)):
            return False, ""

        sym = trade["symbol"]
        side = trade["action"]
        entry = float(trade["entry_price"])
        unrealized_pct = (((ltp - entry) / entry) * 100.0) if side == "BUY" else (((entry - ltp) / entry) * 100.0)
        regime = market_regime_info.get("regime", "NEUTRAL")

        if self._has_upcoming_corporate_event(sym):
            return True, f"Overnight Event Guard: {sym} has earnings/results within 48h; closed before bell."

        if regime == "HIGH_VOLATILITY_PANIC":
            return True, f"Overnight Volatility Guard: India VIX elevated ({market_regime_info.get('vix_ltp')}); avoided gap risk."

        if (side == "BUY" and regime == "BEAR_TREND") or (side.startswith("SELL") and regime == "BULL_TREND"):
            return True, f"Overnight Regime Guard: Nifty regime ({regime}) opposes {side}; closed before bell."

        if unrealized_pct < MIN_OVERNIGHT_CUSHION_PCT:
            return True, (
                f"Overnight Cushion Guard: Profit buffer ({unrealized_pct:+.2f}%) is below "
                f"+{MIN_OVERNIGHT_CUSHION_PCT:.1f}% required to absorb overnight gap risk."
            )

        return False, ""

    # ------------------------------------------------------------------
    # REAL-TIME NEWS & EMOTION RESEARCH
    # ------------------------------------------------------------------
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

            bull_hits = 0
            bear_hits = 0
            headlines = []
            for item in items:
                title = (item.findtext("title") or "").strip()
                headlines.append(title)
                words = set(w.strip(".,:;!?()[]\"'").lower() for w in title.split())
                bull_hits += len(words & BULLISH_KEYWORDS)
                bear_hits += len(words & BEARISH_KEYWORDS)

            net = bull_hits - bear_hits
            if net >= 2:
                emotion = "POSITIVE"
            elif net <= -2:
                emotion = "NEGATIVE"
            else:
                emotion = "NEUTRAL"

            return {
                "emotion": emotion,
                "score": round(float(net), 2),
                "top_headline": headlines[0] if headlines else "No breaking news",
                "all_headlines": headlines[:3]
            }
        except Exception:
            return {"emotion": "NEUTRAL", "score": 0.0, "top_headline": "News feed unavailable", "all_headlines": []}

    # ------------------------------------------------------------------
    # STATE LOADERS & SAVERS
    # ------------------------------------------------------------------
    def _load_weights(self):
        default_weights = {
            "volume_weight": 0.28,
            "sentiment_weight": 0.28,
            "premarket_weight": 0.22,
            "momentum_weight": 0.22,
            "risk_per_trade_pct": 1.0,
            "min_conviction_threshold": 52.0,
            "atr_sl_multiplier": 0.55,
            "atr_tp_multiplier": 1.60,
            "buy_rsi_ceiling": 72.0,
            "short_rsi_floor": 28.0,
            "buy_bias_bonus": 0.0,
            "short_bias_bonus": 0.0,
            "max_profit_giveback_pct": 45.0,
            "learned_max_hold_mins": 180.0,
            "bandit_arms": self._default_bandit_arms(),
            "archetype_memory": {},
            "shadow_samples_learned": 0,
            "session_memory": {},
            "ticker_memory": {},
            "blacklisted_tickers": [],
            "preferred_tickers": [],
            "trade_lessons_log": [],
            "drift_alert": "",
            "last_market_regime": {"regime": "NEUTRAL", "nifty_chg_pct": 0.0, "vix_ltp": 14.0, "fii_net_cr": 0.0},
            "generation": 1,
            "last_evaluated_count": 0,
            "last_updated": self._now_ist().strftime("%Y-%m-%d %H:%M"),
            "learning_notes": "Initialized NSE Direct + Microstructure + Gemini Debate + Drift Brain."
        }
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE, "r") as f:
                    loaded = json.load(f)
                    default_weights.update(loaded)
                    if "bandit_arms" not in loaded:
                        default_weights["bandit_arms"] = self._default_bandit_arms()
            except Exception:
                pass
        return default_weights

    def _save_weights(self):
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(self.weights, f, indent=2)

    def _load_ledger(self):
        if os.path.exists(LEDGER_FILE):
            try:
                with open(LEDGER_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_ledger(self):
        with open(LEDGER_FILE, "w") as f:
            json.dump(self.ledger, f, indent=2)

    def _load_portfolio(self):
        today_str = self._now_ist().strftime("%Y-%m-%d")
        default_portfolio = {
            "initial_capital": INITIAL_CAPITAL,
            "manual_deposits": 0.0,
            "total_capital_contributed": INITIAL_CAPITAL,
            "deposit_history": [],
            "current_equity": INITIAL_CAPITAL,
            "available_cash": INITIAL_CAPITAL,
            "allocated_capital": 0.0,
            "total_realized_pnl": 0.0,
            "total_taxes_paid": 0.0,
            "win_streak": 0,
            "loss_streak": 0,
            "mistakes_identified": [],
            "daily_equity_log": [
                {"date": today_str, "equity": INITIAL_CAPITAL, "pnl": 0.0}
            ]
        }
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, "r") as f:
                    default_portfolio.update(json.load(f))
            except Exception:
                pass
        self._recalculate_cash(default_portfolio)
        return default_portfolio

    def _recalculate_cash(self, port_dict):
        deposits = port_dict.get("deposit_history", [])
        manual_total = round(sum(float(d.get("amount", 0.0)) for d in deposits), 2)
        total_contributed = round(INITIAL_CAPITAL + manual_total, 2)

        port_dict["initial_capital"] = INITIAL_CAPITAL
        port_dict["manual_deposits"] = manual_total
        port_dict["total_capital_contributed"] = total_contributed

        open_allocated = sum(
            float(t.get("capital_used", 0.0))
            for t in self.ledger if t.get("status") == "OPEN"
        )
        realized = sum(
            float(t.get("pnl_inr", 0.0))
            for t in self.ledger if t.get("status") != "OPEN"
        )
        taxes_paid = sum(
            float(t.get("charges_inr", 0.0))
            for t in self.ledger if t.get("status") != "OPEN"
        )
        port_dict["total_realized_pnl"] = round(realized, 2)
        port_dict["total_taxes_paid"] = round(taxes_paid, 2)
        port_dict["current_equity"] = round(max(0.0, total_contributed + realized), 2)
        port_dict["allocated_capital"] = round(open_allocated, 2)
        port_dict["available_cash"] = round(max(0.0, port_dict["current_equity"] - open_allocated), 2)

    def _save_portfolio(self):
        self._recalculate_cash(self.portfolio)
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(self.portfolio, f, indent=2)

    # ------------------------------------------------------------------
    # USER-ONLY MANUAL CAPITAL ADDITION
    # ------------------------------------------------------------------
    def add_manual_funds(self, amount, note="Manual user deposit", notify_telegram=True):
        amt = round(float(amount), 2)
        if amt <= 0:
            raise ValueError("Deposit amount must be greater than ₹0.")

        now_str = self._now_ist().strftime("%Y-%m-%d %I:%M:%S %p IST")
        self.portfolio.setdefault("deposit_history", []).append({
            "timestamp": now_str,
            "amount": amt,
            "note": str(note).strip() or "Manual user deposit"
        })
        self._save_portfolio()

        if notify_telegram:
            msg = (
                f"💵 <b>MANUAL FUNDS ADDED BY USER</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"➕ <b>Amount Added:</b> +₹{amt:,.2f}\n"
                f"📝 <b>Note:</b> {note}\n"
                f"🏦 <b>New Total Equity:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>\n"
                f"💰 <b>Available Cash to Trade:</b> <b>₹{self.portfolio['available_cash']:,.2f}</b>"
            )
            self.notifier.send_message(msg)

        return self.portfolio

    def save_morning_watchlist(self, rec_df):
        records = rec_df.to_dict(orient="records") if not rec_df.empty else []
        with open(MORNING_WATCHLIST_FILE, "w") as f:
            json.dump({"date": self._now_ist().strftime("%Y-%m-%d"), "items": records}, f, indent=2)

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
                changes.append(
                    f"🆕 <b>NEW SETUP DETECTED: {sym}</b> entered Top Picks ({row['action']} @ ₹{row['entry_price']}, Conviction {row['conviction']}/100)."
                )
            else:
                old_item = old_map[sym]
                if old_item["action"] != row["action"]:
                    changes.append(
                        f"🔄 <b>EMOTION & DIRECTION FLIP: {sym}</b> flipped from {old_item['action']} to <b>{row['action']}</b>!"
                    )
                elif abs(float(row["entry_price"]) - float(old_item["entry_price"])) / float(old_item["entry_price"]) > 0.01:
                    changes.append(
                        f"⚡ <b>LEVEL UPDATE: {sym}</b> adjusted to Entry ₹{row['entry_price']} | Target ₹{row['target_1']} | SL ₹{row['stop_loss']}."
                    )

        for old_sym in old_map:
            if old_sym not in new_symbols:
                changes.append(f"❌ <b>DROPPED: {old_sym}</b> removed due to weakening volume/sentiment.")

        return changes

    # ------------------------------------------------------------------
    # 1. FULL QUANTITATIVE + NSE MICROSTRUCTURE + GEMINI DEBATE SCORER
    # ------------------------------------------------------------------
    def generate_intraday_recommendations(self, enriched_df, premarket_df=None, top_k=5):
        if enriched_df.empty:
            return pd.DataFrame()

        self._recalculate_cash(self.portfolio)
        current_equity = float(self.portfolio["current_equity"])
        avail_cash = float(self.portfolio["available_cash"])

        regime_info = self._detect_market_regime()
        regime_name = regime_info["regime"]
        nifty_chg_pct = float(regime_info.get("nifty_chg_pct", 0.0))
        regime_buy_adj = float(regime_info["buy_adj"])
        regime_short_adj = float(regime_info["short_adj"])
        regime_size_mult = float(regime_info["size_multiplier"])
        regime_sl_adj = float(regime_info["sl_multiplier_adj"])

        current_session = self._get_session_bucket()
        session_mem = self.weights.get("session_memory", {}).get(current_session, {})
        session_adj = float(session_mem.get("score_adj", 0.0))
        session_blocked = session_mem.get("trades", 0) >= 4 and session_mem.get("win_rate", 50.0) < 30.0

        candidate_syms = enriched_df["symbol"].tolist()
        mansfield_rs_map = self._batch_compute_mansfield_rs(candidate_syms)

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

        scored_candidates = []
        skipped_for_shadow = []

        for _, row in enriched_df.iterrows():
            sym = row["symbol"]
            if sym in blacklist and len(enriched_df) > 3:
                continue

            ltp = float(row["ltp"])
            chg = float(row["change_pct"])
            vol_ratio = float(row["vol_ratio"])
            atr = float(row.get("atr", ltp * 0.015))
            rsi = float(row.get("rsi", 50.0))
            vwap = float(row.get("vwap_proxy", ltp))
            d_high = float(row.get("day_high", row.get("high", ltp)))
            d_low = float(row.get("day_low", row.get("low", ltp)))
            chart_trend = row.get("chart_trend", "RANGE_CONSOLIDATION")

            # Direct NSE Microstructure fields
            delivery_pct = float(row.get("delivery_pct", 0.0) or 0.0)
            lower_cp = float(row.get("lower_cp", 0.0) or 0.0)
            upper_cp = float(row.get("upper_cp", 0.0) or 0.0)
            max_call_oi = float(row.get("max_call_oi_strike", 0.0) or 0.0)
            max_put_oi = float(row.get("max_put_oi_strike", 0.0) or 0.0)

            if sym in pre_map and pre_map[sym]["iep"] > 50:
                chg = pre_map[sym]["pChange"]

            raw_sent = row.get("raw_sentiment", "NEUTRAL")
            ai_conf = float(row.get("AI Confidence", 50)) / 100.0
            headline = row.get("Why Is It Moving? (Latest News)", "Technical chart & volume setup")

            if (chg >= 0 and raw_sent != "NEGATIVE") or (chart_trend == "BULLISH_UPTREND" and raw_sent == "POSITIVE"):
                side = "BUY"
            elif (chg < 0 and raw_sent != "POSITIVE") or (chart_trend == "BEARISH_DOWNTREND" and raw_sent == "NEGATIVE"):
                side = "SELL (SHORT)"
            else:
                side = "BUY" if chg >= 0 else "SELL (SHORT)"

            # CIRCUIT-BREAKER PROXIMITY VETO (Blocks buying at upper circuit or shorting at lower circuit)
            if side == "BUY" and upper_cp > 0 and ltp >= (upper_cp * 0.985):
                print(f"[CIRCUIT GUARD] Vetoed BUY {sym}: LTP ₹{ltp} is within 1.5% of Upper Circuit ₹{upper_cp}.")
                continue
            if side.startswith("SELL") and lower_cp > 0 and ltp <= (lower_cp * 1.015):
                print(f"[CIRCUIT GUARD] Vetoed SHORT {sym}: LTP ₹{ltp} is within 1.5% of Lower Circuit ₹{lower_cp}.")
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
                side, ltp, atr, chg, rsi, clv, intraday_alpha, archetype, regime_name
            )

            # NSE Delivery Percentage Multiplier (rewards institutional accumulation >50%, penalizes <22% churn)
            deliv_mult = 1.20 if delivery_pct >= 50.0 else (0.75 if (0 < delivery_pct < 22.0) else 1.0)
            vol_score = min(100.0, (vol_ratio / 2.8) * 100.0 * deliv_mult)

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
                trend_bonus = 25.0 if chart_trend == "BULLISH_UPTREND" else 0.0
                vwap_bonus = 15.0 if ltp >= vwap else -10.0
                clv_bonus = 6.0 * clv
                if 50.0 <= rsi <= buy_rsi_max:
                    rsi_bonus = 15.0
                elif rsi > buy_rsi_max:
                    rsi_bonus = -15.0
                else:
                    rsi_bonus = 0.0
                mom_score = max(10.0, min(100.0, 45.0 + trend_bonus + vwap_bonus + rsi_bonus + clv_bonus))
                rs_adj = max(-6.0, min(8.0, (mansfield_rs * 1.5) + (intraday_alpha * 1.2)))
            else:
                trend_bonus = 25.0 if chart_trend == "BEARISH_DOWNTREND" else 0.0
                vwap_bonus = 15.0 if ltp <= vwap else -10.0
                clv_bonus = -6.0 * clv
                if short_rsi_min <= rsi <= 50.0:
                    rsi_bonus = 15.0
                elif rsi < short_rsi_min:
                    rsi_bonus = -15.0
                else:
                    rsi_bonus = 0.0
                mom_score = max(10.0, min(100.0, 45.0 + trend_bonus + vwap_bonus + rsi_bonus + clv_bonus))
                rs_adj = max(-6.0, min(8.0, (-mansfield_rs * 1.5) + (-intraday_alpha * 1.2)))

            base_conviction = (
                (w_vol * vol_score) +
                (w_sent * sent_score) +
                (w_pre * pre_score) +
                (w_mom * mom_score)
            )

            strategy_arm = self._classify_strategy_arm(
                vol_score, sent_score, pre_score, mom_score, mansfield_rs, side
            )
            _, posterior_prob, bandit_adj = self._sample_bandit_arm(strategy_arm)

            ticker_adj = 0.0
            if sym in ticker_memory:
                tm = ticker_memory[sym]
                trades_n = tm.get("trades", 0)
                win_r = tm.get("win_rate", 50.0)
                net_pnl = tm.get("net_pnl_inr", 0.0)
                if trades_n >= 1:
                    if win_r >= 60.0 and net_pnl > 0:
                        ticker_adj = min(8.0, 3.0 + (trades_n * 1.5))
                    elif win_r < 40.0 and net_pnl < 0:
                        ticker_adj = -min(15.0, 5.0 + (trades_n * 3.0))

            chop_adj = -8.0 if (regime_name == "SIDEWAYS_CHOP" and abs(chg) > 3.5) else 0.0
            dir_adj = (buy_bonus + regime_buy_adj) if side == "BUY" else (short_bonus + regime_short_adj)

            conviction = round(
                max(
                    1.0,
                    min(
                        99.0,
                        base_conviction + ticker_adj + dir_adj + session_adj + chop_adj +
                        rs_adj + bandit_adj + arch_score_adj - critic_penalty
                    )
                ),
                1
            )

            effective_sl_mult = base_atr_sl_mult * arch_sl_adj
            intraday_risk_per_share = max(round(atr * effective_sl_mult, 2), round(ltp * 0.006, 2))
            reward_1_per_share = round(intraday_risk_per_share * (atr_tp_mult / max(effective_sl_mult, 0.3)), 2)
            reward_2_per_share = round(intraday_risk_per_share * ((atr_tp_mult + 1.0) / max(effective_sl_mult, 0.3)), 2)

            entry_price = round(ltp, 2)
            oi_capped = False
            if side == "BUY":
                stop_loss = round(entry_price - intraday_risk_per_share, 2)
                target_1 = round(entry_price + reward_1_per_share, 2)
                target_2 = round(entry_price + reward_2_per_share, 2)
                # Cap Target 1 just below institutional Max Call OI resistance strike
                if max_call_oi > entry_price and target_1 > (max_call_oi * 0.997):
                    capped_t1 = round(max_call_oi * 0.997, 2)
                    if capped_t1 > entry_price * 1.004:
                        target_1 = capped_t1
                        oi_capped = True
                if upper_cp > 0 and target_1 > upper_cp:
                    target_1 = upper_cp
            else:
                stop_loss = round(entry_price + intraday_risk_per_share, 2)
                target_1 = round(entry_price - reward_1_per_share, 2)
                target_2 = round(entry_price - reward_2_per_share, 2)
                # Floor Target 1 just above institutional Max Put OI support strike
                if 0 < max_put_oi < entry_price and target_1 < (max_put_oi * 1.003):
                    capped_t1 = round(max_put_oi * 1.003, 2)
                    if capped_t1 < entry_price * 0.996:
                        target_1 = capped_t1
                        oi_capped = True
                if lower_cp > 0 and target_1 < lower_cp:
                    target_1 = lower_cp

            rr_val = round(abs(target_1 - entry_price) / max(intraday_risk_per_share, 0.01), 2)
            calibrated_p = round(0.5 * posterior_prob + 0.5 * (conviction / 100.0), 4)

            cand_obj = {
                "symbol": sym,
                "action": side,
                "conviction": conviction,
                "calibrated_win_prob": calibrated_p,
                "strategy_arm": strategy_arm,
                "archetype": archetype,
                "clv": clv,
                "delivery_pct": delivery_pct,
                "lower_cp": lower_cp,
                "upper_cp": upper_cp,
                "max_call_oi_strike": max_call_oi,
                "max_put_oi_strike": max_put_oi,
                "oi_capped": oi_capped,
                "mansfield_rs": mansfield_rs,
                "intraday_alpha": intraday_alpha,
                "entry_price": entry_price,
                "target_1": target_1,
                "target_2": target_2,
                "stop_loss": stop_loss,
                "intraday_risk_per_share": intraday_risk_per_share,
                "rr_val": rr_val,
                "rr_ratio": f"1 : {rr_val}",
                "chart_trend": chart_trend,
                "vol_ratio": vol_ratio,
                "rsi": rsi,
                "sentiment": raw_sent,
                "headline": headline,
                "vol_score": round(vol_score, 1),
                "sent_score": round(sent_score, 1),
                "pre_score": round(pre_score, 1),
                "mom_score": round(mom_score, 1),
                "debate_note": "Quantitative Bear Critic passed",
            }

            if vetoed:
                cand_obj["skip_reason"] = f"Bear Critic Veto: {', '.join(red_flags)}"
                skipped_for_shadow.append(cand_obj)
                continue

            if session_blocked or conviction < min_conviction or avail_cash < 500:
                cand_obj["skip_reason"] = (
                    "Low-win-rate session window" if session_blocked else f"Conviction {conviction} < {min_conviction}"
                )
                skipped_for_shadow.append(cand_obj)
                continue

            scored_candidates.append(cand_obj)

        self._record_shadow_candidates(skipped_for_shadow)

        if not scored_candidates:
            return pd.DataFrame()

        scored_candidates.sort(key=lambda x: x["conviction"], reverse=True)

        # Run Gemini Flash Bull-vs-Bear Multi-Agent Debate on top 2 finalists only
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

        # Sector Correlation Guard
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

        # Half-Kelly Criterion Capital Allocation
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

            kelly_frac, _ = self._compute_half_kelly_fraction(
                item["calibrated_win_prob"], item["rr_val"]
            )
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
                "symbol": item["symbol"],
                "sector": sec,
                "market_regime": regime_name,
                "strategy_arm": item["strategy_arm"],
                "archetype": item["archetype"],
                "clv": item["clv"],
                "delivery_pct": item["delivery_pct"],
                "lower_cp": item["lower_cp"],
                "upper_cp": item["upper_cp"],
                "oi_capped": item["oi_capped"],
                "debate_note": item["debate_note"],
                "mansfield_rs": item["mansfield_rs"],
                "intraday_alpha": item["intraday_alpha"],
                "kelly_pct": round(kelly_frac * regime_size_mult * 100.0, 1),
                "action": item["action"],
                "conviction": item["conviction"],
                "qty": qty,
                "capital_used": capital_used,
                "max_loss_inr": round(qty * risk_per_share, 2),
                "expected_profit_inr": round(qty * abs(item["target_1"] - entry_price), 2),
                "entry_price": entry_price,
                "target_1": item["target_1"],
                "target_2": item["target_2"],
                "stop_loss": item["stop_loss"],
                "risk_pct": round((risk_per_share / entry_price) * 100, 2),
                "reward_pct": round((abs(item["target_1"] - entry_price) / entry_price) * 100, 2),
                "rr_ratio": item["rr_ratio"],
                "chart_trend": item["chart_trend"],
                "vol_ratio": item["vol_ratio"],
                "rsi": item["rsi"],
                "sentiment": item["sentiment"],
                "headline": item["headline"],
                "vol_score": item["vol_score"],
                "sent_score": item["sent_score"],
                "pre_score": item["pre_score"],
                "mom_score": item["mom_score"]
            })

        self._save_weights()
        return pd.DataFrame(final_suggestions)

    # ------------------------------------------------------------------
    # 2. AUTONOMOUS ENTRY EXECUTION
    # ------------------------------------------------------------------
    def record_recommendations(self, rec_df, notify_telegram=True):
        if rec_df.empty:
            return 0

        self._recalculate_cash(self.portfolio)
        avail_cash = float(self.portfolio["available_cash"])

        now_ist = self._now_ist()
        today_str = now_ist.strftime("%Y-%m-%d")
        now_time = now_ist.strftime("%I:%M:%S %p")
        session_bucket = self._get_session_bucket(now_ist)

        open_symbols = {item["symbol"] for item in self.ledger if item.get("status") == "OPEN"}
        sector_counts, _ = self._get_open_sector_exposure()

        candidate_syms = rec_df["symbol"].tolist()
        fresh_ticks = MarketDataScanner.fetch_live_intraday_snapshot(candidate_syms)

        added = 0
        for _, row in rec_df.iterrows():
            sym = row["symbol"]
            if sym in open_symbols:
                continue

            sec = row.get("sector") or self._get_stock_sector(sym)
            if sector_counts.get(sec, 0) >= MAX_POSITIONS_PER_SECTOR:
                continue

            old_entry = float(row["entry_price"])
            snap = fresh_ticks.get(sym, {})
            live_tick_p = snap.get("current_price", old_entry)
            if live_tick_p <= 0:
                live_tick_p = old_entry
            price_shift = live_tick_p - old_entry
            entry_p = round(live_tick_p, 2)

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
            arm_tag = row.get("strategy_arm", "MOMENTUM_BREAKOUT")
            arch_tag = row.get("archetype", "MOMENTUM_MOVER")
            mrs_val = float(row.get("mansfield_rs", 0.0))
            kelly_pct = float(row.get("kelly_pct", 15.0))
            deliv_pct = float(row.get("delivery_pct", snap.get("delivery_pct", 0.0)) or 0.0)
            lower_cp = float(row.get("lower_cp", snap.get("lower_cp", 0.0)) or 0.0)
            upper_cp = float(row.get("upper_cp", snap.get("upper_cp", 0.0)) or 0.0)
            debate_note = str(row.get("debate_note", "Passed risk debate"))

            trade_record = {
                "trade_id": f"{today_str}_{sym}_{now_ist.strftime('%H%M')}",
                "date": today_str,
                "timestamp": now_time,
                "entry_iso": now_ist.isoformat(),
                "entry_session": session_bucket,
                "market_regime": regime_tag,
                "strategy_arm": arm_tag,
                "archetype": arch_tag,
                "mansfield_rs": mrs_val,
                "delivery_pct": deliv_pct,
                "lower_cp": lower_cp,
                "upper_cp": upper_cp,
                "debate_note": debate_note,
                "kelly_pct": kelly_pct,
                "sector": sec,
                "symbol": sym,
                "action": row["action"],
                "conviction": float(row["conviction"]),
                "qty": qty,
                "capital_used": cap_needed,
                "entry_price": entry_p,
                "initial_target": t1,
                "target_1": t1,
                "target_2": t2,
                "initial_sl": sl,
                "stop_loss": sl,
                "current_price": entry_p,
                "peak_pnl_inr": 0.0,
                "unrealized_pnl_inr": 0.0,
                "entry_rsi": float(row.get("rsi", 50.0)),
                "entry_sentiment": row.get("sentiment", "NEUTRAL"),
                "chart_trend": row.get("chart_trend", "RANGE_CONSOLIDATION"),
                "status": "OPEN",
                "trade_management_note": f"Deliv: {deliv_pct:.1f}% | Critic: {debate_note}",
                "exit_price": None,
                "exit_time": None,
                "hold_minutes": 0.0,
                "day_high": None,
                "day_low": None,
                "gross_pnl_inr": 0.0,
                "charges_inr": 0.0,
                "pnl_pct": 0.0,
                "pnl_inr": 0.0,
                "mistake_diagnosis": None,
                "factor_scores": {
                    "volume": float(row["vol_score"]),
                    "sentiment": float(row["sent_score"]),
                    "premarket": float(row["pre_score"]),
                    "momentum": float(row["mom_score"])
                },
                "headline": row["headline"]
            }
            self.ledger.append(trade_record)
            open_symbols.add(sym)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            added += 1

            if notify_telegram:
                side_header = "🟢 <b>AI AUTONOMOUS BUY ENTRY</b>" if row["action"] == "BUY" else "🔴 <b>AI AUTONOMOUS SHORT ENTRY</b>"
                entry_ticket = (
                    f"{side_header}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📌 <b>Stock:</b> <code>{sym}</code> ({row['action']}) | <b>Sector:</b> {sec}\n"
                    f"📊 <b>Delivery %:</b> {deliv_pct:.1f}% | <b>Mansfield RS:</b> {mrs_val:+.2f}%\n"
                    f"🎰 <b>Arm:</b> <code>{arm_tag}</code> | <b>TRA:</b> <code>{arch_tag}</code>\n"
                    f"📦 <b>Quantity:</b> <b>{qty} Shares</b> @ <b>₹{entry_p}</b> (Half-Kelly: {kelly_pct:.1f}%)\n"
                    f"💰 <b>Wallet Used:</b> ₹{cap_needed:,.2f} <i>(Cash Left: ₹{avail_cash:,.2f})</i>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>Target 1:</b> ₹{t1} <i>(+₹{exp_profit:,.0f})</i> | 🚀 <b>Target 2:</b> ₹{t2}\n"
                    f"🛑 <b>Initial Stop-Loss:</b> ₹{sl} <i>(-₹{max_loss:,.0f} risk)</i>\n"
                    f"⚖️ <b>Risk Debate:</b> <i>{debate_note}</i>\n"
                    f"📰 <b>Catalyst ({row['sentiment']}):</b> {row['headline']}"
                )
                self.notifier.send_message(entry_ticket)

        if added > 0:
            self._save_ledger()
            self._save_portfolio()
        return added

    # ------------------------------------------------------------------
    # 3. LIVE MANAGER + CIRCUIT LOCKOUT + SLIPPAGE + AFTER-TAX SETTLEMENT
    # ------------------------------------------------------------------
    def manage_live_positions(self, notify_telegram=True):
        self._evaluate_shadow_trades()
        self._check_page_hinkley_drift()

        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        if not open_trades:
            return []

        regime_info = self._detect_market_regime()
        nifty_chg_pct = float(regime_info.get("nifty_chg_pct", 0.0))
        symbols = [t["symbol"] for t in open_trades]
        snapshots = MarketDataScanner.fetch_live_intraday_snapshot(symbols)
        actions_taken = []
        now_ist = self._now_ist()
        today_str = now_ist.strftime("%Y-%m-%d")
        now_time = now_ist.strftime("%I:%M:%S %p")

        giveback_limit_pct = float(self.weights.get("max_profit_giveback_pct", 45.0))
        max_hold_mins = float(self.weights.get("learned_max_hold_mins", 180.0))

        for trade in open_trades:
            sym = trade["symbol"]
            snap = snapshots.get(sym)
            if not snap:
                continue

            ltp = snap["current_price"]
            d_high = snap["day_high"]
            d_low = snap["day_low"]
            vwap = snap["intraday_vwap"]
            lower_cp = float(snap.get("lower_cp", trade.get("lower_cp", 0.0)) or 0.0)
            upper_cp = float(snap.get("upper_cp", trade.get("upper_cp", 0.0)) or 0.0)

            entry = float(trade["entry_price"])
            t1 = float(trade["target_1"])
            t2 = float(trade["target_2"])
            sl = float(trade["stop_loss"])
            init_sl = float(trade.get("initial_sl", sl))
            qty = int(trade.get("qty", 1))
            side = trade["action"]
            arch_used = trade.get("archetype", "MOMENTUM_MOVER")

            risk_unit = max(abs(entry - init_sl), entry * 0.005)
            hold_mins = round(self._calculate_hold_minutes(trade), 1)
            trade["hold_minutes"] = hold_mins
            trade["current_price"] = ltp
            trade["day_high"] = d_high
            trade["day_low"] = d_low

            unrealized_per_share = (ltp - entry) if side == "BUY" else (entry - ltp)
            unrealized_inr = round(unrealized_per_share * qty, 2)
            trade["unrealized_pnl_inr"] = unrealized_inr
            r_multiple = unrealized_per_share / risk_unit

            prev_peak = float(trade.get("peak_pnl_inr", 0.0))
            peak_pnl = max(prev_peak, unrealized_inr)
            trade["peak_pnl_inr"] = round(peak_pnl, 2)

            # NSE LOWER/UPPER CIRCUIT LOCKOUT SIMULATION (Realistic illiquidity handling)
            if side == "BUY" and lower_cp > 0 and ltp <= lower_cp:
                trade["trade_management_note"] = f"LOWER CIRCUIT LOCKED @ ₹{lower_cp} — Zero buyers; SL fill queued for unlock."
                continue
            if side.startswith("SELL") and upper_cp > 0 and ltp >= upper_cp:
                trade["trade_management_note"] = f"UPPER CIRCUIT LOCKED @ ₹{upper_cp} — Zero sellers; cover queued for unlock."
                continue

            stk_chg_pct = ((ltp - entry) / entry) * 100.0 if entry > 0 else 0.0
            live_alpha = stk_chg_pct - nifty_chg_pct

            live_news = None
            emotion_flipped = False
            if unrealized_inr < 0 or (peak_pnl > 0 and unrealized_inr < peak_pnl * 0.6):
                live_news = self._analyze_live_news_emotion(sym)
                if side == "BUY" and live_news["emotion"] == "NEGATIVE":
                    emotion_flipped = True
                elif side.startswith("SELL") and live_news["emotion"] == "POSITIVE":
                    emotion_flipped = True

            overnight_exit, overnight_reason = self._check_overnight_hold_eligibility(trade, ltp, regime_info)

            exit_triggered = False
            exit_status = ""
            raw_exit_price = ltp
            exit_note = ""

            if emotion_flipped:
                exit_triggered = True
                exit_status = "EMOTION SHIFT EXIT"
                raw_exit_price = ltp
                exit_note = f"News emotion flipped to {live_news['emotion']}: {live_news['top_headline']}"

            elif (
                side == "BUY"
                and trade.get("entry_sentiment") == "POSITIVE"
                and hold_mins >= 20.0
                and ltp < vwap
                and live_alpha < -0.8
                and unrealized_inr < 0
            ):
                exit_triggered = True
                exit_status = "DISTRIBUTION TRAP EXIT"
                raw_exit_price = ltp
                exit_note = f"Bullish news failed to hold VWAP (₹{vwap}) with {live_alpha:+.2f}% Nifty alpha."

            elif overnight_exit:
                exit_triggered = True
                exit_status = "OVERNIGHT RISK GUARD EXIT"
                raw_exit_price = ltp
                exit_note = overnight_reason

            elif (side == "BUY" and (ltp >= t1 or d_high >= t1)) or (side.startswith("SELL") and (ltp <= t1 or d_low <= t1)):
                strong_momentum = (ltp >= t1 and ltp > vwap and t1 < t2) if side == "BUY" else (ltp <= t1 and ltp < vwap and t1 > t2)
                if strong_momentum:
                    new_sl = round(entry + (0.85 * risk_unit), 2) if side == "BUY" else round(entry - (0.85 * risk_unit), 2)
                    trade["stop_loss"] = max(sl, new_sl) if side == "BUY" else min(sl, new_sl)
                    trade["target_1"] = t2
                    trade["trade_management_note"] = f"AI extended target to ₹{t2} & locked SL at ₹{trade['stop_loss']}."
                    actions_taken.append(f"🚀 <b>TARGET EXTENDED ({sym}):</b> New Target ₹{t2} | Locked SL ₹{trade['stop_loss']}")
                else:
                    exit_triggered = True
                    exit_status = "WIN (TARGET HIT)"
                    raw_exit_price = t1
                    exit_note = "AI booked full target profit."

            elif (side == "BUY" and ltp <= sl) or (side.startswith("SELL") and ltp >= sl):
                exit_triggered = True
                is_be = (sl >= entry) if side == "BUY" else (sl <= entry)
                exit_status = "BREAKEVEN EXIT" if is_be else "LOSS (SL HIT)"
                raw_exit_price = sl
                exit_note = "Exited at dynamic Stop-Loss."

            elif peak_pnl >= (risk_unit * qty * 0.75) and unrealized_inr > 0:
                giveback_pct = ((peak_pnl - unrealized_inr) / peak_pnl) * 100.0
                if giveback_pct >= giveback_limit_pct:
                    exit_triggered = True
                    exit_status = "WIN (PROFIT LOCKED)"
                    raw_exit_price = ltp
                    exit_note = f"AI locked profit after {giveback_pct:.0f}% pullback from peak ₹{peak_pnl:,.0f}."

            elif r_multiple <= -0.65 and ((side == "BUY" and ltp < vwap) or (side.startswith("SELL") and ltp > vwap)):
                exit_triggered = True
                exit_status = "EARLY CUT (LOSS MINIMIZED)"
                raw_exit_price = ltp
                exit_note = f"Cut early against VWAP (₹{vwap}) to protect capital."

            elif hold_mins >= max_hold_mins and unrealized_inr <= 0:
                exit_triggered = True
                exit_status = "STAGNATION EXIT (AI TIMING)"
                raw_exit_price = ltp
                exit_note = f"Exited after {hold_mins:.0f}m without momentum follow-through."

            elif r_multiple >= 0.8 and ((side == "BUY" and sl < entry) or (side.startswith("SELL") and sl > entry)):
                trade["stop_loss"] = entry
                trade["trade_management_note"] = f"SL trailed to breakeven ₹{entry}."

            if exit_triggered:
                # Apply realistic TRA Archetype liquidity slippage on exit
                slip_pct = self._get_archetype_slippage_pct(arch_used) / 100.0
                if side == "BUY":
                    exit_price = round(raw_exit_price * (1.0 - slip_pct), 2)
                else:
                    exit_price = round(raw_exit_price * (1.0 + slip_pct), 2)

                is_overnight = trade.get("date") != today_str
                gross_pnl_inr = ((exit_price - entry) * qty) if side == "BUY" else ((entry - exit_price) * qty)
                charges_inr = self._calculate_nse_charges(side, qty, entry, exit_price, is_overnight=is_overnight)
                net_pnl_inr = round(gross_pnl_inr - charges_inr, 2)
                cap_used = max(float(trade.get("capital_used", entry * qty)), 1.0)
                net_pnl_pct = round((net_pnl_inr / cap_used) * 100.0, 2)

                trade["status"] = exit_status
                trade["exit_price"] = exit_price
                trade["exit_time"] = f"{now_time} IST"
                trade["unrealized_pnl_inr"] = 0.0
                trade["gross_pnl_inr"] = round(gross_pnl_inr, 2)
                trade["charges_inr"] = charges_inr
                trade["pnl_pct"] = net_pnl_pct
                trade["pnl_inr"] = net_pnl_inr
                trade["trade_management_note"] = f"{exit_note} (After ₹{charges_inr:.2f} taxes & slippage)"

                arm_used = trade.get("strategy_arm", "MOMENTUM_BREAKOUT")
                self._update_bandit_arm(arm_used, net_pnl_inr, risk_unit * qty, weight_scale=1.0, count_as_real=True)
                self._update_archetype_memory(arch_used, net_pnl_inr, hit_sl=("SL HIT" in exit_status), weight_scale=1.0)

                self._recalculate_cash(self.portfolio)
                lesson = self._research_and_learn_from_trade(trade, live_news)

                msg = f"{'✅' if net_pnl_inr >= 0 else '🛑'} <b>AI CLOSED {sym} ({exit_status}):</b> ₹{exit_price} | Net P&L: <b>₹{net_pnl_inr:+,.0f}</b>"
                actions_taken.append(msg)

                if notify_telegram:
                    header = "✅ <b>AI TRADE COMPLETED — NET PROFIT BOOKED</b>" if net_pnl_inr >= 0 else "🔴 <b>AI TRADE COMPLETED — CUT & RESEARCHED</b>"
                    tg_ticket = (
                        f"{header}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> ({side}) | <b>Arm:</b> <code>{arm_used}</code>\n"
                        f"🏷️ <b>AI Decision:</b> {exit_status} ({hold_mins:.0f} mins)\n"
                        f"🔢 <b>Entry:</b> ₹{entry} ➔ <b>Exit (w/ slippage):</b> <b>₹{exit_price}</b>\n"
                        f"🧾 <b>Gross P&L:</b> ₹{gross_pnl_inr:+,.2f} | <b>STT/Taxes/Fees:</b> -₹{charges_inr:.2f}\n"
                        f"💰 <b>Net Realized P&L:</b> <b>₹{net_pnl_inr:+,.2f} ({net_pnl_pct:+.2f}%)</b>\n"
                        f"🏦 <b>Updated Wallet Equity:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>\n"
                        f"🧠 <b>Post-Trade AI Research & Lesson:</b>\n<i>{lesson}</i>"
                    )
                    self.notifier.send_message(tg_ticket)

        self._save_ledger()
        self._save_portfolio()
        self._save_weights()
        return actions_taken

    # ------------------------------------------------------------------
    # 4. PER-TRADE RESEARCH, MISTAKE AUTOPSY & CONTINUOUS LEARNING
    # ------------------------------------------------------------------
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
                reasons.append(f"gave back +₹{peak_pnl:,.0f} peak profit before tightening giveback cutoff")
            if side == "BUY" and entry_rsi > 68:
                reasons.append(f"overbought entry RSI {entry_rsi}")
            if not reasons:
                reasons.append(f"{arm_used} on {arch_used} underperformed in {session_used} [{regime_used}]")

            lesson = (
                f"MISTAKE AUTOPSY ({sym} {side}, ₹{pnl_inr:+,.0f} net): Failed due to {', '.join(reasons)}. "
                f"Action: Penalized {arm_used} Beta posterior & {arch_used} profile."
            )
            trade["mistake_diagnosis"] = lesson
            self.portfolio["mistakes_identified"] = (
                [lesson] + self.portfolio.get("mistakes_identified", [])
            )[:20]
        else:
            lesson = (
                f"WINNING EDGE ({sym} {side}, +₹{pnl_inr:,.0f} net in {hold_mins:.0f}m): "
                f"Reinforced {arm_used} Alpha posterior on {arch_used} in {session_used} [{regime_used}]."
            )
            trade["mistake_diagnosis"] = lesson

        self.weights.setdefault("trade_lessons_log", []).insert(0, {
            "timestamp": self._now_ist().strftime("%Y-%m-%d %I:%M %p IST"),
            "symbol": sym,
            "pnl_inr": round(pnl_inr, 2),
            "session": session_used,
            "regime": regime_used,
            "arm": arm_used,
            "archetype": arch_used,
            "emotion_at_exit": latest_emotion,
            "lesson": lesson
        })
        self.weights["trade_lessons_log"] = self.weights["trade_lessons_log"][:50]

        self._optimize_weights_from_history(force_recalibrate=True)
        return lesson

    def evaluate_open_trades_and_learn(self, notify_telegram=True):
        self.manage_live_positions(notify_telegram=notify_telegram)
        learning_summary = self._optimize_weights_from_history(force_recalibrate=True)

        today_str = self._now_ist().strftime("%Y-%m-%d")
        today_pnl = sum(
            float(t.get("pnl_inr", 0))
            for t in self.ledger
            if t.get("date") == today_str and t.get("status") != "OPEN"
        )
        self._recalculate_cash(self.portfolio)
        eq_log = self.portfolio.get("daily_equity_log", [])
        if eq_log and eq_log[-1].get("date") == today_str:
            eq_log[-1]["equity"] = self.portfolio["current_equity"]
            eq_log[-1]["pnl"] = round(today_pnl, 2)
        else:
            eq_log.append({
                "date": today_str,
                "equity": self.portfolio["current_equity"],
                "pnl": round(today_pnl, 2)
            })
        self.portfolio["daily_equity_log"] = eq_log[-60:]
        self._save_portfolio()

        open_count = sum(1 for t in self.ledger if t.get("status") == "OPEN")
        return {
            "evaluated": len(self.ledger) - open_count,
            "open_positions": open_count,
            "message": learning_summary
        }

    def _optimize_weights_from_history(self, force_recalibrate=False):
        settled = [t for t in self.ledger if t.get("status") != "OPEN" and "factor_scores" in t]
        if not settled:
            return "No settled trades yet to train the AI."

        if not force_recalibrate and len(settled) == int(self.weights.get("last_evaluated_count", 0)):
            return self.weights.get("learning_notes", "Model up to date.")

        base = {"volume": 0.28, "sentiment": 0.28, "premarket": 0.22, "momentum": 0.22}
        adjustments = {"volume": 0.0, "sentiment": 0.0, "premarket": 0.0, "momentum": 0.0}
        recent = settled[-50:]

        wins_count = 0
        sl_hits = 0
        target_hits = 0
        giveback_losses = 0
        winning_convictions = []
        losing_convictions = []
        losing_buy_rsis = []
        winning_hold_times = []

        for t in recent:
            pnl = float(t.get("pnl_pct", 0.0))
            pnl_inr = float(t.get("pnl_inr", 0.0))
            peak_inr = float(t.get("peak_pnl_inr", 0.0))
            status_str = str(t.get("status", ""))
            conv = float(t.get("conviction", 60.0))
            hold_m = float(t.get("hold_minutes", 0.0))

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
            scores = t["factor_scores"]
            for factor_key in adjustments:
                factor_strength = (float(scores.get(factor_key, 50.0)) - 50.0) / 100.0
                adjustments[factor_key] += direction * magnitude * factor_strength * 0.018

        raw_vol = max(0.12, min(0.50, base["volume"] + adjustments["volume"]))
        raw_sent = max(0.12, min(0.50, base["sentiment"] + adjustments["sentiment"]))
        raw_pre = max(0.12, min(0.50, base["premarket"] + adjustments["premarket"]))
        raw_mom = max(0.12, min(0.50, base["momentum"] + adjustments["momentum"]))

        total = raw_vol + raw_sent + raw_pre + raw_mom
        self.weights["volume_weight"] = round(raw_vol / total, 3)
        self.weights["sentiment_weight"] = round(raw_sent / total, 3)
        self.weights["premarket_weight"] = round(raw_pre / total, 3)
        self.weights["momentum_weight"] = round(1.0 - (
            self.weights["volume_weight"] +
            self.weights["sentiment_weight"] +
            self.weights["premarket_weight"]
        ), 3)

        ticker_stats = {}
        for t in settled:
            sym = t["symbol"]
            pnl_inr = float(t.get("pnl_inr", 0.0))
            if sym not in ticker_stats:
                ticker_stats[sym] = {"trades": 0, "wins": 0, "losses": 0, "net_pnl_inr": 0.0}
            ticker_stats[sym]["trades"] += 1
            if pnl_inr > 0:
                ticker_stats[sym]["wins"] += 1
            else:
                ticker_stats[sym]["losses"] += 1
            ticker_stats[sym]["net_pnl_inr"] = round(ticker_stats[sym]["net_pnl_inr"] + pnl_inr, 2)

        blacklisted, preferred = [], []
        for sym, st_data in ticker_stats.items():
            st_data["win_rate"] = round((st_data["wins"] / st_data["trades"]) * 100.0, 1)
            if st_data["trades"] >= 2 and st_data["win_rate"] < 35.0 and st_data["net_pnl_inr"] < -500:
                blacklisted.append(sym)
            elif st_data["trades"] >= 1 and st_data["win_rate"] >= 65.0 and st_data["net_pnl_inr"] > 300:
                preferred.append(sym)

        self.weights["ticker_memory"] = ticker_stats
        self.weights["blacklisted_tickers"] = blacklisted
        self.weights["preferred_tickers"] = preferred

        session_stats = {}
        for t in settled:
            sess = t.get("entry_session", "MORNING_TREND (09:45-11:30)")
            pnl_inr = float(t.get("pnl_inr", 0.0))
            if sess not in session_stats:
                session_stats[sess] = {"trades": 0, "wins": 0, "net_pnl_inr": 0.0, "score_adj": 0.0}
            session_stats[sess]["trades"] += 1
            if pnl_inr > 0:
                session_stats[sess]["wins"] += 1
            session_stats[sess]["net_pnl_inr"] = round(session_stats[sess]["net_pnl_inr"] + pnl_inr, 2)

        for sess, s_data in session_stats.items():
            wr = (s_data["wins"] / s_data["trades"]) * 100.0
            s_data["win_rate"] = round(wr, 1)
            s_data["score_adj"] = round(max(-12.0, min(8.0, (wr - 50.0) * 0.25)), 1) if s_data["trades"] >= 2 else 0.0

        self.weights["session_memory"] = session_stats

        buy_trades = [t for t in recent if t.get("action") == "BUY"]
        short_trades = [t for t in recent if str(t.get("action", "")).startswith("SELL")]

        buy_wr = (sum(1 for t in buy_trades if float(t.get("pnl_inr", 0)) > 0) / len(buy_trades) * 100.0) if buy_trades else 50.0
        short_wr = (sum(1 for t in short_trades if float(t.get("pnl_inr", 0)) > 0) / len(short_trades) * 100.0) if short_trades else 50.0

        self.weights["buy_bias_bonus"] = round(max(-6.0, min(6.0, (buy_wr - 50.0) * 0.15)), 2) if len(buy_trades) >= 2 else 0.0
        self.weights["short_bias_bonus"] = round(max(-6.0, min(6.0, (short_wr - 50.0) * 0.15)), 2) if len(short_trades) >= 2 else 0.0

        sl_rate = (sl_hits / len(recent)) * 100.0
        tp_rate = (target_hits / len(recent)) * 100.0

        if sl_rate > 40.0:
            self.weights["atr_sl_multiplier"] = round(min(0.72, 0.55 + ((sl_rate - 40.0) * 0.004)), 2)
        elif sl_rate < 20.0:
            self.weights["atr_sl_multiplier"] = 0.52

        if tp_rate >= 45.0:
            self.weights["atr_tp_multiplier"] = 1.80
        elif tp_rate < 20.0 and len(recent) >= 5:
            self.weights["atr_tp_multiplier"] = 1.40
        else:
            self.weights["atr_tp_multiplier"] = 1.60

        if giveback_losses >= 2:
            self.weights["max_profit_giveback_pct"] = max(25.0, round(45.0 - (giveback_losses * 5.0), 1))

        if winning_hold_times:
            avg_win_hold = sum(winning_hold_times) / len(winning_hold_times)
            self.weights["learned_max_hold_mins"] = round(max(45.0, min(240.0, avg_win_hold * 1.5)), 1)

        if losing_convictions and winning_convictions:
            avg_loss_conv = sum(losing_convictions) / len(losing_convictions)
            avg_win_conv = sum(winning_convictions) / len(winning_convictions)
            if avg_win_conv > avg_loss_conv:
                learned_cutoff = (avg_loss_conv + avg_win_conv) / 2.0
                self.weights["min_conviction_threshold"] = round(max(50.0, min(68.0, learned_cutoff)), 1)

        if losing_buy_rsis and len([r for r in losing_buy_rsis if r >= 67.0]) >= 2:
            self.weights["buy_rsi_ceiling"] = 66.0

        recent_win_rate = (wins_count / len(recent)) * 100.0
        total_contrib = float(self.portfolio.get("total_capital_contributed", INITIAL_CAPITAL))
        if recent_win_rate >= 60.0 and self.portfolio["current_equity"] >= total_contrib:
            self.weights["risk_per_trade_pct"] = min(1.5, round(1.0 + ((recent_win_rate - 50.0) * 0.015), 2))
        elif recent_win_rate < 45.0 or self.portfolio["current_equity"] < total_contrib:
            self.weights["risk_per_trade_pct"] = max(0.6, round(1.0 - ((50.0 - recent_win_rate) * 0.015), 2))

        self.weights["generation"] = len(settled) + 1
        self.weights["last_evaluated_count"] = len(settled)
        self.weights["last_updated"] = self._now_ist().strftime("%Y-%m-%d %I:%M %p IST")

        arms = self.weights.get("bandit_arms", self._default_bandit_arms())
        top_arm = max(
            arms.items(),
            key=lambda kv: float(kv[1].get("alpha", 2.0)) / (float(kv[1].get("alpha", 2.0)) + float(kv[1].get("beta", 2.0)))
        )
        top_arm_prob = float(top_arm[1]["alpha"]) / (float(top_arm[1]["alpha"]) + float(top_arm[1]["beta"]))
        shadow_n = int(self.weights.get("shadow_samples_learned", 0))

        note = (
            f"Gen #{self.weights['generation']} ({len(settled)} real + {shadow_n} shadow setups): "
            f"Top Arm={top_arm[0]} ({top_arm_prob:.0%}), "
            f"Min Conviction={self.weights['min_conviction_threshold']}"
        )
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

        return {
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