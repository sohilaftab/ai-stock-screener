import os
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
from market_data import MarketDataScanner
from telegram_notifier import TelegramNotifier

IST = ZoneInfo("Asia/Kolkata")

WEIGHTS_FILE = "model_weights.json"
LEDGER_FILE = "trade_history.json"
PORTFOLIO_FILE = "portfolio_state.json"
MORNING_WATCHLIST_FILE = "morning_watchlist.json"

INITIAL_CAPITAL = 150000.0

# Lightweight financial emotion lexicon for real-time news shift detection
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
        self.portfolio = self._load_portfolio()

    @staticmethod
    def _now_ist():
        return datetime.now(IST)

    @staticmethod
    def _get_session_bucket(dt=None):
        """Categorizes the current IST time so the AI learns which market windows are profitable."""
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
        """Computes how many minutes a trade has been open using IST timestamps."""
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
    # REAL-TIME NEWS & EMOTION RESEARCH FOR ACTIVE / SETTLED STOCKS
    # ------------------------------------------------------------------
    def _analyze_live_news_emotion(self, sym):
        """
        Fetches latest Google News RSS headlines for an NSE stock to detect
        real-time market emotion shifts during active trades and post-trade research.
        """
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
            "max_profit_giveback_pct": 45.0,   # Learned trailing giveback cutoff once in profit
            "learned_max_hold_mins": 180.0,    # Learned optimal holding horizon before stagnation exit
            "session_memory": {},              # Learned win rate & P&L per time-of-day window
            "ticker_memory": {},               # Per-stock win/loss & P&L memory
            "blacklisted_tickers": [],
            "preferred_tickers": [],
            "trade_lessons_log": [],           # Structured research lessons from every closed trade
            "generation": 1,
            "last_evaluated_count": 0,
            "last_updated": self._now_ist().strftime("%Y-%m-%d %H:%M"),
            "learning_notes": "Initialized 6-Layer Autonomous Emotion & Timing Brain."
        }
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE, "r") as f:
                    default_weights.update(json.load(f))
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
        """
        STRICT CLOSED-WALLET ACCOUNTING WITH USER-ONLY MANUAL DEPOSITS:
        - Base seed is fixed at Rs 1,50,000.00.
        - Manual deposits come ONLY from explicit user actions (`deposit_history`).
        - The AI NEVER adds capital on its own.
        """
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
        port_dict["total_realized_pnl"] = round(realized, 2)
        port_dict["current_equity"] = round(max(0.0, total_contributed + realized), 2)
        port_dict["allocated_capital"] = round(open_allocated, 2)
        port_dict["available_cash"] = round(max(0.0, port_dict["current_equity"] - open_allocated), 2)

    def _save_portfolio(self):
        self._recalculate_cash(self.portfolio)
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(self.portfolio, f, indent=2)

    # ------------------------------------------------------------------
    # USER-ONLY MANUAL CAPITAL ADDITION (NEVER CALLED BY AI)
    # ------------------------------------------------------------------
    def add_manual_funds(self, amount, note="Manual user deposit", notify_telegram=True):
        """
        Adds user-deposited funds into the portfolio wallet.
        Called ONLY when the human user explicitly submits a deposit in the web dashboard.
        """
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
    # 1. ADAPTIVE OPPORTUNITY & TIMING SCORER
    # ------------------------------------------------------------------
    def generate_intraday_recommendations(self, enriched_df, premarket_df=None, top_k=5):
        """
        Evaluates live market data, news emotion, and learned session-timing windows
        to decide if now is a suitable moment to enter a trade.
        """
        if enriched_df.empty:
            return pd.DataFrame()

        self._recalculate_cash(self.portfolio)
        current_equity = float(self.portfolio["current_equity"])
        avail_cash = float(self.portfolio["available_cash"])
        if avail_cash < 500:
            return pd.DataFrame()

        # Check what the AI has learned about the CURRENT time-of-day window
        current_session = self._get_session_bucket()
        session_mem = self.weights.get("session_memory", {}).get(current_session, {})
        session_adj = float(session_mem.get("score_adj", 0.0))

        # If AI learned this time window has <30% win rate over >=4 trades, stay out unless conviction is extraordinary
        if session_mem.get("trades", 0) >= 4 and session_mem.get("win_rate", 50.0) < 30.0:
            print(f"[AI TIMING GUARD] Skipping entries during low-win-rate window: {current_session}")
            return pd.DataFrame()

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

        atr_sl_mult = float(self.weights.get("atr_sl_multiplier", 0.55))
        atr_tp_mult = float(self.weights.get("atr_tp_multiplier", 1.60))
        buy_rsi_max = float(self.weights.get("buy_rsi_ceiling", 72.0))
        short_rsi_min = float(self.weights.get("short_rsi_floor", 28.0))
        min_conviction = float(self.weights.get("min_conviction_threshold", 52.0))
        buy_bonus = float(self.weights.get("buy_bias_bonus", 0.0))
        short_bonus = float(self.weights.get("short_bias_bonus", 0.0))

        ticker_memory = self.weights.get("ticker_memory", {})
        blacklist = set(self.weights.get("blacklisted_tickers", []))

        scored_candidates = []
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
            chart_trend = row.get("chart_trend", "RANGE_CONSOLIDATION")

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

            vol_score = min(100.0, (vol_ratio / 2.8) * 100.0)

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
                if 50.0 <= rsi <= buy_rsi_max:
                    rsi_bonus = 15.0
                elif rsi > buy_rsi_max:
                    rsi_bonus = -15.0
                else:
                    rsi_bonus = 0.0
                mom_score = max(10.0, min(100.0, 45.0 + trend_bonus + vwap_bonus + rsi_bonus))
            else:
                trend_bonus = 25.0 if chart_trend == "BEARISH_DOWNTREND" else 0.0
                vwap_bonus = 15.0 if ltp <= vwap else -10.0
                if short_rsi_min <= rsi <= 50.0:
                    rsi_bonus = 15.0
                elif rsi < short_rsi_min:
                    rsi_bonus = -15.0
                else:
                    rsi_bonus = 0.0
                mom_score = max(10.0, min(100.0, 45.0 + trend_bonus + vwap_bonus + rsi_bonus))

            base_conviction = (
                (w_vol * vol_score) +
                (w_sent * sent_score) +
                (w_pre * pre_score) +
                (w_mom * mom_score)
            )

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

            dir_adj = buy_bonus if side == "BUY" else short_bonus
            conviction = round(max(1.0, min(99.0, base_conviction + ticker_adj + dir_adj + session_adj)), 1)

            intraday_risk_per_share = max(round(atr * atr_sl_mult, 2), round(ltp * 0.006, 2))
            reward_1_per_share = round(intraday_risk_per_share * (atr_tp_mult / max(atr_sl_mult, 0.3)), 2)
            reward_2_per_share = round(intraday_risk_per_share * ((atr_tp_mult + 1.0) / max(atr_sl_mult, 0.3)), 2)

            entry_price = round(ltp, 2)
            if side == "BUY":
                stop_loss = round(entry_price - intraday_risk_per_share, 2)
                target_1 = round(entry_price + reward_1_per_share, 2)
                target_2 = round(entry_price + reward_2_per_share, 2)
            else:
                stop_loss = round(entry_price + intraday_risk_per_share, 2)
                target_1 = round(entry_price - reward_1_per_share, 2)
                target_2 = round(entry_price - reward_2_per_share, 2)

            rr_val = round(reward_1_per_share / max(intraday_risk_per_share, 0.01), 2)

            scored_candidates.append({
                "symbol": sym,
                "action": side,
                "conviction": conviction,
                "entry_price": entry_price,
                "target_1": target_1,
                "target_2": target_2,
                "stop_loss": stop_loss,
                "intraday_risk_per_share": intraday_risk_per_share,
                "rr_ratio": f"1 : {rr_val}",
                "chart_trend": chart_trend,
                "vol_ratio": vol_ratio,
                "rsi": rsi,
                "sentiment": raw_sent,
                "headline": headline,
                "vol_score": round(vol_score, 1),
                "sent_score": round(sent_score, 1),
                "pre_score": round(pre_score, 1),
                "mom_score": round(mom_score, 1)
            })

        if not scored_candidates:
            return pd.DataFrame()

        scored_candidates.sort(key=lambda x: x["conviction"], reverse=True)

        # AI only takes setups that genuinely pass its learned conviction threshold (no forced entries)
        qualified = [c for c in scored_candidates if c["conviction"] >= min_conviction]
        if not qualified:
            print(f"[AI SELECTIVITY] No setups met learned min conviction ({min_conviction}). Waiting for better opportunity.")
            return pd.DataFrame()

        top_candidates = qualified[:top_k]

        remaining_wallet_cash = avail_cash
        risk_pct_allowed = float(self.weights.get("risk_per_trade_pct", 1.0)) / 100.0
        max_risk_inr_per_trade = current_equity * risk_pct_allowed
        total_conviction = sum(c["conviction"] for c in top_candidates) or 1.0

        final_suggestions = []
        for idx, item in enumerate(top_candidates):
            entry_price = item["entry_price"]
            risk_per_share = item["intraday_risk_per_share"]

            if remaining_wallet_cash < entry_price:
                continue

            slots_left = len(top_candidates) - idx
            equal_slot_share = remaining_wallet_cash / max(slots_left, 1)
            conviction_weight = (item["conviction"] / (total_conviction / len(top_candidates)))
            slot_budget = min(remaining_wallet_cash, equal_slot_share * min(1.15, max(0.75, conviction_weight)))

            qty_by_cash = int(slot_budget // entry_price)
            qty_by_risk = int(max_risk_inr_per_trade // max(risk_per_share, 0.5))

            qty = max(1, min(qty_by_cash, max(qty_by_risk, int((equal_slot_share * 0.85) // entry_price))))
            if qty * entry_price > remaining_wallet_cash:
                qty = int(remaining_wallet_cash // entry_price)

            if qty <= 0:
                continue

            capital_used = round(qty * entry_price, 2)
            remaining_wallet_cash = round(remaining_wallet_cash - capital_used, 2)

            final_suggestions.append({
                "symbol": item["symbol"],
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

        return pd.DataFrame(final_suggestions)

    # ------------------------------------------------------------------
    # 2. AUTONOMOUS ENTRY EXECUTION
    # ------------------------------------------------------------------
    def record_recommendations(self, rec_df, notify_telegram=True):
        """
        Executes virtual trades whenever the AI identifies high-conviction setups
        inside a favorable learned market window.
        """
        if rec_df.empty:
            return 0

        self._recalculate_cash(self.portfolio)
        avail_cash = float(self.portfolio["available_cash"])

        now_ist = self._now_ist()
        today_str = now_ist.strftime("%Y-%m-%d")
        now_time = now_ist.strftime("%I:%M:%S %p")
        session_bucket = self._get_session_bucket(now_ist)

        open_symbols = {item["symbol"] for item in self.ledger if item.get("status") == "OPEN"}

        candidate_syms = rec_df["symbol"].tolist()
        fresh_ticks = MarketDataScanner.fetch_live_intraday_snapshot(candidate_syms)

        added = 0
        for _, row in rec_df.iterrows():
            sym = row["symbol"]
            if sym in open_symbols:
                continue

            old_entry = float(row["entry_price"])
            live_tick_p = fresh_ticks.get(sym, {}).get("current_price", old_entry)
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

            trade_record = {
                "trade_id": f"{today_str}_{sym}_{now_ist.strftime('%H%M')}",
                "date": today_str,
                "timestamp": now_time,
                "entry_iso": now_ist.isoformat(),
                "entry_session": session_bucket,
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
                "trade_management_note": f"AI entered during {session_bucket} (₹{avail_cash:,.0f} cash left).",
                "exit_price": None,
                "exit_time": None,
                "hold_minutes": 0.0,
                "day_high": None,
                "day_low": None,
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
            added += 1

            if notify_telegram:
                side_header = "🟢 <b>AI AUTONOMOUS BUY ENTRY</b>" if row["action"] == "BUY" else "🔴 <b>AI AUTONOMOUS SHORT ENTRY</b>"
                entry_ticket = (
                    f"{side_header}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📌 <b>Stock:</b> <code>{sym}</code> ({row['action']})\n"
                    f"🕒 <b>AI Chosen Window:</b> {now_time} IST | <code>{session_bucket}</code>\n"
                    f"🧠 <b>AI Conviction:</b> {row['conviction']}/100 | Chart: <code>{row['chart_trend']}</code>\n"
                    f"📦 <b>Quantity:</b> <b>{qty} Shares</b> @ <b>₹{entry_p}</b>\n"
                    f"💰 <b>Wallet Used:</b> ₹{cap_needed:,.2f} <i>(Cash Left: ₹{avail_cash:,.2f})</i>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>Target 1:</b> ₹{t1} <i>(+₹{exp_profit:,.0f})</i> | 🚀 <b>Target 2:</b> ₹{t2}\n"
                    f"🛑 <b>Initial Stop-Loss:</b> ₹{sl} <i>(-₹{max_loss:,.0f} risk)</i>\n"
                    f"📰 <b>Catalyst ({row['sentiment']}):</b> {row['headline']}"
                )
                self.notifier.send_message(entry_ticket)

        if added > 0:
            self._save_ledger()
            self._save_portfolio()
        return added

    # ------------------------------------------------------------------
    # 3. DYNAMIC EMOTION, MOMENTUM & GIVEBACK CUTOFF ENGINE
    # ------------------------------------------------------------------
    def manage_live_positions(self, notify_telegram=True):
        """
        Continuously evaluates live price action, peak profit giveback, hold decay,
        and breaking news emotion shifts to decide autonomously when to exit any trade.
        Immediately researches and learns from every closed trade.
        """
        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        if not open_trades:
            return []

        symbols = [t["symbol"] for t in open_trades]
        snapshots = MarketDataScanner.fetch_live_intraday_snapshot(symbols)
        actions_taken = []
        now_ist = self._now_ist()
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

            entry = float(trade["entry_price"])
            t1 = float(trade["target_1"])
            t2 = float(trade["target_2"])
            sl = float(trade["stop_loss"])
            init_sl = float(trade.get("initial_sl", sl))
            qty = int(trade.get("qty", 1))
            side = trade["action"]

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

            # Track highest profit reached during this trade's life
            prev_peak = float(trade.get("peak_pnl_inr", 0.0))
            peak_pnl = max(prev_peak, unrealized_inr)
            trade["peak_pnl_inr"] = round(peak_pnl, 2)

            # Check live news emotion shift if trade is losing or giving back gains
            live_news = None
            emotion_flipped = False
            if unrealized_inr < 0 or (peak_pnl > 0 and unrealized_inr < peak_pnl * 0.6):
                live_news = self._analyze_live_news_emotion(sym)
                if side == "BUY" and live_news["emotion"] == "NEGATIVE":
                    emotion_flipped = True
                elif side.startswith("SELL") and live_news["emotion"] == "POSITIVE":
                    emotion_flipped = True

            exit_triggered = False
            exit_status = ""
            exit_price = ltp
            exit_note = ""

            # 1) BREAKING NEWS / EMOTION REVERSAL CUTOFF
            if emotion_flipped:
                exit_triggered = True
                exit_status = "EMOTION SHIFT EXIT"
                exit_price = ltp
                exit_note = f"News emotion flipped to {live_news['emotion']}: {live_news['top_headline']}"

            # 2) TARGET HIT OR MOMENTUM EXTENSION
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
                    exit_price = t1
                    exit_note = "AI booked full target profit."

            # 3) STOP-LOSS HIT
            elif (side == "BUY" and ltp <= sl) or (side.startswith("SELL") and ltp >= sl):
                exit_triggered = True
                is_be = (sl >= entry) if side == "BUY" else (sl <= entry)
                exit_status = "BREAKEVEN EXIT" if is_be else "LOSS (SL HIT)"
                exit_price = sl
                exit_note = "Exited at dynamic Stop-Loss."

            # 4) LEARNED PEAK-PROFIT GIVEBACK CUTOFF (Locks in profit before full reversal)
            elif peak_pnl >= (risk_unit * qty * 0.75) and unrealized_inr > 0:
                giveback_pct = ((peak_pnl - unrealized_inr) / peak_pnl) * 100.0
                if giveback_pct >= giveback_limit_pct:
                    exit_triggered = True
                    exit_status = "WIN (PROFIT LOCKED)"
                    exit_price = ltp
                    exit_note = f"AI locked profit after {giveback_pct:.0f}% pullback from peak ₹{peak_pnl:,.0f}."

            # 5) EARLY WEAKNESS CUT BELOW/ABOVE VWAP
            elif r_multiple <= -0.65 and ((side == "BUY" and ltp < vwap) or (side.startswith("SELL") and ltp > vwap)):
                exit_triggered = True
                exit_status = "EARLY CUT (LOSS MINIMIZED)"
                exit_price = ltp
                exit_note = f"Cut early against VWAP (₹{vwap}) to protect capital."

            # 6) LEARNED HOLDING DECAY / STAGNATION EXIT
            elif hold_mins >= max_hold_mins and unrealized_inr <= 0:
                exit_triggered = True
                exit_status = "STAGNATION EXIT (AI TIMING)"
                exit_price = ltp
                exit_note = f"Exited after {hold_mins:.0f}m without momentum follow-through."

            # 7) TRAILING BREAKEVEN PROTECTION
            elif r_multiple >= 0.8 and ((side == "BUY" and sl < entry) or (side.startswith("SELL") and sl > entry)):
                trade["stop_loss"] = entry
                trade["trade_management_note"] = f"SL trailed to breakeven ₹{entry}."

            if exit_triggered:
                pnl_pct = (((exit_price - entry) / entry) * 100.0) if side == "BUY" else (((entry - exit_price) / entry) * 100.0)
                pnl_inr = ((exit_price - entry) * qty) if side == "BUY" else ((entry - exit_price) * qty)

                trade["status"] = exit_status
                trade["exit_price"] = round(exit_price, 2)
                trade["exit_time"] = f"{now_time} IST"
                trade["unrealized_pnl_inr"] = 0.0
                trade["pnl_pct"] = round(pnl_pct, 2)
                trade["pnl_inr"] = round(pnl_inr, 2)
                trade["trade_management_note"] = exit_note

                self._recalculate_cash(self.portfolio)

                # Immediately research this trade & learn from its outcome/mistakes
                lesson = self._research_and_learn_from_trade(trade, live_news)

                msg = f"{'✅' if pnl_inr >= 0 else '🛑'} <b>AI CLOSED {sym} ({exit_status}):</b> ₹{exit_price} | P&L: <b>₹{pnl_inr:+,.0f}</b>"
                actions_taken.append(msg)

                if notify_telegram:
                    header = "✅ <b>AI TRADE COMPLETED — PROFIT BOOKED</b>" if pnl_inr >= 0 else "🔴 <b>AI TRADE COMPLETED — CUT & RESEARCHED</b>"
                    tg_ticket = (
                        f"{header}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> ({side}) | <b>Hold:</b> {hold_mins:.0f} mins\n"
                        f"🏷️ <b>AI Decision:</b> {exit_status}\n"
                        f"🔢 <b>Entry:</b> ₹{entry} ➔ <b>Exit:</b> <b>₹{round(exit_price, 2)}</b>\n"
                        f"💰 <b>Net Realized P&L:</b> <b>₹{pnl_inr:+,.2f} ({pnl_pct:+.2f}%)</b>\n"
                        f"🏦 <b>Updated Wallet Equity:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>\n"
                        f"🧠 <b>Post-Trade AI Research & Lesson:</b>\n<i>{lesson}</i>"
                    )
                    self.notifier.send_message(tg_ticket)

        self._save_ledger()
        self._save_portfolio()
        return actions_taken

    # ------------------------------------------------------------------
    # 4. PER-TRADE RESEARCH, MISTAKE AUTOPSY & CONTINUOUS LEARNING
    # ------------------------------------------------------------------
    def _research_and_learn_from_trade(self, trade, pre_fetched_news=None):
        """
        Researches the stock's news, timing window, profit giveback, and signal accuracy
        immediately upon trade completion so the AI improves on the very next trade.
        """
        sym = trade["symbol"]
        pnl_inr = float(trade.get("pnl_inr", 0.0))
        peak_pnl = float(trade.get("peak_pnl_inr", 0.0))
        side = trade.get("action", "BUY")
        session_used = trade.get("entry_session", self._get_session_bucket())
        hold_mins = float(trade.get("hold_minutes", 0.0))

        news_ctx = pre_fetched_news or self._analyze_live_news_emotion(sym)
        latest_headline = news_ctx.get("top_headline", "No new headline")
        latest_emotion = news_ctx.get("emotion", "NEUTRAL")

        if pnl_inr < 0:
            f_scores = trade.get("factor_scores", {})
            culprit = max(f_scores.items(), key=lambda x: x[1])[0] if f_scores else "momentum"
            entry_rsi = float(trade.get("entry_rsi", 50.0))

            reasons = []
            if (side == "BUY" and latest_emotion == "NEGATIVE") or (side.startswith("SELL") and latest_emotion == "POSITIVE"):
                reasons.append(f"sudden {latest_emotion} news ('{latest_headline[:65]}')")
            if peak_pnl > abs(pnl_inr) * 0.5 and peak_pnl > 200:
                reasons.append(f"gave back +₹{peak_pnl:,.0f} peak profit before tightening giveback cutoff")
            if side == "BUY" and entry_rsi > 68:
                reasons.append(f"overbought entry RSI {entry_rsi}")
            if not reasons:
                reasons.append(f"over-weighted {culprit.upper()} ({f_scores.get(culprit, 50)}) in {session_used}")

            lesson = (
                f"MISTAKE AUTOPSY ({sym} {side}, ₹{pnl_inr:+,.0f}): Failed due to {', '.join(reasons)}. "
                f"Action: Lowered {session_used} priority, tightened profit-giveback & {culprit} weight."
            )
            trade["mistake_diagnosis"] = lesson
            self.portfolio["mistakes_identified"] = (
                [lesson] + self.portfolio.get("mistakes_identified", [])
            )[:20]
        else:
            lesson = (
                f"WINNING EDGE ({sym} {side}, +₹{pnl_inr:,.0f} in {hold_mins:.0f}m): "
                f"Confirmed edge in {session_used} with {latest_emotion} news alignment."
            )
            trade["mistake_diagnosis"] = lesson

        self.weights.setdefault("trade_lessons_log", []).insert(0, {
            "timestamp": self._now_ist().strftime("%Y-%m-%d %I:%M %p IST"),
            "symbol": sym,
            "pnl_inr": round(pnl_inr, 2),
            "session": session_used,
            "emotion_at_exit": latest_emotion,
            "lesson": lesson
        })
        self.weights["trade_lessons_log"] = self.weights["trade_lessons_log"][:50]

        self._optimize_weights_from_history(force_recalibrate=True)
        return lesson

    def evaluate_open_trades_and_learn(self, notify_telegram=True):
        """
        Periodic Deep Calibration & Equity Snapshot (no forced price override).
        Runs live management first, then logs equity and recalibrates all 6 learning layers.
        """
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
        """
        6-LAYER AUTONOMOUS SELF-LEARNING ENGINE:
        1. Factor Pillar Expectancy (Volume vs News Emotion vs Pre-Market vs Momentum)
        2. Per-Ticker Win Rate & Net P&L Memory (Auto-Blacklist & Star Performers)
        3. Time-of-Day Session Window Learning (Teaches AI WHEN to take trades)
        4. Directional Bias (BUY vs SHORT accuracy)
        5. Dynamic Exit & Giveback Calibration (Teaches AI WHEN to cut trades)
        6. Minimum Conviction & RSI Exhaustion Threshold Tuning
        """
        settled = [t for t in self.ledger if t.get("status") != "OPEN" and "factor_scores" in t]
        if not settled:
            return "No settled trades yet to train the AI."

        if not force_recalibrate and len(settled) == int(self.weights.get("last_evaluated_count", 0)):
            return self.weights.get("learning_notes", "Model up to date.")

        # LAYER 1: Factor Weights
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

            if pnl > 0:
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

            direction = 1.0 if pnl > 0 else (-1.25 if pnl < 0 else 0.0)
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

        # LAYER 2: Per-Ticker Memory
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

        # LAYER 3: Learned Time-of-Day Session Windows
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

        # LAYER 4: Directional Bias
        buy_trades = [t for t in recent if t.get("action") == "BUY"]
        short_trades = [t for t in recent if str(t.get("action", "")).startswith("SELL")]

        buy_wr = (sum(1 for t in buy_trades if float(t.get("pnl_inr", 0)) > 0) / len(buy_trades) * 100.0) if buy_trades else 50.0
        short_wr = (sum(1 for t in short_trades if float(t.get("pnl_inr", 0)) > 0) / len(short_trades) * 100.0) if short_trades else 50.0

        self.weights["buy_bias_bonus"] = round(max(-6.0, min(6.0, (buy_wr - 50.0) * 0.15)), 2) if len(buy_trades) >= 2 else 0.0
        self.weights["short_bias_bonus"] = round(max(-6.0, min(6.0, (short_wr - 50.0) * 0.15)), 2) if len(short_trades) >= 2 else 0.0

        # LAYER 5: Dynamic Exit, Profit Giveback & Hold-Horizon Tuning
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

        # Tighten profit giveback cutoff if winning trades reversed into losses
        if giveback_losses >= 2:
            self.weights["max_profit_giveback_pct"] = max(25.0, round(45.0 - (giveback_losses * 5.0), 1))

        if winning_hold_times:
            avg_win_hold = sum(winning_hold_times) / len(winning_hold_times)
            self.weights["learned_max_hold_mins"] = round(max(45.0, min(240.0, avg_win_hold * 1.5)), 1)

        # LAYER 6: Minimum Conviction & RSI Ceiling Calibration
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

        best_pillar = max(
            [("Volume", self.weights["volume_weight"]),
             ("News Emotion", self.weights["sentiment_weight"]),
             ("Pre-Market", self.weights["premarket_weight"]),
             ("Chart/Tech", self.weights["momentum_weight"])],
            key=lambda x: x[1]
        )
        note = (
            f"Gen #{self.weights['generation']} ({len(settled)} trades): "
            f"Top Edge={best_pillar[0]} ({int(best_pillar[1]*100)}%), "
            f"Min Conviction={self.weights['min_conviction_threshold']}, "
            f"Giveback Cutoff={self.weights['max_profit_giveback_pct']}%, "
            f"Max Hold={self.weights['learned_max_hold_mins']}m"
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
            "total_pnl_inr": total_pnl_inr
        }