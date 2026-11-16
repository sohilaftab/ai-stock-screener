import os
import json
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


class AdaptiveTradeEngine:
    def __init__(self):
        self.notifier = TelegramNotifier()
        self.weights = self._load_weights()
        self.ledger = self._load_ledger()
        self.portfolio = self._load_portfolio()

    @staticmethod
    def _now_ist():
        return datetime.now(IST)

    def _is_past_320pm_ist(self):
        now_ist = self._now_ist()
        return (now_ist.hour > 15) or (now_ist.hour == 15 and now_ist.minute >= 20)

    def _get_320pm_ist_price(self, sym, fallback_price):
        """
        Fetches the exact 3:20 PM IST price from intraday bars (15:19 1m bar or 15:15 5m bar)
        so intraday trades never settle on post-3:20 PM or 3:30 PM closing auction prices.
        """
        sym_ns = sym if sym.endswith(".NS") else f"{sym}.NS"
        for interval, max_min in (("1m", 19), ("5m", 15)):
            try:
                df = yf.Ticker(sym_ns).history(period="1d", interval=interval)
                if df.empty:
                    continue
                if df.index.tz is None:
                    df.index = df.index.tz_localize(IST)
                else:
                    df.index = df.index.tz_convert(IST)

                valid_bars = df[
                    (df.index.hour < 15)
                    | ((df.index.hour == 15) & (df.index.minute <= max_min))
                ]
                if not valid_bars.empty:
                    return round(float(valid_bars["Close"].iloc[-1]), 2)
            except Exception as e:
                print(f"[WARN] 3:20 PM {interval} bar lookup failed for {sym_ns}: {e}")
        return round(float(fallback_price), 2)

    def _load_weights(self):
        default_weights = {
            "volume_weight": 0.28,
            "sentiment_weight": 0.28,
            "premarket_weight": 0.22,
            "momentum_weight": 0.22,
            "risk_per_trade_pct": 1.0,
            "min_conviction_threshold": 52.0,  # Learned minimum quality bar to allow a trade
            "atr_sl_multiplier": 0.55,         # Learned Stop-Loss distance (in ATRs)
            "atr_tp_multiplier": 1.60,         # Learned Target 1 distance (in ATRs)
            "buy_rsi_ceiling": 72.0,           # Learned max RSI for BUY entries (avoids buying tops)
            "short_rsi_floor": 28.0,           # Learned min RSI for SHORT entries
            "buy_bias_bonus": 0.0,             # Learned directional adjustment for BUY setups
            "short_bias_bonus": 0.0,           # Learned directional adjustment for SHORT setups
            "ticker_memory": {},               # Per-stock win/loss & P&L memory
            "blacklisted_tickers": [],         # Stocks blocked due to repeated Stop-Loss hits
            "preferred_tickers": [],           # Stocks rewarded for consistent profitable execution
            "generation": 1,
            "last_evaluated_count": 0,
            "last_updated": self._now_ist().strftime("%Y-%m-%d %H:%M"),
            "learning_notes": "Initialized 5-Layer Self-Learning Brain on Rs 1,50,000 closed wallet."
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
        STRICT CLOSED-WALLET ACCOUNTING:
        - Initial seed is fixed at Rs 1,50,000.00 (never adds external money).
        - Current Equity = Rs 1,50,000.00 + sum of all settled Profits/Losses.
        - Allocated Capital = exact cash locked in currently OPEN trades.
        - Available Cash = Current Equity - Allocated Capital (strictly clamped >= 0).
        """
        port_dict["initial_capital"] = INITIAL_CAPITAL

        open_allocated = sum(
            float(t.get("capital_used", 0.0))
            for t in self.ledger if t.get("status") == "OPEN"
        )
        realized = sum(
            float(t.get("pnl_inr", 0.0))
            for t in self.ledger if t.get("status") != "OPEN"
        )
        port_dict["total_realized_pnl"] = round(realized, 2)
        port_dict["current_equity"] = round(max(0.0, INITIAL_CAPITAL + realized), 2)
        port_dict["allocated_capital"] = round(open_allocated, 2)
        port_dict["available_cash"] = round(max(0.0, port_dict["current_equity"] - open_allocated), 2)

    def _save_portfolio(self):
        self._recalculate_cash(self.portfolio)
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(self.portfolio, f, indent=2)

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
                    f"🆕 <b>NEW PRE-MARKET ENTRY: {sym}</b> jumped into Top Picks ({row['action']} @ ₹{row['entry_price']}, Conviction {row['conviction']}/100)."
                )
            else:
                old_item = old_map[sym]
                if old_item["action"] != row["action"]:
                    changes.append(
                        f"🔄 <b>DIRECTION FLIP: {sym}</b> flipped from {old_item['action']} to <b>{row['action']}</b> after Pre-Open order flow!"
                    )
                elif abs(float(row["entry_price"]) - float(old_item["entry_price"])) / float(old_item["entry_price"]) > 0.01:
                    changes.append(
                        f"⚡ <b>GAP ADJUSTMENT: {sym}</b> levels updated to Entry ₹{row['entry_price']} | Target ₹{row['target_1']} | SL ₹{row['stop_loss']}."
                    )

        for old_sym in old_map:
            if old_sym not in new_symbols:
                changes.append(f"❌ <b>DROPPED: {old_sym}</b> removed from Pre-9 AM list due to weak pre-market volume/sentiment.")

        return changes

    def generate_intraday_recommendations(self, enriched_df, premarket_df=None, top_k=5):
        """
        Uses all 5 learned layers from `model_weights.json`:
        1. Skips or heavily penalizes `blacklisted_tickers` that repeatedly lost money.
        2. Boosts `preferred_tickers` with proven historical win rates.
        3. Applies learned `buy_rsi_ceiling`, `short_rsi_floor`, and directional bias bonuses.
        4. Applies learned `min_conviction_threshold` so weak trades are filtered out.
        5. Computes Stop-Loss and Target using learned `atr_sl_multiplier` and `atr_tp_multiplier`.
        """
        if enriched_df.empty:
            return pd.DataFrame()

        self._recalculate_cash(self.portfolio)
        current_equity = float(self.portfolio["current_equity"])
        avail_cash = float(self.portfolio["available_cash"])
        budget_pool = avail_cash if avail_cash > 1000 else current_equity

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

            # Layer 1: Skip blacklisted stocks unless there are fewer than 3 total candidates
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

            # Layer 3: Use learned RSI ceiling/floor so the AI stops buying overbought tops
            if side == "BUY":
                trend_bonus = 25.0 if chart_trend == "BULLISH_UPTREND" else 0.0
                vwap_bonus = 15.0 if ltp >= vwap else -10.0
                if 50.0 <= rsi <= buy_rsi_max:
                    rsi_bonus = 15.0
                elif rsi > buy_rsi_max:
                    rsi_bonus = -15.0  # Penalize overbought exhaustion learned from past losses
                else:
                    rsi_bonus = 0.0
                mom_score = max(10.0, min(100.0, 45.0 + trend_bonus + vwap_bonus + rsi_bonus))
            else:
                trend_bonus = 25.0 if chart_trend == "BEARISH_DOWNTREND" else 0.0
                vwap_bonus = 15.0 if ltp <= vwap else -10.0
                if short_rsi_min <= rsi <= 50.0:
                    rsi_bonus = 15.0
                elif rsi < short_rsi_min:
                    rsi_bonus = -15.0  # Penalize oversold exhaustion on shorts
                else:
                    rsi_bonus = 0.0
                mom_score = max(10.0, min(100.0, 45.0 + trend_bonus + vwap_bonus + rsi_bonus))

            base_conviction = (
                (w_vol * vol_score) +
                (w_sent * sent_score) +
                (w_pre * pre_score) +
                (w_mom * mom_score)
            )

            # Apply Layer 1 & Layer 2 Historical Memory Adjustments
            ticker_adj = 0.0
            if sym in ticker_memory:
                tm = ticker_memory[sym]
                trades_n = tm.get("trades", 0)
                win_r = tm.get("win_rate", 50.0)
                net_pnl = tm.get("net_pnl_inr", 0.0)
                if trades_n >= 1:
                    if win_r >= 60.0 and net_pnl > 0:
                        ticker_adj = min(8.0, 3.0 + (trades_n * 1.5))  # Reward proven winners
                    elif win_r < 40.0 and net_pnl < 0:
                        ticker_adj = -min(15.0, 5.0 + (trades_n * 3.0))  # Penalize repeat losers

            dir_adj = buy_bonus if side == "BUY" else short_bonus
            conviction = round(max(1.0, min(99.0, base_conviction + ticker_adj + dir_adj)), 1)

            # Layer 4: Learned ATR Stop-Loss & Target Multipliers
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

        # Layer 5: Apply Learned Minimum Conviction Filter (Always keep at least top 2 if market is quiet)
        qualified = [c for c in scored_candidates if c["conviction"] >= min_conviction]
        if len(qualified) < 2:
            qualified = scored_candidates[:max(2, min(3, len(scored_candidates)))]
        top_candidates = qualified[:top_k]

        # STRICT SEQUENTIAL WALLET ALLOCATION
        remaining_wallet_cash = budget_pool
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

            max_loss_inr = round(qty * risk_per_share, 2)
            expected_profit_inr = round(qty * abs(item["target_1"] - entry_price), 2)
            risk_pct = round((risk_per_share / entry_price) * 100, 2)
            reward_pct = round((abs(item["target_1"] - entry_price) / entry_price) * 100, 2)

            final_suggestions.append({
                "symbol": item["symbol"],
                "action": item["action"],
                "conviction": item["conviction"],
                "qty": qty,
                "capital_used": capital_used,
                "max_loss_inr": max_loss_inr,
                "expected_profit_inr": expected_profit_inr,
                "entry_price": entry_price,
                "target_1": item["target_1"],
                "target_2": item["target_2"],
                "stop_loss": item["stop_loss"],
                "risk_pct": risk_pct,
                "reward_pct": reward_pct,
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

    def record_recommendations(self, rec_df, notify_telegram=True):
        """
        Executes virtual trades out of the exact available_cash in your wallet.
        Strictly blocks any new intraday entries at or after 3:20 PM IST.
        """
        if rec_df.empty:
            return 0

        if self._is_past_320pm_ist():
            print("[INFO] Past 3:20 PM IST intraday cutoff. Blocking new intraday entries.")
            return 0

        self._recalculate_cash(self.portfolio)
        avail_cash = float(self.portfolio["available_cash"])

        now_ist = self._now_ist()
        today_str = now_ist.strftime("%Y-%m-%d")
        now_time = now_ist.strftime("%I:%M:%S %p")
        existing_today = {
            item["symbol"] for item in self.ledger if item.get("date") == today_str
        }

        candidate_syms = rec_df["symbol"].tolist()
        fresh_ticks = MarketDataScanner.fetch_live_intraday_snapshot(candidate_syms)

        added = 0
        for _, row in rec_df.iterrows():
            sym = row["symbol"]
            if sym in existing_today:
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
                "trade_id": f"{today_str}_{sym}",
                "date": today_str,
                "timestamp": now_time,
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
                "unrealized_pnl_inr": 0.0,
                "entry_rsi": float(row.get("rsi", 50.0)),
                "chart_trend": row.get("chart_trend", "RANGE_CONSOLIDATION"),
                "status": "OPEN",
                "trade_management_note": f"Funded from wallet (₹{avail_cash:,.0f} cash remaining; 3:20 PM IST auto-exit).",
                "exit_price": None,
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
            added += 1

            if notify_telegram:
                side_header = "🟢 <b>NEW BUY TRADE TAKEN (LIVE TICK)</b>" if row["action"] == "BUY" else "🔴 <b>NEW SHORT SELL TRADE TAKEN (LIVE TICK)</b>"
                entry_ticket = (
                    f"{side_header}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📌 <b>Stock:</b> <code>{sym}</code> ({row['action']})\n"
                    f"🕒 <b>Exact Execution Time:</b> {now_time} IST ({today_str})\n"
                    f"🧠 <b>AI Conviction:</b> {row['conviction']}/100 | Chart: <code>{row['chart_trend']}</code>\n"
                    f"📦 <b>Quantity:</b> <b>{qty} Shares</b> @ <b>₹{entry_p}</b>\n"
                    f"💰 <b>Wallet Used:</b> ₹{cap_needed:,.2f} <i>(Remaining Cash: ₹{avail_cash:,.2f} / Wallet Equity: ₹{self.portfolio['current_equity']:,.2f})</i>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 <b>Target 1:</b> ₹{t1} <i>(+₹{exp_profit:,.0f} profit)</i>\n"
                    f"🚀 <b>Target 2:</b> ₹{t2}\n"
                    f"🛑 <b>Stop-Loss:</b> ₹{sl} <i>(-₹{max_loss:,.0f} max risk)</i>\n"
                    f"⏰ <b>Hard Intraday Cutoff:</b> 03:20 PM IST\n"
                    f"📰 <b>Catalyst ({row['sentiment']}):</b> {row['headline']}"
                )
                self.notifier.send_message(entry_ticket)

        if added > 0:
            self._save_ledger()
            self._save_portfolio()
        return added

    def manage_live_positions(self, notify_telegram=True):
        """
        Monitors active positions via live NSE tick prices & 1-minute intraday candles.
        - ENFORCES HARD AUTO SQUARE-OFF AT OR AFTER 3:20 PM IST using the exact 3:20 PM bar price.
        - Whenever a trade closes, immediately triggers `_optimize_weights_from_history()`.
        """
        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        if not open_trades:
            return []

        symbols = [t["symbol"] for t in open_trades]
        snapshots = MarketDataScanner.fetch_live_intraday_snapshot(symbols)
        actions_taken = []
        closed_any = False

        now_ist = self._now_ist()
        now_time = now_ist.strftime("%I:%M:%S %p")
        past_320_cutoff = self._is_past_320pm_ist()

        for trade in open_trades:
            sym = trade["symbol"]
            entry = float(trade["entry_price"])
            t1 = float(trade["target_1"])
            t2 = float(trade["target_2"])
            sl = float(trade["stop_loss"])
            init_sl = float(trade.get("initial_sl", sl))
            qty = int(trade.get("qty", 1))
            side = trade["action"]

            snap = snapshots.get(sym)
            tg_ticket = None

            # --------------------------------------------------------------
            # HARD 3:20 PM IST INTRADAY SQUARE-OFF (Runs even if live snap is empty)
            # --------------------------------------------------------------
            if past_320_cutoff:
                fallback_ltp = snap["current_price"] if snap else float(trade.get("current_price", entry))
                exit_320 = self._get_320pm_ist_price(sym, fallback_ltp)

                if side == "BUY":
                    pnl_pct = ((exit_320 - entry) / entry) * 100.0
                    pnl_inr = (exit_320 - entry) * qty
                    exit_label = "SELL (3:20 PM Auto Square-Off)"
                else:
                    pnl_pct = ((entry - exit_320) / entry) * 100.0
                    pnl_inr = (entry - exit_320) * qty
                    exit_label = "BUY TO COVER (3:20 PM Auto Square-Off)"

                trade["status"] = "3:20 PM WIN (AUTO SQUARE-OFF)" if pnl_inr >= 0 else "3:20 PM LOSS (AUTO SQUARE-OFF)"
                trade["exit_price"] = exit_320
                trade["current_price"] = exit_320
                trade["unrealized_pnl_inr"] = 0.0
                trade["pnl_pct"] = round(pnl_pct, 2)
                trade["pnl_inr"] = round(pnl_inr, 2)
                trade["trade_management_note"] = f"Auto squared-off at 3:20 PM IST limit @ ₹{exit_320:.2f}."
                trade["mistake_diagnosis"] = (
                    "Intraday square-off at 3:20 PM IST in profit."
                    if pnl_inr >= 0
                    else f"{sym} ({side}) did not reach target before 3:20 PM IST cutoff (P&L: ₹{pnl_inr:+,.0f})."
                )
                closed_any = True
                self._recalculate_cash(self.portfolio)

                msg = f"⏰ <b>3:20 PM AUTO SQUARE-OFF ({sym}):</b> Closed {qty} shares @ ₹{exit_320} (P&L: ₹{trade['pnl_inr']:+,.0f})."
                actions_taken.append(msg)
                tg_ticket = (
                    f"⏰ <b>TRADE COMPLETED — 3:20 PM IST INTRADAY AUTO SQUARE-OFF</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📌 <b>Stock:</b> <code>{sym}</code> | <b>Order:</b> {exit_label}\n"
                    f"🕒 <b>Cutoff Time:</b> 03:20 PM IST\n"
                    f"📦 <b>Qty Closed:</b> {qty} Shares\n"
                    f"🔢 <b>Entry:</b> ₹{entry} ➔ <b>3:20 PM Exit:</b> <b>₹{exit_320}</b>\n"
                    f"💰 <b>Net Realized P&L:</b> <b>₹{trade['pnl_inr']:+,.2f} ({trade['pnl_pct']:+.2f}%)</b>\n"
                    f"🏦 <b>Updated Wallet Balance:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>"
                )
                if notify_telegram and tg_ticket:
                    self.notifier.send_message(tg_ticket)
                continue

            # --------------------------------------------------------------
            # REGULAR INTRADAY MONITORING (Before 3:20 PM IST)
            # --------------------------------------------------------------
            if not snap:
                continue

            ltp = snap["current_price"]
            d_high = snap["day_high"]
            d_low = snap["day_low"]
            vwap = snap["intraday_vwap"]

            risk_unit = max(abs(entry - init_sl), entry * 0.005)
            trade["current_price"] = ltp
            trade["day_high"] = d_high
            trade["day_low"] = d_low

            if side == "BUY":
                unrealized_per_share = ltp - entry
                trade["unrealized_pnl_inr"] = round(unrealized_per_share * qty, 2)
                r_multiple = unrealized_per_share / risk_unit

                if ltp >= t1 or d_high >= t1:
                    if ltp >= t1 and ltp > vwap and t1 < t2 and now_ist.hour < 15:
                        new_sl = round(entry + (0.8 * risk_unit), 2)
                        trade["stop_loss"] = max(sl, new_sl)
                        trade["target_1"] = t2
                        trade["trade_management_note"] = "Target extended to T2; profit locked."
                        msg = f"🚀 <b>TARGET EXTENDED ({sym}):</b> Crossed ₹{t1}! New Target: <b>₹{t2}</b> | Trailing SL: <b>₹{trade['stop_loss']}</b>."
                        actions_taken.append(msg)
                        tg_ticket = (
                            f"🚀 <b>LIVE TRADE MODIFIED — TARGET EXTENDED</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>Stock:</b> <code>{sym}</code> (BUY {qty} Qty)\n"
                            f"📈 <b>Current LTP:</b> ₹{ltp} (Above VWAP ₹{vwap})\n"
                            f"🎯 <b>New Extended Target:</b> <b>₹{t2}</b> <i>(was ₹{t1})</i>\n"
                            f"🛡️ <b>Profit-Locked SL:</b> <b>₹{trade['stop_loss']}</b> <i>(Entry was ₹{entry})</i>\n"
                            f"💵 <b>Unrealized Profit:</b> +₹{trade['unrealized_pnl_inr']:,.0f}"
                        )
                    else:
                        trade["status"] = "WIN (TARGET HIT)"
                        trade["exit_price"] = t1
                        trade["pnl_pct"] = round(((t1 - entry) / entry) * 100, 2)
                        trade["pnl_inr"] = round((t1 - entry) * qty, 2)
                        trade["unrealized_pnl_inr"] = 0.0
                        trade["trade_management_note"] = "Booked full target profit."
                        trade["mistake_diagnosis"] = "Target achieved cleanly."
                        closed_any = True
                        self._recalculate_cash(self.portfolio)
                        msg = f"✅ <b>TRADE COMPLETED — TARGET HIT ({sym}):</b> Sold {qty} shares @ ₹{t1} for <b>+₹{trade['pnl_inr']:,.0f}</b> profit!"
                        actions_taken.append(msg)
                        tg_ticket = (
                            f"✅ <b>TRADE COMPLETED — TARGET BOOKED (SELL EXIT)</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>Stock:</b> <code>{sym}</code> (Closed BUY)\n"
                            f"🕒 <b>Exit Time:</b> {now_time} IST\n"
                            f"📦 <b>Qty Sold:</b> {qty} Shares\n"
                            f"🔢 <b>Entry:</b> ₹{entry} ➔ <b>Exit:</b> <b>₹{t1}</b>\n"
                            f"💰 <b>Net Realized Profit:</b> <b>+₹{trade['pnl_inr']:,.2f} ({trade['pnl_pct']:+.2f}%)</b>\n"
                            f"🏦 <b>Updated Wallet Balance:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>"
                        )

                elif ltp <= sl:
                    trade["status"] = "BREAKEVEN EXIT" if sl >= entry else "LOSS (SL HIT)"
                    trade["exit_price"] = sl
                    trade["pnl_pct"] = round(((sl - entry) / entry) * 100, 2)
                    trade["pnl_inr"] = round((sl - entry) * qty, 2)
                    trade["unrealized_pnl_inr"] = 0.0
                    trade["trade_management_note"] = "Stopped out at active SL."
                    closed_any = True
                    self._recalculate_cash(self.portfolio)
                    msg = f"🛑 <b>TRADE COMPLETED — SL HIT ({sym}):</b> Sold {qty} shares @ ₹{sl} (P&L: ₹{trade['pnl_inr']:+,.0f})."
                    actions_taken.append(msg)
                    tg_ticket = (
                        f"🛑 <b>TRADE COMPLETED — STOP-LOSS EXIT</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> (Closed BUY)\n"
                        f"🕒 <b>Exit Time:</b> {now_time} IST\n"
                        f"📦 <b>Qty Sold:</b> {qty} Shares\n"
                        f"🔢 <b>Entry:</b> ₹{entry} ➔ <b>Exit:</b> <b>₹{sl}</b>\n"
                        f"📉 <b>Realized P&L:</b> <b>₹{trade['pnl_inr']:+,.2f} ({trade['pnl_pct']:+.2f}%)</b>\n"
                        f"🏦 <b>Updated Wallet Balance:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>"
                    )

                elif r_multiple >= 0.8 and sl < entry:
                    trade["stop_loss"] = entry
                    trade["trade_management_note"] = "Trailed SL to Breakeven (Risk = Rs 0)."
                    msg = f"🛡️ <b>SL TRAILED TO BREAKEVEN ({sym}):</b> Up +{r_multiple:.1f}R (₹{ltp}). SL moved to ₹{entry}."
                    actions_taken.append(msg)
                    tg_ticket = (
                        f"🛡️ <b>LIVE TRADE MODIFIED — SL TRAILED TO BREAKEVEN</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> (BUY {qty} Qty)\n"
                        f"📈 <b>Current LTP:</b> ₹{ltp} (+{r_multiple:.1f}R in profit)\n"
                        f"🛡️ <b>Updated Stop-Loss:</b> <b>₹{entry}</b> <i>(Risk reduced to ₹0)</i>\n"
                        f"🎯 <b>Active Target:</b> ₹{t1}"
                    )

                elif r_multiple <= -0.65 and ltp < vwap:
                    trade["status"] = "EARLY CUT (LOSS MINIMIZED)"
                    trade["exit_price"] = ltp
                    trade["pnl_pct"] = round(((ltp - entry) / entry) * 100, 2)
                    trade["pnl_inr"] = round((ltp - entry) * qty, 2)
                    trade["unrealized_pnl_inr"] = 0.0
                    saved_inr = round((ltp - sl) * qty, 2)
                    trade["trade_management_note"] = f"Cut early below VWAP; saved Rs {saved_inr:,.0f}."
                    closed_any = True
                    self._recalculate_cash(self.portfolio)
                    msg = f"⚠️ <b>TRADE COMPLETED — EARLY CUT ({sym}):</b> Sold @ ₹{ltp} (P&L: ₹{trade['pnl_inr']:+,.0f}), saving ₹{saved_inr:,.0f}!"
                    actions_taken.append(msg)
                    tg_ticket = (
                        f"⚠️ <b>TRADE COMPLETED — EARLY EXIT TO MINIMIZE LOSS</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> (Closed BUY early below VWAP ₹{vwap})\n"
                        f"🕒 <b>Exit Time:</b> {now_time} IST\n"
                        f"📦 <b>Qty Sold:</b> {qty} Shares @ <b>₹{ltp}</b>\n"
                        f"📉 <b>Realized P&L:</b> <b>₹{trade['pnl_inr']:+,.2f} ({trade['pnl_pct']:+.2f}%)</b>\n"
                        f"💡 <b>Capital Saved vs Full SL:</b> +₹{saved_inr:,.0f}\n"
                        f"🏦 <b>Updated Wallet Balance:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>"
                    )

            else:  # SELL (SHORT)
                unrealized_per_share = entry - ltp
                trade["unrealized_pnl_inr"] = round(unrealized_per_share * qty, 2)
                r_multiple = unrealized_per_share / risk_unit

                if ltp <= t1 or d_low <= t1:
                    if ltp <= t1 and ltp < vwap and t1 > t2 and now_ist.hour < 15:
                        new_sl = round(entry - (0.8 * risk_unit), 2)
                        trade["stop_loss"] = min(sl, new_sl)
                        trade["target_1"] = t2
                        trade["trade_management_note"] = "Short target extended to T2; profit locked."
                        msg = f"🚀 <b>SHORT TARGET EXTENDED ({sym}):</b> Dropped past ₹{t1}! New Target: <b>₹{t2}</b> | SL: <b>₹{trade['stop_loss']}</b>."
                        actions_taken.append(msg)
                        tg_ticket = (
                            f"🚀 <b>LIVE SHORT MODIFIED — TARGET EXTENDED</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>Stock:</b> <code>{sym}</code> (SHORT {qty} Qty)\n"
                            f"📉 <b>Current LTP:</b> ₹{ltp} (Below VWAP ₹{vwap})\n"
                            f"🎯 <b>New Extended Target:</b> <b>₹{t2}</b>\n"
                            f"🛡️ <b>Profit-Locked SL:</b> <b>₹{trade['stop_loss']}</b>"
                        )
                    else:
                        trade["status"] = "WIN (TARGET HIT)"
                        trade["exit_price"] = t1
                        trade["pnl_pct"] = round(((entry - t1) / entry) * 100, 2)
                        trade["pnl_inr"] = round((entry - t1) * qty, 2)
                        trade["unrealized_pnl_inr"] = 0.0
                        trade["trade_management_note"] = "Booked full short target profit."
                        trade["mistake_diagnosis"] = "Short target achieved cleanly."
                        closed_any = True
                        self._recalculate_cash(self.portfolio)
                        msg = f"✅ <b>TRADE COMPLETED — SHORT TARGET HIT ({sym}):</b> Covered {qty} shares @ ₹{t1} for <b>+₹{trade['pnl_inr']:,.0f}</b>!"
                        actions_taken.append(msg)
                        tg_ticket = (
                            f"✅ <b>TRADE COMPLETED — SHORT TARGET COVERED (BUY EXIT)</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 <b>Stock:</b> <code>{sym}</code> (Closed SHORT)\n"
                            f"🕒 <b>Exit Time:</b> {now_time} IST\n"
                            f"📦 <b>Qty Covered:</b> {qty} Shares\n"
                            f"🔢 <b>Short Entry:</b> ₹{entry} ➔ <b>Buy Cover:</b> <b>₹{t1}</b>\n"
                            f"💰 <b>Net Realized Profit:</b> <b>+₹{trade['pnl_inr']:,.2f} ({trade['pnl_pct']:+.2f}%)</b>\n"
                            f"🏦 <b>Updated Wallet Balance:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>"
                        )

                elif ltp >= sl:
                    trade["status"] = "BREAKEVEN EXIT" if sl <= entry else "LOSS (SL HIT)"
                    trade["exit_price"] = sl
                    trade["pnl_pct"] = round(((entry - sl) / entry) * 100, 2)
                    trade["pnl_inr"] = round((entry - sl) * qty, 2)
                    trade["unrealized_pnl_inr"] = 0.0
                    trade["trade_management_note"] = "Stopped out at active SL."
                    closed_any = True
                    self._recalculate_cash(self.portfolio)
                    msg = f"🛑 <b>TRADE COMPLETED — SHORT SL HIT ({sym}):</b> Covered {qty} shares @ ₹{sl} (P&L: ₹{trade['pnl_inr']:+,.0f})."
                    actions_taken.append(msg)
                    tg_ticket = (
                        f"🛑 <b>TRADE COMPLETED — SHORT STOP-LOSS EXIT</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> (Closed SHORT)\n"
                        f"🕒 <b>Exit Time:</b> {now_time} IST\n"
                        f"📦 <b>Qty Covered:</b> {qty} Shares\n"
                        f"🔢 <b>Short Entry:</b> ₹{entry} ➔ <b>Exit:</b> <b>₹{sl}</b>\n"
                        f"📉 <b>Realized P&L:</b> <b>₹{trade['pnl_inr']:+,.2f} ({trade['pnl_pct']:+.2f}%)</b>\n"
                        f"🏦 <b>Updated Wallet Balance:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>"
                    )

                elif r_multiple >= 0.8 and sl > entry:
                    trade["stop_loss"] = entry
                    trade["trade_management_note"] = "Trailed Short SL to Breakeven (Risk = Rs 0)."
                    msg = f"🛡️ <b>SHORT SL TRAILED TO BREAKEVEN ({sym}):</b> Moved SL to ₹{entry}."
                    actions_taken.append(msg)
                    tg_ticket = (
                        f"🛡️ <b>LIVE SHORT MODIFIED — SL TRAILED TO BREAKEVEN</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> (SHORT {qty} Qty)\n"
                        f"📉 <b>Current LTP:</b> ₹{ltp}\n"
                        f"🛡️ <b>Updated Stop-Loss:</b> <b>₹{entry}</b> <i>(Risk = ₹0)</i>"
                    )

                elif r_multiple <= -0.65 and ltp > vwap:
                    trade["status"] = "EARLY CUT (LOSS MINIMIZED)"
                    trade["exit_price"] = ltp
                    trade["pnl_pct"] = round(((entry - ltp) / entry) * 100, 2)
                    trade["pnl_inr"] = round((entry - ltp) * qty, 2)
                    trade["unrealized_pnl_inr"] = 0.0
                    saved_inr = round((sl - ltp) * qty, 2)
                    trade["trade_management_note"] = f"Covered early above VWAP; saved Rs {saved_inr:,.0f}."
                    closed_any = True
                    self._recalculate_cash(self.portfolio)
                    msg = f"⚠️ <b>TRADE COMPLETED — EARLY SHORT COVER ({sym}):</b> Covered @ ₹{ltp} (P&L: ₹{trade['pnl_inr']:+,.0f}), saving ₹{saved_inr:,.0f}!"
                    actions_taken.append(msg)
                    tg_ticket = (
                        f"⚠️ <b>TRADE COMPLETED — EARLY SHORT COVER</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> (Covered above VWAP ₹{vwap})\n"
                        f"🕒 <b>Exit Time:</b> {now_time} IST\n"
                        f"📦 <b>Qty Covered:</b> {qty} Shares @ <b>₹{ltp}</b>\n"
                        f"📉 <b>Realized P&L:</b> <b>₹{trade['pnl_inr']:+,.2f} ({trade['pnl_pct']:+.2f}%)</b>\n"
                        f"💡 <b>Saved vs Full SL:</b> +₹{saved_inr:,.0f}\n"
                        f"🏦 <b>Updated Wallet Balance:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>"
                    )

            if notify_telegram and tg_ticket:
                self.notifier.send_message(tg_ticket)

        self._save_ledger()
        self._save_portfolio()
        if closed_any:
            self._optimize_weights_from_history()
        return actions_taken

    def evaluate_open_trades_and_learn(self, notify_telegram=True):
        """
        End-of-Day Settlement + Mistake Autopsy + 5-Layer Statistical Self-Learning Optimizer.
        Safety guarantee: Any trade still OPEN is squared off using its exact 3:20 PM IST
        intraday candle close (never NSE's 3:30 PM closing auction price).
        """
        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        if not open_trades:
            learning_summary = self._optimize_weights_from_history(force_recalibrate=True)
            return {"evaluated": 0, "message": f"All trades already settled by 3:20 PM IST. {learning_summary}"}

        tickers = list({f"{t['symbol']}.NS" for t in open_trades})
        data = yf.download(
            tickers=tickers,
            period="5d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )

        evaluated_count = 0
        new_mistakes = []

        for trade in self.ledger:
            if trade.get("status") != "OPEN":
                continue

            sym = trade["symbol"]
            sym_ns = f"{sym}.NS"
            try:
                df = data[sym_ns].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
                if df.empty:
                    continue

                latest = df.iloc[-1]
                day_high = float(latest["High"])
                day_low = float(latest["Low"])
                daily_close = float(latest["Close"])

                # Strictly cap intraday square-off price to the 3:20 PM IST candle close
                price_at_320pm = self._get_320pm_ist_price(sym, daily_close)

                entry = float(trade["entry_price"])
                t1 = float(trade["target_1"])
                sl = float(trade["stop_loss"])
                qty = int(trade.get("qty", max(1, int(30000 // entry))))
                side = trade["action"]

                if side == "BUY":
                    hit_target = day_high >= t1
                    hit_sl = day_low <= sl
                    if hit_target and not hit_sl:
                        status, exit_p = "WIN (TARGET HIT)", t1
                    elif hit_sl and not hit_target:
                        status, exit_p = "LOSS (SL HIT)", sl
                    elif hit_target and hit_sl:
                        status, exit_p = ("WIN (TARGET HIT)", t1) if price_at_320pm >= entry else ("LOSS (SL HIT)", sl)
                    else:
                        exit_p = price_at_320pm
                        status = "3:20 PM WIN (AUTO SQUARE-OFF)" if exit_p >= entry else "3:20 PM LOSS (AUTO SQUARE-OFF)"
                    pnl_pct = ((exit_p - entry) / entry) * 100.0
                    pnl_inr = (exit_p - entry) * qty
                    exit_action_label = "SELL (3:20 PM IST Cutoff)"
                else:
                    hit_target = day_low <= t1
                    hit_sl = day_high >= sl
                    if hit_target and not hit_sl:
                        status, exit_p = "WIN (TARGET HIT)", t1
                    elif hit_sl and not hit_target:
                        status, exit_p = "LOSS (SL HIT)", sl
                    elif hit_target and hit_sl:
                        status, exit_p = ("WIN (TARGET HIT)", t1) if price_at_320pm <= entry else ("LOSS (SL HIT)", sl)
                    else:
                        exit_p = price_at_320pm
                        status = "3:20 PM WIN (AUTO SQUARE-OFF)" if exit_p <= entry else "3:20 PM LOSS (AUTO SQUARE-OFF)"
                    pnl_pct = ((entry - exit_p) / entry) * 100.0
                    pnl_inr = (entry - exit_p) * qty
                    exit_action_label = "BUY TO COVER (3:20 PM IST Cutoff)"

                trade["status"] = status
                trade["exit_price"] = round(exit_p, 2)
                trade["current_price"] = round(price_at_320pm, 2)
                trade["unrealized_pnl_inr"] = 0.0
                trade["day_high"] = round(day_high, 2)
                trade["day_low"] = round(day_low, 2)
                trade["pnl_pct"] = round(pnl_pct, 2)
                trade["pnl_inr"] = round(pnl_inr, 2)
                trade["trade_management_note"] = f"Settled at 3:20 PM IST cutoff price ₹{exit_p:.2f}."

                if pnl_inr < 0:
                    f_scores = trade.get("factor_scores", {})
                    culprit = max(f_scores.items(), key=lambda x: x[1])[0] if f_scores else "momentum"
                    entry_rsi = float(trade.get("entry_rsi", 50.0))
                    rsi_note = f" (Bought at high RSI {entry_rsi})" if (side == "BUY" and entry_rsi > 68) else ""
                    diagnosis = (
                        f"{sym} ({side}) lost ₹{abs(pnl_inr):,.0f}{rsi_note}: Over-relied on {culprit.upper()} "
                        f"score ({f_scores.get(culprit, 50)}). Penalizing {sym} & {culprit} weight."
                    )
                    trade["mistake_diagnosis"] = diagnosis
                    new_mistakes.append(diagnosis)
                else:
                    trade["mistake_diagnosis"] = "Execution aligned with trend & 3:20 PM risk rules."

                evaluated_count += 1
                self._recalculate_cash(self.portfolio)

                if notify_telegram:
                    result_icon = "✅ <b>TRADE COMPLETED (PROFIT)</b>" if pnl_inr >= 0 else "🔴 <b>TRADE COMPLETED (LOSS)</b>"
                    close_ticket = (
                        f"{result_icon}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 <b>Stock:</b> <code>{sym}</code> | <b>Order:</b> {exit_action_label}\n"
                        f"🏷️ <b>Outcome:</b> {status}\n"
                        f"🕒 <b>Cutoff Time:</b> 03:20 PM IST\n"
                        f"📦 <b>Quantity:</b> {qty} Shares\n"
                        f"🔢 <b>Entry Price:</b> ₹{entry} ➔ <b>Exit Price:</b> <b>₹{trade['exit_price']}</b>\n"
                        f"📊 <b>Day Range:</b> High ₹{trade['day_high']} | Low ₹{trade['day_low']}\n"
                        f"💰 <b>Net Realized P&L:</b> <b>₹{trade['pnl_inr']:+,.2f} ({trade['pnl_pct']:+.2f}%)</b>\n"
                        f"🏦 <b>New Wallet Balance:</b> <b>₹{self.portfolio['current_equity']:,.2f}</b>\n"
                        f"🧠 <b>AI Diagnosis:</b> <i>{trade['mistake_diagnosis']}</i>"
                    )
                    self.notifier.send_message(close_ticket)

            except Exception as e:
                print(f"[WARN] Could not evaluate {sym}: {e}")

        if new_mistakes:
            self.portfolio["mistakes_identified"] = (new_mistakes + self.portfolio.get("mistakes_identified", []))[:15]

        self._save_ledger()
        self._save_portfolio()

        today_str = self._now_ist().strftime("%Y-%m-%d")
        today_pnl = sum(float(t.get("pnl_inr", 0)) for t in self.ledger if t.get("date") == today_str and t.get("status") != "OPEN")
        eq_log = self.portfolio.get("daily_equity_log", [])
        if eq_log and eq_log[-1].get("date") == today_str:
            eq_log[-1]["equity"] = self.portfolio["current_equity"]
            eq_log[-1]["pnl"] = round(today_pnl, 2)
        else:
            eq_log.append({"date": today_str, "equity": self.portfolio["current_equity"], "pnl": round(today_pnl, 2)})
        self.portfolio["daily_equity_log"] = eq_log[-60:]
        self._save_portfolio()

        learning_summary = self._optimize_weights_from_history(force_recalibrate=True)

        if notify_telegram and evaluated_count > 0:
            perf = self.get_performance_metrics()
            summary_ticket = (
                f"🏦 <b>CLOSED WALLET & 5-LAYER AI BRAIN UPDATED</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🌱 <b>Original Seed Capital:</b> ₹{perf['initial_capital']:,.2f}\n"
                f"💼 <b>New Total Wallet Equity:</b> <b>₹{perf['current_equity']:,.2f}</b> ({perf['total_pnl_pct']:+.2f}%)\n"
                f"💵 <b>Next-Day Trading Budget:</b> <b>₹{perf['available_cash']:,.2f}</b>\n"
                f"📈 <b>Cumulative Net P&L:</b> <b>₹{perf['total_pnl_inr']:+,.2f}</b>\n"
                f"🎯 <b>Win Rate:</b> {perf['win_rate']}% ({perf['settled_trades']} settled)\n"
                f"🧠 <b>Self-Learning Evolution:</b> <i>{learning_summary}</i>"
            )
            self.notifier.send_message(summary_ticket)

        return {"evaluated": evaluated_count, "message": learning_summary}

    def _optimize_weights_from_history(self, force_recalibrate=False):
        """
        5-LAYER STATISTICAL SELF-LEARNING ENGINE:
        Computes clean, idempotent parameter updates directly from all settled trades in `trade_history.json`:
        1. Factor Pillar Expectancy (Volume vs News vs Pre-Market vs Momentum)
        2. Per-Ticker Win Rate & Net P&L Memory (Auto-Blacklist & Preferred list)
        3. Directional Bias (BUY vs SHORT accuracy)
        4. ATR Stop-Loss & Target Multiplier Tuning (from SL hit rate vs Target hit rate)
        5. RSI Overbought/Oversold Ceiling & Minimum Conviction Threshold Tuning
        """
        settled = [t for t in self.ledger if t.get("status") != "OPEN" and "factor_scores" in t]
        if not settled:
            return "No settled trades yet to train the AI."

        if not force_recalibrate and len(settled) == int(self.weights.get("last_evaluated_count", 0)):
            return self.weights.get("learning_notes", "Model up to date.")

        # ------------------------------------------------------------------
        # LAYER 1: Clean Baseline Factor Weight Optimization
        # ------------------------------------------------------------------
        base = {"volume": 0.28, "sentiment": 0.28, "premarket": 0.22, "momentum": 0.22}
        adjustments = {"volume": 0.0, "sentiment": 0.0, "premarket": 0.0, "momentum": 0.0}
        recent = settled[-40:]

        wins_count = 0
        sl_hits = 0
        target_hits = 0
        winning_convictions = []
        losing_convictions = []
        losing_buy_rsis = []

        for t in recent:
            pnl = float(t.get("pnl_pct", 0.0))
            status_str = str(t.get("status", ""))
            conv = float(t.get("conviction", 60.0))

            if pnl > 0:
                wins_count += 1
                winning_convictions.append(conv)
            else:
                losing_convictions.append(conv)
                if t.get("action") == "BUY" and "entry_rsi" in t:
                    losing_buy_rsis.append(float(t["entry_rsi"]))

            if "SL HIT" in status_str:
                sl_hits += 1
            elif "TARGET HIT" in status_str:
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

        # ------------------------------------------------------------------
        # LAYER 2: Ticker Memory, Auto-Blacklisting & Star Performers
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # LAYER 3: Directional Bias Calibration (BUY vs SHORT Accuracy)
        # ------------------------------------------------------------------
        buy_trades = [t for t in recent if t.get("action") == "BUY"]
        short_trades = [t for t in recent if str(t.get("action", "")).startswith("SELL")]

        buy_wr = (sum(1 for t in buy_trades if float(t.get("pnl_inr", 0)) > 0) / len(buy_trades) * 100.0) if buy_trades else 50.0
        short_wr = (sum(1 for t in short_trades if float(t.get("pnl_inr", 0)) > 0) / len(short_trades) * 100.0) if short_trades else 50.0

        self.weights["buy_bias_bonus"] = round(max(-6.0, min(6.0, (buy_wr - 50.0) * 0.15)), 2) if len(buy_trades) >= 2 else 0.0
        self.weights["short_bias_bonus"] = round(max(-6.0, min(6.0, (short_wr - 50.0) * 0.15)), 2) if len(short_trades) >= 2 else 0.0

        # ------------------------------------------------------------------
        # LAYER 4: Adaptive ATR Stop-Loss & Target Multiplier Tuning
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # LAYER 5: Minimum Conviction Threshold & RSI Ceiling Calibration
        # ------------------------------------------------------------------
        if losing_convictions and winning_convictions:
            avg_loss_conv = sum(losing_convictions) / len(losing_convictions)
            avg_win_conv = sum(winning_convictions) / len(winning_convictions)
            if avg_win_conv > avg_loss_conv:
                learned_cutoff = (avg_loss_conv + avg_win_conv) / 2.0
                self.weights["min_conviction_threshold"] = round(max(50.0, min(68.0, learned_cutoff)), 1)

        if losing_buy_rsis:
            high_rsi_losses = [r for r in losing_buy_rsis if r >= 67.0]
            if len(high_rsi_losses) >= 2:
                self.weights["buy_rsi_ceiling"] = 66.0

        recent_win_rate = (wins_count / len(recent)) * 100.0
        if recent_win_rate >= 60.0 and self.portfolio["current_equity"] >= INITIAL_CAPITAL:
            self.weights["risk_per_trade_pct"] = min(1.5, round(1.0 + ((recent_win_rate - 50.0) * 0.015), 2))
        elif recent_win_rate < 45.0 or self.portfolio["current_equity"] < INITIAL_CAPITAL:
            self.weights["risk_per_trade_pct"] = max(0.6, round(1.0 - ((50.0 - recent_win_rate) * 0.015), 2))

        self.weights["generation"] = len(settled) + 1
        self.weights["last_evaluated_count"] = len(settled)
        self.weights["last_updated"] = self._now_ist().strftime("%Y-%m-%d %I:%M %p")

        best_pillar = max(
            [("Volume", self.weights["volume_weight"]),
             ("News AI", self.weights["sentiment_weight"]),
             ("Pre-Market", self.weights["premarket_weight"]),
             ("Chart/Tech", self.weights["momentum_weight"])],
            key=lambda x: x[1]
        )
        bl_note = f" | Blocked: {', '.join(blacklisted)}" if blacklisted else ""
        pref_note = f" | Top Edge: {', '.join(preferred[:3])}" if preferred else ""
        note = (
            f"Gen #{self.weights['generation']} ({len(settled)} trades analyzed): "
            f"Best Pillar={best_pillar[0]} ({int(best_pillar[1]*100)}%), "
            f"Min Conviction={self.weights['min_conviction_threshold']}, "
            f"ATR SL={self.weights['atr_sl_multiplier']}x / TP={self.weights['atr_tp_multiplier']}x"
            f"{pref_note}{bl_note}"
        )
        self.weights["learning_notes"] = note
        self._save_weights()
        return note

    def get_performance_metrics(self):
        self._recalculate_cash(self.portfolio)
        settled = [t for t in self.ledger if t.get("status") != "OPEN"]
        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        unrealized_total = sum(float(t.get("unrealized_pnl_inr", 0.0)) for t in open_trades)

        wins = [t for t in settled if float(t.get("pnl_inr", 0)) > 0]
        win_rate = round((len(wins) / len(settled)) * 100.0, 1) if settled else 0.0
        total_pnl_inr = round(sum(float(t.get("pnl_inr", 0)) for t in settled), 2)
        roi_pct = round((total_pnl_inr / INITIAL_CAPITAL) * 100.0, 2)

        return {
            "initial_capital": INITIAL_CAPITAL,
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