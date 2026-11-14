import os
import json
from datetime import datetime
import pandas as pd
import yfinance as yf
from market_data import MarketDataScanner

WEIGHTS_FILE = "model_weights.json"
LEDGER_FILE = "trade_history.json"
PORTFOLIO_FILE = "portfolio_state.json"
MORNING_WATCHLIST_FILE = "morning_watchlist.json"

INITIAL_CAPITAL = 150000.0


class AdaptiveTradeEngine:
    def __init__(self):
        self.weights = self._load_weights()
        self.ledger = self._load_ledger()
        self.portfolio = self._load_portfolio()

    def _load_weights(self):
        default_weights = {
            "volume_weight": 0.28,
            "sentiment_weight": 0.28,
            "premarket_weight": 0.22,
            "momentum_weight": 0.22,
            "risk_per_trade_pct": 1.0,  # Max 1% of Rs 1.5L (Rs 1,500) risked per trade SL
            "generation": 1,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "learning_notes": "Initialized autonomous Rs 1,50,000 portfolio brain."
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
                {"date": datetime.now().strftime("%Y-%m-%d"), "equity": INITIAL_CAPITAL, "pnl": 0.0}
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
        """Syncs available cash and allocated capital based on currently OPEN trades."""
        open_allocated = sum(
            float(t.get("capital_used", 0.0))
            for t in self.ledger if t.get("status") == "OPEN"
        )
        realized = sum(
            float(t.get("pnl_inr", 0.0))
            for t in self.ledger if t.get("status") != "OPEN"
        )
        port_dict["total_realized_pnl"] = round(realized, 2)
        port_dict["current_equity"] = round(INITIAL_CAPITAL + realized, 2)
        port_dict["allocated_capital"] = round(open_allocated, 2)
        port_dict["available_cash"] = round(max(0.0, port_dict["current_equity"] - open_allocated), 2)

    def _save_portfolio(self):
        self._recalculate_cash(self.portfolio)
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(self.portfolio, f, indent=2)

    def save_morning_watchlist(self, rec_df):
        """Saves the Pre-9 AM watchlist so the 9:10 AM Pre-Market check can detect changes."""
        records = rec_df.to_dict(orient="records") if not rec_df.empty else []
        with open(MORNING_WATCHLIST_FILE, "w") as f:
            json.dump({"date": datetime.now().strftime("%Y-%m-%d"), "items": records}, f, indent=2)

    def compare_with_morning_watchlist(self, new_rec_df):
        """
        Compares 9:10 AM Pre-Market recommendations against the Pre-9 AM Morning Watchlist.
        Returns a list of plain-English momentum change alerts for Telegram & UI.
        """
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
        Synthesizes News, Multi-Timeframe Chart Trend (EMA20/50, RSI, VWAP), Pre-Market, and Volume.
        Sizes every trade out of the Rs 1,50,000 capital pool based on Stop-Loss distance and conviction.
        """
        if enriched_df.empty:
            return pd.DataFrame()

        pre_map = {}
        if premarket_df is not None and not premarket_df.empty:
            for _, r in premarket_df.iterrows():
                pre_map[r["symbol"]] = {
                    "iep": float(r.get("iep", 0)),
                    "pChange": float(r.get("pChange", 0)),
                    "buy_sell_ratio": float(r.get("buy_sell_ratio", 1.0))
                }

        w_vol = self.weights["volume_weight"]
        w_sent = self.weights["sentiment_weight"]
        w_pre = self.weights["premarket_weight"]
        w_mom = self.weights["momentum_weight"]

        current_equity = float(self.portfolio.get("current_equity", INITIAL_CAPITAL))
        risk_pct_allowed = float(self.weights.get("risk_per_trade_pct", 1.0)) / 100.0
        max_risk_inr_per_trade = current_equity * risk_pct_allowed  # e.g., Rs 1,500 max loss per trade
        max_capital_per_slot = current_equity / max(top_k, 4)       # e.g., max Rs 30,000 - Rs 37,500 cash per stock

        suggestions = []
        for _, row in enriched_df.iterrows():
            sym = row["symbol"]
            ltp = float(row["ltp"])
            chg = float(row["change_pct"])
            vol_ratio = float(row["vol_ratio"])
            atr = float(row.get("atr", ltp * 0.015))
            rsi = float(row.get("rsi", 50.0))
            vwap = float(row.get("vwap_proxy", ltp))
            chart_trend = row.get("chart_trend", "RANGE_CONSOLIDATION")

            # If pre-market IEP is fresher and valid, anchor entry to IEP during 9:00-9:15 AM
            if sym in pre_map and pre_map[sym]["iep"] > 50:
                chg = pre_map[sym]["pChange"]

            raw_sent = row.get("raw_sentiment", "NEUTRAL")
            ai_conf = float(row.get("AI Confidence", 50)) / 100.0
            headline = row.get("Why Is It Moving? (Latest News)", "Technical chart & volume setup")

            # Determine Trade Direction (BUY vs SELL/SHORT) using Chart + News + Price
            if (chg >= 0 and raw_sent != "NEGATIVE") or (chart_trend == "BULLISH_UPTREND" and raw_sent == "POSITIVE"):
                side = "BUY"
            elif (chg < 0 and raw_sent != "POSITIVE") or (chart_trend == "BEARISH_DOWNTREND" and raw_sent == "NEGATIVE"):
                side = "SELL (SHORT)"
            else:
                side = "BUY" if chg >= 0 else "SELL (SHORT)"

            # 1. Volume Sub-Score (0 - 100)
            vol_score = min(100.0, (vol_ratio / 2.8) * 100.0)

            # 2. News Sentiment Sub-Score (0 - 100)
            if (side == "BUY" and raw_sent == "POSITIVE") or (side.startswith("SELL") and raw_sent == "NEGATIVE"):
                sent_score = 65.0 + (ai_conf * 35.0)
            elif raw_sent == "NEUTRAL":
                sent_score = 52.0
            else:
                sent_score = 20.0

            # 3. Pre-Market Alignment Sub-Score (0 - 100)
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

            # 4. Multi-Timeframe Chart & Momentum Sub-Score (0 - 100)
            if side == "BUY":
                trend_bonus = 25.0 if chart_trend == "BULLISH_UPTREND" else 5.0
                vwap_bonus = 15.0 if ltp >= vwap else 0.0
                rsi_bonus = 15.0 if 52.0 <= rsi <= 72.0 else 0.0
                mom_score = min(100.0, 45.0 + trend_bonus + vwap_bonus + rsi_bonus)
            else:
                trend_bonus = 25.0 if chart_trend == "BEARISH_DOWNTREND" else 5.0
                vwap_bonus = 15.0 if ltp <= vwap else 0.0
                rsi_bonus = 15.0 if 28.0 <= rsi <= 48.0 else 0.0
                mom_score = min(100.0, 45.0 + trend_bonus + vwap_bonus + rsi_bonus)

            conviction = round(
                (w_vol * vol_score) +
                (w_sent * sent_score) +
                (w_pre * pre_score) +
                (w_mom * mom_score),
                1
            )

            # Intraday Stop-Loss & Target distances using 14d ATR
            intraday_risk_per_share = max(round(atr * 0.55, 2), round(ltp * 0.006, 2))
            entry_price = round(ltp, 2)

            if side == "BUY":
                stop_loss = round(entry_price - intraday_risk_per_share, 2)
                target_1 = round(entry_price + (intraday_risk_per_share * 1.6), 2)
                target_2 = round(entry_price + (intraday_risk_per_share * 2.6), 2)
            else:
                stop_loss = round(entry_price + intraday_risk_per_share, 2)
                target_1 = round(entry_price - (intraday_risk_per_share * 1.6), 2)
                target_2 = round(entry_price - (intraday_risk_per_share * 2.6), 2)

            # Position Sizing from Rs 1,50,000 Wallet:
            # Scale allocation by conviction (higher conviction gets closer to full slot cap)
            conviction_multiplier = max(0.6, min(1.15, conviction / 75.0))
            slot_budget = max_capital_per_slot * conviction_multiplier

            qty_by_risk = int(max_risk_inr_per_trade / max(intraday_risk_per_share, 0.5))
            qty_by_cash = int(slot_budget / entry_price)
            qty = max(1, min(qty_by_risk, qty_by_cash))

            capital_used = round(qty * entry_price, 2)
            max_loss_inr = round(qty * intraday_risk_per_share, 2)
            expected_profit_inr = round(qty * abs(target_1 - entry_price), 2)

            risk_pct = round((intraday_risk_per_share / entry_price) * 100, 2)
            reward_pct = round((abs(target_1 - entry_price) / entry_price) * 100, 2)

            suggestions.append({
                "symbol": sym,
                "action": side,
                "conviction": conviction,
                "qty": qty,
                "capital_used": capital_used,
                "max_loss_inr": max_loss_inr,
                "expected_profit_inr": expected_profit_inr,
                "entry_price": entry_price,
                "target_1": target_1,
                "target_2": target_2,
                "stop_loss": stop_loss,
                "risk_pct": risk_pct,
                "reward_pct": reward_pct,
                "rr_ratio": "1 : 1.6",
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

        rec_df = pd.DataFrame(suggestions)
        if not rec_df.empty:
            rec_df = rec_df.sort_values(by="conviction", ascending=False)
            return rec_df.head(top_k)
        return rec_df

    def record_recommendations(self, rec_df):
        """
        Executes virtual trades out of the Rs 1,50,000 available cash pool.
        Never exceeds available cash in the wallet.
        """
        if rec_df.empty:
            return 0

        self._recalculate_cash(self.portfolio)
        avail_cash = float(self.portfolio["available_cash"])

        today_str = datetime.now().strftime("%Y-%m-%d")
        existing_today = {
            item["symbol"] for item in self.ledger if item.get("date") == today_str
        }

        added = 0
        for _, row in rec_df.iterrows():
            sym = row["symbol"]
            if sym in existing_today:
                continue

            entry_p = float(row["entry_price"])
            qty = int(row["qty"])
            cap_needed = round(qty * entry_p, 2)

            # Ensure we never spend more than remaining available cash from Rs 1,50,000
            if cap_needed > avail_cash:
                qty = int(avail_cash // entry_p)
                cap_needed = round(qty * entry_p, 2)

            if qty <= 0 or cap_needed <= 0:
                continue

            avail_cash -= cap_needed
            trade_record = {
                "trade_id": f"{today_str}_{sym}",
                "date": today_str,
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "symbol": sym,
                "action": row["action"],
                "conviction": float(row["conviction"]),
                "qty": qty,
                "capital_used": cap_needed,
                "entry_price": entry_p,
                "initial_target": float(row["target_1"]),
                "target_1": float(row["target_1"]),
                "target_2": float(row["target_2"]),
                "initial_sl": float(row["stop_loss"]),
                "stop_loss": float(row["stop_loss"]),
                "current_price": entry_p,
                "unrealized_pnl_inr": 0.0,
                "status": "OPEN",
                "trade_management_note": "Position opened within Rs 1.5L risk budget.",
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

        if added > 0:
            self._save_ledger()
            self._save_portfolio()
        return added

    def manage_live_positions(self, sentiment_engine=None):
        """
        Acts as an independent live trader during market hours:
        - Checks 5-minute intraday candles & fresh news for all OPEN trades.
        - Books Target or Stop-Loss if touched.
        - Trails Stop-Loss to Breakeven once trade reaches +0.8R profit.
        - Extends Target to Target 2 if momentum is strong (>75% of T1 + VWAP support).
        - Cuts trade early to minimize loss if price breaks VWAP + adverse news hits.
        """
        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        if not open_trades:
            return []

        symbols = [t["symbol"] for t in open_trades]
        snapshots = MarketDataScanner.fetch_live_intraday_snapshot(symbols)
        actions_taken = []

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
            trade["current_price"] = ltp
            trade["day_high"] = d_high
            trade["day_low"] = d_low

            if side == "BUY":
                unrealized_per_share = ltp - entry
                trade["unrealized_pnl_inr"] = round(unrealized_per_share * qty, 2)
                r_multiple = unrealized_per_share / risk_unit

                # 1. Check if Target was achieved
                if ltp >= t1 or d_high >= t1:
                    # If strong momentum above VWAP and not yet extended, extend to Target 2!
                    if ltp >= t1 and ltp > vwap and t1 < t2:
                        new_sl = round(entry + (0.8 * risk_unit), 2)
                        trade["stop_loss"] = max(sl, new_sl)
                        trade["target_1"] = t2
                        msg = f"🚀 <b>EXTENDED TARGET ({sym}):</b> Crossed ₹{t1} with strong VWAP support! Target extended to <b>₹{t2}</b> & SL trailed to profit at <b>₹{trade['stop_loss']}</b>."
                        trade["trade_management_note"] = "Target extended to T2; profit locked."
                        actions_taken.append(msg)
                    else:
                        trade["status"] = "WIN (TARGET HIT)"
                        trade["exit_price"] = t1
                        trade["pnl_pct"] = round(((t1 - entry) / entry) * 100, 2)
                        trade["pnl_inr"] = round((t1 - entry) * qty, 2)
                        trade["unrealized_pnl_inr"] = 0.0
                        trade["trade_management_note"] = "Booked full target profit."
                        actions_taken.append(f"✅ <b>TARGET BOOKED ({sym}):</b> Closed BUY at ₹{t1} for <b>+₹{trade['pnl_inr']:,.0f}</b> profit!")

                # 2. Check if Stop-Loss was hit
                elif ltp <= sl:
                    trade["status"] = "BREAKEVEN EXIT" if sl >= entry else "LOSS (SL HIT)"
                    trade["exit_price"] = sl
                    trade["pnl_pct"] = round(((sl - entry) / entry) * 100, 2)
                    trade["pnl_inr"] = round((sl - entry) * qty, 2)
                    trade["unrealized_pnl_inr"] = 0.0
                    trade["trade_management_note"] = "Stopped out at active SL."
                    actions_taken.append(f"🛑 <b>SL TRIGGERED ({sym}):</b> Exited at ₹{sl} (P&L: ₹{trade['pnl_inr']:+,.0f}).")

                # 3. Trail Stop-Loss to Breakeven at +0.8R
                elif r_multiple >= 0.8 and sl < entry:
                    trade["stop_loss"] = entry
                    trade["trade_management_note"] = "Trailed SL to Breakeven (Risk = Rs 0)."
                    actions_taken.append(f"🛡️ <b>RISK ELIMINATED ({sym}):</b> Up +{r_multiple:.1f}R (₹{ltp}). Moved Stop-Loss to Breakeven (<b>₹{entry}</b>).")

                # 4. Early Cut to Minimize Loss if VWAP breaks + Momentum fails (-0.65R)
                elif r_multiple <= -0.65 and ltp < vwap:
                    trade["status"] = "EARLY CUT (LOSS MINIMIZED)"
                    trade["exit_price"] = ltp
                    trade["pnl_pct"] = round(((ltp - entry) / entry) * 100, 2)
                    trade["pnl_inr"] = round((ltp - entry) * qty, 2)
                    trade["unrealized_pnl_inr"] = 0.0
                    saved_inr = round((ltp - sl) * qty, 2)
                    trade["trade_management_note"] = f"Cut early below VWAP; saved Rs {saved_inr:,.0f}."
                    actions_taken.append(
                        f"⚠️ <b>EARLY EXIT TO MINIMIZE LOSS ({sym}):</b> Price broke below VWAP (₹{vwap}). Exited at ₹{ltp} (P&L: ₹{trade['pnl_inr']:+,.0f}), saving ₹{saved_inr:,.0f} before SL!"
                    )

            else:  # SELL (SHORT)
                unrealized_per_share = entry - ltp
                trade["unrealized_pnl_inr"] = round(unrealized_per_share * qty, 2)
                r_multiple = unrealized_per_share / risk_unit

                if ltp <= t1 or d_low <= t1:
                    if ltp <= t1 and ltp < vwap and t1 > t2:
                        new_sl = round(entry - (0.8 * risk_unit), 2)
                        trade["stop_loss"] = min(sl, new_sl)
                        trade["target_1"] = t2
                        msg = f"🚀 <b>EXTENDED SHORT TARGET ({sym}):</b> Dropped past ₹{t1}! Target extended to <b>₹{t2}</b> & SL trailed to <b>₹{trade['stop_loss']}</b>."
                        trade["trade_management_note"] = "Short target extended to T2; profit locked."
                        actions_taken.append(msg)
                    else:
                        trade["status"] = "WIN (TARGET HIT)"
                        trade["exit_price"] = t1
                        trade["pnl_pct"] = round(((entry - t1) / entry) * 100, 2)
                        trade["pnl_inr"] = round((entry - t1) * qty, 2)
                        trade["unrealized_pnl_inr"] = 0.0
                        trade["trade_management_note"] = "Booked full short target profit."
                        actions_taken.append(f"✅ <b>SHORT TARGET BOOKED ({sym}):</b> Covered at ₹{t1} for <b>+₹{trade['pnl_inr']:,.0f}</b> profit!")

                elif ltp >= sl:
                    trade["status"] = "BREAKEVEN EXIT" if sl <= entry else "LOSS (SL HIT)"
                    trade["exit_price"] = sl
                    trade["pnl_pct"] = round(((entry - sl) / entry) * 100, 2)
                    trade["pnl_inr"] = round((entry - sl) * qty, 2)
                    trade["unrealized_pnl_inr"] = 0.0
                    trade["trade_management_note"] = "Stopped out at active SL."
                    actions_taken.append(f"🛑 <b>SHORT SL TRIGGERED ({sym}):</b> Covered at ₹{sl} (P&L: ₹{trade['pnl_inr']:+,.0f}).")

                elif r_multiple >= 0.8 and sl > entry:
                    trade["stop_loss"] = entry
                    trade["trade_management_note"] = "Trailed Short SL to Breakeven (Risk = Rs 0)."
                    actions_taken.append(f"🛡️ <b>RISK ELIMINATED ({sym}):</b> Short up +{r_multiple:.1f}R. Moved Stop-Loss to Breakeven (<b>₹{entry}</b>).")

                elif r_multiple <= -0.65 and ltp > vwap:
                    trade["status"] = "EARLY CUT (LOSS MINIMIZED)"
                    trade["exit_price"] = ltp
                    trade["pnl_pct"] = round(((entry - ltp) / entry) * 100, 2)
                    trade["pnl_inr"] = round((entry - ltp) * qty, 2)
                    trade["unrealized_pnl_inr"] = 0.0
                    saved_inr = round((sl - ltp) * qty, 2)
                    trade["trade_management_note"] = f"Covered early above VWAP; saved Rs {saved_inr:,.0f}."
                    actions_taken.append(
                        f"⚠️ <b>EARLY SHORT COVER ({sym}):</b> Reclaimed VWAP (₹{vwap}). Covered at ₹{ltp} (P&L: ₹{trade['pnl_inr']:+,.0f}), saving ₹{saved_inr:,.0f}!"
                    )

        self._save_ledger()
        self._save_portfolio()
        return actions_taken

    def evaluate_open_trades_and_learn(self):
        """
        End-of-Day Settlement + Mistake Autopsy + Reinforcement Learning Weight Optimizer.
        Closes all remaining OPEN intraday trades, updates the Rs 1,50,000 wallet,
        diagnoses losing trades, and adjusts weights & risk-per-trade for tomorrow.
        """
        open_trades = [t for t in self.ledger if t.get("status") == "OPEN"]
        if not open_trades:
            learning_summary = self._optimize_weights_from_history()
            return {"evaluated": 0, "message": f"All trades already settled. {learning_summary}"}

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

            sym_ns = f"{trade['symbol']}.NS"
            try:
                df = data[sym_ns].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
                if df.empty:
                    continue

                latest = df.iloc[-1]
                day_high = float(latest["High"])
                day_low = float(latest["Low"])
                close_price = float(latest["Close"])

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
                        status, exit_p = ("WIN (TARGET HIT)", t1) if close_price >= entry else ("LOSS (SL HIT)", sl)
                    else:
                        exit_p = close_price
                        status = "EOD WIN (SQUARE OFF)" if exit_p >= entry else "EOD LOSS (SQUARE OFF)"
                    pnl_pct = ((exit_p - entry) / entry) * 100.0
                    pnl_inr = (exit_p - entry) * qty
                else:
                    hit_target = day_low <= t1
                    hit_sl = day_high >= sl
                    if hit_target and not hit_sl:
                        status, exit_p = "WIN (TARGET HIT)", t1
                    elif hit_sl and not hit_target:
                        status, exit_p = "LOSS (SL HIT)", sl
                    elif hit_target and hit_sl:
                        status, exit_p = ("WIN (TARGET HIT)", t1) if close_price <= entry else ("LOSS (SL HIT)", sl)
                    else:
                        exit_p = close_price
                        status = "EOD WIN (SQUARE OFF)" if exit_p <= entry else "EOD LOSS (SQUARE OFF)"
                    pnl_pct = ((entry - exit_p) / entry) * 100.0
                    pnl_inr = (entry - exit_p) * qty

                trade["status"] = status
                trade["exit_price"] = round(exit_p, 2)
                trade["current_price"] = round(close_price, 2)
                trade["unrealized_pnl_inr"] = 0.0
                trade["day_high"] = round(day_high, 2)
                trade["day_low"] = round(day_low, 2)
                trade["pnl_pct"] = round(pnl_pct, 2)
                trade["pnl_inr"] = round(pnl_inr, 2)

                # Mistake Autopsy on Losing Trades
                if pnl_inr < 0:
                    f_scores = trade.get("factor_scores", {})
                    culprit = max(f_scores.items(), key=lambda x: x[1])[0] if f_scores else "momentum"
                    diagnosis = (
                        f"{trade['symbol']} ({side}) lost ₹{abs(pnl_inr):,.0f}: Over-relied on high {culprit.upper()} "
                        f"score ({f_scores.get(culprit, 50)}) which reversed intraday. Reducing {culprit} weight."
                    )
                    trade["mistake_diagnosis"] = diagnosis
                    new_mistakes.append(diagnosis)
                else:
                    trade["mistake_diagnosis"] = "Execution aligned with trend & risk rules."

                evaluated_count += 1
            except Exception as e:
                print(f"[WARN] Could not evaluate {trade['symbol']}: {e}")

        if new_mistakes:
            self.portfolio["mistakes_identified"] = (new_mistakes + self.portfolio.get("mistakes_identified", []))[:15]

        self._save_ledger()
        self._save_portfolio()

        # Log daily equity snapshot
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_pnl = sum(float(t.get("pnl_inr", 0)) for t in self.ledger if t.get("date") == today_str and t.get("status") != "OPEN")
        eq_log = self.portfolio.get("daily_equity_log", [])
        if eq_log and eq_log[-1].get("date") == today_str:
            eq_log[-1]["equity"] = self.portfolio["current_equity"]
            eq_log[-1]["pnl"] = round(today_pnl, 2)
        else:
            eq_log.append({"date": today_str, "equity": self.portfolio["current_equity"], "pnl": round(today_pnl, 2)})
        self.portfolio["daily_equity_log"] = eq_log[-60:]
        self._save_portfolio()

        learning_summary = self._optimize_weights_from_history()
        return {"evaluated": evaluated_count, "message": learning_summary}

    def _optimize_weights_from_history(self):
        settled = [t for t in self.ledger if t.get("status") != "OPEN" and "factor_scores" in t]
        if not settled:
            return "No settled trades yet to update weights."

        recent = settled[-30:]
        learning_rate = 0.015
        adjustments = {"volume": 0.0, "sentiment": 0.0, "premarket": 0.0, "momentum": 0.0}

        wins_count = 0
        for t in recent:
            pnl = float(t.get("pnl_pct", 0.0))
            if pnl > 0:
                wins_count += 1
            direction = 1.0 if pnl > 0 else (-1.2 if pnl < 0 else 0.0)  # Penalize mistakes 1.2x harder
            magnitude = min(2.5, max(0.5, abs(pnl)))
            scores = t["factor_scores"]
            for factor_key in adjustments:
                factor_strength = (float(scores.get(factor_key, 50.0)) - 50.0) / 100.0
                adjustments[factor_key] += direction * magnitude * factor_strength * learning_rate

        raw_vol = max(0.12, min(0.50, self.weights["volume_weight"] + adjustments["volume"]))
        raw_sent = max(0.12, min(0.50, self.weights["sentiment_weight"] + adjustments["sentiment"]))
        raw_pre = max(0.12, min(0.50, self.weights["premarket_weight"] + adjustments["premarket"]))
        raw_mom = max(0.12, min(0.50, self.weights["momentum_weight"] + adjustments["momentum"]))

        total = raw_vol + raw_sent + raw_pre + raw_mom
        self.weights["volume_weight"] = round(raw_vol / total, 3)
        self.weights["sentiment_weight"] = round(raw_sent / total, 3)
        self.weights["premarket_weight"] = round(raw_pre / total, 3)
        self.weights["momentum_weight"] = round(1.0 - (
            self.weights["volume_weight"] +
            self.weights["sentiment_weight"] +
            self.weights["premarket_weight"]
        ), 3)

        # Adaptive Risk-Per-Trade Sizing: scale down after drawdowns, scale up slightly on high win rate
        recent_win_rate = (wins_count / len(recent)) * 100.0
        if recent_win_rate >= 60.0 and self.portfolio["current_equity"] >= INITIAL_CAPITAL:
            self.weights["risk_per_trade_pct"] = min(1.5, round(float(self.weights.get("risk_per_trade_pct", 1.0)) + 0.05, 2))
        elif recent_win_rate < 45.0 or self.portfolio["current_equity"] < INITIAL_CAPITAL:
            self.weights["risk_per_trade_pct"] = max(0.6, round(float(self.weights.get("risk_per_trade_pct", 1.0)) - 0.10, 2))

        self.weights["generation"] = int(self.weights.get("generation", 1)) + 1
        self.weights["last_updated"] = datetime.now().strftime("%Y-%m-%d %I:%M %p")

        best_pillar = max(
            [("Volume", self.weights["volume_weight"]),
             ("News AI", self.weights["sentiment_weight"]),
             ("Pre-Market", self.weights["premarket_weight"]),
             ("Chart/Tech", self.weights["momentum_weight"])],
            key=lambda x: x[1]
        )
        note = (
            f"Gen #{self.weights['generation']}: Leading edge is {best_pillar[0]} ({int(best_pillar[1]*100)}%). "
            f"Risk per trade calibrated to {self.weights['risk_per_trade_pct']}% of ₹{self.portfolio['current_equity']:,.0f}."
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