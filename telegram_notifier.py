import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ALERT_CACHE_FILE = BASE_DIR / ".telegram_alert_cache.json"


class TelegramNotifier:
    def __init__(self, bot_token: str = "", chat_id: str = "", cooldown_sec: int = 300):
        self.bot_token = (
            bot_token
            or os.getenv("TELEGRAM_BOT_TOKEN", "")
            or os.getenv("BOT_TOKEN", "")
        ).strip()
        self.chat_id = (
            chat_id
            or os.getenv("TELEGRAM_CHAT_ID", "")
            or os.getenv("CHAT_ID", "")
        ).strip()
        self.cooldown_sec = int(cooldown_sec)
        self._recent_alerts = self._load_cache()

    @staticmethod
    def _load_cache() -> dict:
        if ALERT_CACHE_FILE.exists():
            try:
                with open(ALERT_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        now = time.time()
        pruned = {
            k: v for k, v in self._recent_alerts.items()
            if (now - float(v)) < max(self.cooldown_sec * 4, 1800)
        }
        self._recent_alerts = pruned
        try:
            with open(ALERT_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._recent_alerts, f)
        except Exception:
            pass

    def _is_duplicate(self, text: str) -> bool:
        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        now = time.time()
        last_ts = float(self._recent_alerts.get(digest, 0.0))
        if last_ts > 0 and (now - last_ts) < self.cooldown_sec:
            return True
        self._recent_alerts[digest] = now
        self._save_cache()
        return False

    def send_message(self, text: str, force: bool = False) -> bool:
        if not text:
            return False
        if not force and self._is_duplicate(text):
            print("[INFO] Skipped duplicate Telegram alert within cooldown window.")
            return False

        if not self.bot_token or not self.chat_id:
            print("[INFO] Telegram credentials not set; alert logged locally.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=8) as resp:
                ok = resp.status == 200
                if ok:
                    print("[INFO] Telegram alert delivered successfully.")
                return ok
        except Exception as e:
            print(f"[WARN] Telegram alert failed: {e}")
            return False

    def send_entry_alert(
        self,
        symbol: str,
        action: str,
        qty: int,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        cash_left: float,
        reason: str = "",
    ) -> bool:
        side_label = "BUY" if str(action).upper() == "BUY" else "SHORT"
        icon = "🟢" if side_label == "BUY" else "🔴"
        reason_line = f"\n<b>Note:</b> {reason[:90]}" if reason else ""
        msg = (
            f"{icon} <b>{symbol} — {side_label}</b>\n"
            f"<b>Qty:</b> {qty} @ ₹{entry_price:,.2f}\n"
            f"<b>SL:</b> ₹{stop_loss:,.2f} | <b>T1:</b> ₹{target_1:,.2f}\n"
            f"<b>Cash Left:</b> ₹{cash_left:,.0f}{reason_line}"
        )
        return self.send_message(msg)

    def send_exit_alert(
        self,
        symbol: str,
        action: str,
        status: str,
        entry_price: float,
        exit_price: float,
        net_pnl_inr: float,
        net_pnl_pct: float,
        equity: float,
    ) -> bool:
        icon = "✅" if net_pnl_inr >= 0 else "🛑"
        msg = (
            f"{icon} <b>{symbol} ({action}) — {status}</b>\n"
            f"<b>Entry → Exit:</b> ₹{entry_price:,.2f} → ₹{exit_price:,.2f}\n"
            f"<b>Net P&L:</b> ₹{net_pnl_inr:+,.2f} ({net_pnl_pct:+.2f}%)\n"
            f"<b>Equity:</b> ₹{equity:,.0f}"
        )
        return self.send_message(msg)