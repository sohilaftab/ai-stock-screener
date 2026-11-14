import os
import requests

class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    def send_message(self, html_text):
        """Sends an HTML-formatted alert to your Telegram account or channel."""
        if not self.bot_token or not self.chat_id:
            print("[WARN] Telegram credentials not set. Printing alert to console instead:\n")
            print(html_text)
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            print("[INFO] Telegram alert delivered successfully.")
            return True
        except requests.RequestException as e:
            print(f"[ERROR] Failed to send Telegram message: {e}")
            return False