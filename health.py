"""Rate-limited alerting, failure counters and a file/env kill switch."""
import os
import time

HALT_FILE = "HALT_TRADING"


def is_halted() -> bool:
    """Kill switch: create a file named HALT_TRADING (or set PULSEAI_HALT=1) to stop NEW entries."""
    return os.path.exists(HALT_FILE) or os.getenv("PULSEAI_HALT", "").strip() in ("1", "true", "TRUE", "yes")


class HealthMonitor:
    def __init__(self, notifier=None, min_alert_gap_sec: int = 1800, failure_threshold: int = 3):
        self.notifier = notifier
        self.min_alert_gap_sec = min_alert_gap_sec
        self.failure_threshold = failure_threshold
        self._last_alert = {}
        self.failures = {}
        self.last_success = {}

    def record_success(self, component: str):
        self.failures[component] = 0
        self.last_success[component] = time.time()

    def record_failure(self, component: str, err) -> int:
        self.failures[component] = self.failures.get(component, 0) + 1
        n = self.failures[component]
        if n >= self.failure_threshold:
            self.alert(f"fail:{component}", f"⚠️ <b>PulseAI health</b>: <code>{component}</code> failed {n}x in a row.\n{str(err)[:300]}")
        return n

    def alert(self, key: str, message: str) -> bool:
        """Send at most one alert per key per ``min_alert_gap_sec``."""
        now = time.time()
        if now - self._last_alert.get(key, 0.0) < self.min_alert_gap_sec:
            return False
        self._last_alert[key] = now
        if self.notifier is not None:
            try:
                self.notifier.send_message(message)
            except Exception:
                pass
        else:
            print(message)
        return True