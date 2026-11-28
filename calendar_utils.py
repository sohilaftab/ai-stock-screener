"""
NSE derivatives expiry calendar (pure functions, easy to test).

Rules (NSE, effective Sept 2025 and still current in 2026):
  * Nifty weekly expiry  : every Tuesday.
  * All NSE monthly expiry (index + stock F&O): last Tuesday of the month.
  * If that day is a trading holiday, expiry moves to the previous trading day.
"""
from datetime import date, timedelta
import calendar


def _last_tuesday(year: int, month: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 1:  # Tuesday
        d -= timedelta(days=1)
    return d


def _shift_back_to_trading_day(d: date, is_trading_day) -> date:
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def expiry_status(d: date, is_trading_day) -> dict:
    """
    Returns {"is_weekly_expiry", "is_monthly_expiry", "is_expiry_day", "note"} for date ``d``.
    ``is_trading_day`` is a callable(date) -> bool (weekends/holidays are False).
    """
    monday = d - timedelta(days=d.weekday())
    candidate_tuesdays = [monday + timedelta(days=1), monday + timedelta(days=8)]  # this week, next week

    is_weekly = False
    for tue in candidate_tuesdays:
        if _shift_back_to_trading_day(tue, is_trading_day) == d:
            is_weekly = True

    month_candidates = {(d.year, d.month)}
    nxt = d + timedelta(days=10)
    month_candidates.add((nxt.year, nxt.month))
    is_monthly = any(
        _shift_back_to_trading_day(_last_tuesday(y, m), is_trading_day) == d for y, m in month_candidates
    )

    if is_monthly:
        note = "NSE MONTHLY F&O EXPIRY day (index + stock) - expect heavy-weight volatility"
    elif is_weekly:
        note = "Nifty weekly expiry day - expect index-linked volatility"
    else:
        note = "Normal NSE session"
    return {
        "is_weekly_expiry": bool(is_weekly),
        "is_monthly_expiry": bool(is_monthly),
        "is_expiry_day": bool(is_weekly or is_monthly),
        "note": note,
    }