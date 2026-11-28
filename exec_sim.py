"""
Bar-by-bar execution simulation used by live position management, shadow-trade
labelling and the back-tester, so all three judge stops/targets the same way.

Conservative rules (they under-state, never over-state, performance):
  * Stop and target both inside one bar          -> assume the STOP hit first.
  * A bar that gaps through the stop at its open -> fill at the open (worse than the stop).
  * Target fills are booked at the target price (no credit for gap-through improvement).
"""
from typing import Iterable, Optional


def resolve_barrier_outcome(side_is_buy: bool, bars: Iterable[dict], sl: float, t1: float) -> dict:
    """
    ``bars`` are chronological dicts with keys open/high/low/close.
    Returns {"reason": "SL"|"T1"|"NONE", "exit_px": float|None, "bar_index": int|None,
             "gap": bool, "ambiguous": bool}
    """
    for i, b in enumerate(bars):
        o, h, l = float(b["open"]), float(b["high"]), float(b["low"])
        if side_is_buy:
            if o <= sl:
                return {"reason": "SL", "exit_px": o, "bar_index": i, "gap": True, "ambiguous": False}
            sl_hit, t1_hit = l <= sl, h >= t1
        else:
            if o >= sl:
                return {"reason": "SL", "exit_px": o, "bar_index": i, "gap": True, "ambiguous": False}
            sl_hit, t1_hit = h >= sl, l <= t1

        if sl_hit and t1_hit:
            return {"reason": "SL", "exit_px": float(sl), "bar_index": i, "gap": False, "ambiguous": True}
        if sl_hit:
            return {"reason": "SL", "exit_px": float(sl), "bar_index": i, "gap": False, "ambiguous": False}
        if t1_hit:
            return {"reason": "T1", "exit_px": float(t1), "bar_index": i, "gap": False, "ambiguous": False}
    return {"reason": "NONE", "exit_px": None, "bar_index": None, "gap": False, "ambiguous": False}


def worse_stop_fill(side_is_buy: bool, stop_price: float, last_price: Optional[float]) -> float:
    """A stop order fills at the stop or worse - never better than the last traded price."""
    if last_price is None:
        return float(stop_price)
    return float(min(stop_price, last_price)) if side_is_buy else float(max(stop_price, last_price))


def apply_slippage(price: float, is_buy_order: bool, slippage_pct: float) -> float:
    """Buy orders fill higher, sell orders fill lower."""
    s = slippage_pct / 100.0
    return float(price * (1.0 + s)) if is_buy_order else float(price * (1.0 - s))