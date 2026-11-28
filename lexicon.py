"""Shared keyword lexicon (used as the live-news fallback and when FinBERT is unavailable)."""

BULLISH_KEYWORDS = {
    "surge", "soar", "jump", "gain", "profit", "beat", "order", "win", "won",
    "contract", "approval", "upgrade", "buy", "bullish", "growth", "expansion",
    "dividend", "bonus", "acquisition", "partnership", "record", "strong", "rally",
}
BEARISH_KEYWORDS = {
    "fall", "drop", "plunge", "slump", "loss", "miss", "downgrade", "sell",
    "bearish", "fraud", "probe", "raid", "penalty", "fine", "ban", "recall",
    "default", "debt", "resign", "cut", "weak", "crash", "warning", "lawsuit",
}

# Headlines that describe the whole market / lists of stocks carry almost no
# stock-specific information.  They are down-weighted, not dropped.
GENERIC_HEADLINE_PATTERNS = (
    r"\bsensex\b", r"\bnifty\b", r"stocks? to (watch|buy|sell)", r"market (today|live|update)",
    r"top (gainers|losers)", r"(opening|closing) bell", r"live updates?", r"\bpicks?\b.*\bstocks?\b",
    r"\bstocks? in news\b", r"\bbuzzing stocks\b", r"\bweekly recap\b",
)