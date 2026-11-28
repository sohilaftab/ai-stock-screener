"""
FinBERT news-sentiment engine for NSE stocks.

Fixes vs. the previous version
------------------------------
* main.py called ``analyze_candidates`` / ``analyze_stock_sentiment(sym, headlines)`` which
  did not exist / had the wrong signature; the TypeError was swallowed, so every stock
  silently scored NEUTRAL.  Both entry points now exist and are covered by tests.
* Uses full class probabilities (P(pos) - P(neg)) instead of the argmax label only.
* One batched model forward pass for all headlines; RSS fetching runs in a thread pool.
* Recency weighting (12h half-life), de-duplication, and down-weighting of market-wide
  "Sensex jumps / stocks to watch" headlines that say nothing about one stock.
* 10-minute per-symbol cache, hard network timeouts.
* Failures are visible: every result carries a ``source`` and the batch logs stats
  instead of quietly degrading to NEUTRAL.
* Lazy model load, and a keyword fallback if torch/transformers are not installed.
"""
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from lexicon import BEARISH_KEYWORDS, BULLISH_KEYWORDS, GENERIC_HEADLINE_PATTERNS

try:
    import feedparser
    HAS_FEEDPARSER = True
except Exception:  # pragma: no cover
    HAS_FEEDPARSER = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:  # pragma: no cover
    HAS_REQUESTS = False

NEUTRAL_BAND = 0.20            # |net| below this => NEUTRAL
RECENCY_HALF_LIFE_HOURS = 12.0
GENERIC_WEIGHT = 0.25
_GENERIC_RE = re.compile("|".join(GENERIC_HEADLINE_PATTERNS), re.IGNORECASE)


class FinBERTSentimentEngine:
    def __init__(
        self,
        model_name="ProsusAI/finbert",
        classifier=None,
        fetch_entries_fn=None,
        cache_ttl_sec=600,
        max_headlines=6,
        max_workers=6,
        eager_load=False,
    ):
        self.model_name = model_name
        self._classifier = classifier          # callable(list[str]) -> list[list[{"label","score"}]]
        self._fetch_entries = fetch_entries_fn or self._fetch_entries_google
        self.cache_ttl_sec = cache_ttl_sec
        self.max_headlines = max_headlines
        self.max_workers = max_workers
        self._cache = {}
        self.stats = {"fetch_ok": 0, "fetch_fail": 0, "no_news": 0, "model_errors": 0, "fallback_keyword": 0}
        if eager_load and self._classifier is None:
            self._ensure_classifier()

    # ------------------------------------------------------------------ model
    def _ensure_classifier(self):
        """Lazy-load FinBERT; fall back to keywords if the stack is not installed."""
        if self._classifier is not None:
            return bool(self._classifier)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            print(f"[INFO] Loading {self.model_name} NLP model into memory...")
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            nlp = pipeline("text-classification", model=mdl, tokenizer=tok, device=-1, top_k=None,
                           truncation=True, max_length=128)

            def _clf(texts):
                out = nlp(texts, batch_size=16)
                # pipeline returns list[dict] for one str, list[list[dict]] for a list
                if out and isinstance(out[0], dict):
                    out = [out]
                return out

            self._classifier = _clf
            return True
        except Exception as e:
            print(f"[WARN] FinBERT unavailable ({e}); using keyword fallback.")
            self._classifier = False
            return False

    def _classify(self, headlines):
        """Returns list of (p_pos, p_neg, p_neu) aligned with ``headlines``."""
        if not headlines:
            return []
        if self._ensure_classifier() and self._classifier:
            try:
                raw = self._classifier(list(headlines))
                probs = []
                for scores in raw:
                    m = {s["label"].lower(): float(s["score"]) for s in scores}
                    probs.append((m.get("positive", 0.0), m.get("negative", 0.0), m.get("neutral", 0.0)))
                if len(probs) == len(headlines):
                    return probs
            except Exception as e:
                self.stats["model_errors"] += 1
                print(f"[WARN] FinBERT inference failed ({e}); keyword fallback for this batch.")
        return [self._keyword_probs(h) for h in headlines]

    def _keyword_probs(self, headline):
        self.stats["fallback_keyword"] += 1
        words = {w.strip(".,:;!?()[]\"'").lower() for w in headline.split()}
        net = len(words & BULLISH_KEYWORDS) - len(words & BEARISH_KEYWORDS)
        if net >= 1:
            return (0.6, 0.1, 0.3)
        if net <= -1:
            return (0.1, 0.6, 0.3)
        return (0.1, 0.1, 0.8)

    # ------------------------------------------------------------------- news
    def _fetch_entries_google(self, symbol):
        """Returns [(title, published_datetime_utc_or_None)] from Google News RSS (timeout + 1 retry)."""
        if not (HAS_FEEDPARSER and HAS_REQUESTS):
            raise RuntimeError("feedparser/requests not installed")
        query = urllib.parse.quote(f"{symbol} stock NSE India news when:2d")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        last_err = None
        for _attempt in range(2):
            try:
                resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)
                out = []
                for e in feed.entries[: self.max_headlines * 2]:
                    title = e.title.rsplit(" - ", 1)[0] if " - " in e.title else e.title
                    pub = None
                    if getattr(e, "published_parsed", None):
                        pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                    out.append((title.strip(), pub))
                return out
            except Exception as ex:
                last_err = ex
                time.sleep(0.4)
        raise RuntimeError(f"news fetch failed: {last_err}")

    def fetch_google_news(self, symbol, max_headlines=5):
        """Backwards-compatible: list of headline strings."""
        try:
            return [t for t, _ in self._fetch_entries(symbol)][:max_headlines]
        except Exception:
            return []

    # -------------------------------------------------------------- aggregation
    @staticmethod
    def _age_hours(pub, now):
        if pub is None:
            return 12.0
        return max(0.0, (now - pub).total_seconds() / 3600.0)

    def _aggregate(self, entries, probs):
        """entries: [(title, pub)], probs: [(pos, neg, neu)] -> dict."""
        now = datetime.now(timezone.utc)
        num = den = 0.0
        best_title, best_extremity = entries[0][0], -1.0
        for (title, pub), (pp, pn, _pu) in zip(entries, probs):
            net_i = pp - pn
            w = max(0.5 ** (self._age_hours(pub, now) / RECENCY_HALF_LIFE_HOURS), 0.05)
            if _GENERIC_RE.search(title):
                w *= GENERIC_WEIGHT
            num += w * net_i
            den += w
            extremity = abs(net_i) * w
            if extremity > best_extremity:
                best_extremity, best_title = extremity, title
        net = num / den if den > 0 else 0.0
        if net >= NEUTRAL_BAND:
            label = "POSITIVE"
        elif net <= -NEUTRAL_BAND:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        confidence = round(min(1.0, abs(net)), 3) if label != "NEUTRAL" else 0.5
        return {"sentiment": label, "confidence": confidence, "net_score": round(net, 3),
                "top_headline": best_title, "n_headlines": len(entries)}

    def _dedupe(self, entries):
        seen, out = set(), []
        for title, pub in entries:
            key = re.sub(r"\W+", " ", title.lower()).strip()[:90]
            if key and key not in seen:
                seen.add(key)
                out.append((title, pub))
        return out[: self.max_headlines]

    # --------------------------------------------------------------- public API
    def analyze_stock_sentiment(self, symbol, headlines=None):
        """
        Single-symbol analysis.  ``headlines`` (optional list[str]) lets a caller supply
        pre-fetched headlines (kept for backwards compatibility with the old call site).
        """
        return self._analyze_many([symbol], {symbol: headlines} if headlines else None)[symbol]

    def analyze_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the columns the trade engine reads:
          raw_sentiment, AI Confidence (0-100), Why Is It Moving? (Latest News),
          plus sentiment_net, sentiment_source, headline_count for diagnostics.
        """
        if df is None or df.empty:
            return df
        symbols = [str(s) for s in df["symbol"].tolist()]
        results = self._analyze_many(symbols)

        out = df.copy()
        out["raw_sentiment"] = [results[s]["sentiment"] for s in symbols]
        out["AI Confidence"] = [round(max(10.0, min(99.0, results[s]["confidence"] * 100.0)), 1) for s in symbols]
        out["Why Is It Moving? (Latest News)"] = [results[s]["top_headline"] for s in symbols]
        out["sentiment_net"] = [results[s]["net_score"] for s in symbols]
        out["sentiment_source"] = [results[s]["source"] for s in symbols]
        out["headline_count"] = [results[s]["n_headlines"] for s in symbols]

        src = out["sentiment_source"].value_counts().to_dict()
        out.attrs["sentiment_stats"] = {**self.stats, "sources": src}
        scored = src.get("FINBERT", 0) + src.get("KEYWORD_FALLBACK", 0)
        print(f"[SENTIMENT] {scored}/{len(symbols)} scored from news | sources={src}")
        return out

    # ---------------------------------------------------------------- internals
    def _analyze_many(self, symbols, supplied=None):
        now = time.time()
        results, to_fetch = {}, []
        for s in symbols:
            cached = self._cache.get(s)
            if cached and now - cached[0] < self.cache_ttl_sec and not supplied:
                results[s] = cached[1]
            else:
                to_fetch.append(s)

        entries_map = {}
        if supplied:
            for s in to_fetch:
                entries_map[s] = [(h, None) for h in (supplied.get(s) or [])]
        elif to_fetch:
            def _one(sym):
                try:
                    return sym, self._fetch_entries(sym), None
                except Exception as e:
                    return sym, [], e

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for sym, ents, err in pool.map(_one, to_fetch):
                    if err is not None:
                        self.stats["fetch_fail"] += 1
                        results[sym] = self._empty("FETCH_ERROR")
                        continue
                    self.stats["fetch_ok"] += 1
                    entries_map[sym] = ents

        # Batch every unique headline through the model exactly once.
        deduped = {s: self._dedupe(e) for s, e in entries_map.items()}
        unique = list({t for ents in deduped.values() for t, _ in ents})
        probs_by_title = dict(zip(unique, self._classify(unique))) if unique else {}
        used_model = bool(self._classifier)

        for s, ents in deduped.items():
            if not ents:
                self.stats["no_news"] += 1
                results[s] = self._empty("NO_NEWS")
                continue
            res = self._aggregate(ents, [probs_by_title[t] for t, _ in ents])
            res["source"] = "FINBERT" if used_model else "KEYWORD_FALLBACK"
            results[s] = res
            if not supplied:
                self._cache[s] = (now, res)
        return results

    @staticmethod
    def _empty(source):
        headline = ("No major news catalyst in the last 48 hours." if source == "NO_NEWS"
                    else "News feed unavailable.")
        return {"sentiment": "NEUTRAL", "confidence": 0.0, "net_score": 0.0,
                "top_headline": headline, "n_headlines": 0, "source": source}