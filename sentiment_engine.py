import urllib.parse
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class FinBERTSentimentEngine:
    def __init__(self, model_name="ProsusAI/finbert"):
        print("[INFO] Loading ProsusAI/finbert NLP model into memory...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Initialize Hugging Face text-classification pipeline on CPU (device=-1)
        self.nlp = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1
        )

    def fetch_google_news(self, symbol, max_headlines=5):
        """Fetches recent Indian financial news headlines for a stock ticker via Google News RSS."""
        query = urllib.parse.quote(f"{symbol} stock NSE India news when:2d")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(rss_url)
        headlines = []
        for entry in feed.entries[:max_headlines]:
            # Clean source suffix (e.g., " - Moneycontrol") if desired, or keep full title
            title = entry.title.rsplit(" - ", 1)[0] if " - " in entry.title else entry.title
            headlines.append(title)
        return headlines

    def analyze_stock_sentiment(self, symbol):
        """
        Scrapes headlines and runs FinBERT inference.
        Returns aggregate label (POSITIVE / NEGATIVE / NEUTRAL), confidence score, and top headline.
        """
        headlines = self.fetch_google_news(symbol, max_headlines=5)
        if not headlines:
            return {
                "sentiment": "NEUTRAL",
                "confidence": 0.0,
                "top_headline": "No major news catalyst in the last 48 hours."
            }

        # Run batch inference through FinBERT
        results = self.nlp(headlines)
        
        score_map = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        best_headline = headlines[0]
        highest_extremity = -1.0

        for headline, res in zip(headlines, results):
            label = res["label"].lower()
            score = float(res["score"])
            score_map[label] += score

            # Track the strongest non-neutral headline to display in the Telegram alert
            if label in ["positive", "negative"] and score > highest_extremity:
                highest_extremity = score
                best_headline = headline

        # Determine dominant sentiment
        dominant_label = max(score_map, key=score_map.get).upper()
        avg_confidence = round(score_map[dominant_label.lower()] / len(headlines), 2)

        return {
            "sentiment": dominant_label,
            "confidence": avg_confidence,
            "top_headline": best_headline
        }