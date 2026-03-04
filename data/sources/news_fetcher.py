"""
News sentiment fetcher using Google News RSS feeds.
Extracts headlines and computes sentiment using VADER and TextBlob.
"""

import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import feedparser
from loguru import logger

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False


class NewsFetcher:
    """Fetches and analyzes news sentiment from Google News RSS."""

    GOOGLE_NEWS_RSS = (
        "https://news.google.com/rss/search?"
        "q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    def __init__(self, fetch_delay: float = 1.5):
        self.fetch_delay = fetch_delay
        self.vader = None

        if _VADER_AVAILABLE:
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            self.vader = SentimentIntensityAnalyzer()

    def fetch_stock_news(
        self, symbol: str, company_name: str, max_articles: int = 20
    ) -> list[dict]:
        """
        Fetch recent news articles for a stock from Google News RSS.
        Returns list of dicts with title, published date, and source.
        """
        query = quote(f"{company_name} NSE stock")
        url = self.GOOGLE_NEWS_RSS.format(query=query)

        try:
            feed = feedparser.parse(url)

            articles = []
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=48)  # Last 48 hours

            for entry in feed.entries[:max_articles]:
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                # Filter to recent articles
                if pub_date and pub_date < cutoff:
                    continue

                articles.append({
                    "title": entry.get("title", ""),
                    "published": pub_date.isoformat() if pub_date else None,
                    "source": entry.get("source", {}).get("title", ""),
                })

            return articles

        except Exception as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            return []

    def analyze_sentiment(self, headlines: list[str]) -> dict:
        """
        Analyze sentiment of a list of headlines using VADER and TextBlob.
        Returns aggregated sentiment features.
        """
        if not headlines:
            return _neutral_sentiment()

        vader_scores = []
        textblob_polarities = []
        textblob_subjectivities = []

        for headline in headlines:
            if not headline.strip():
                continue

            # VADER sentiment
            if self.vader:
                score = self.vader.polarity_scores(headline)
                vader_scores.append(score["compound"])

            # TextBlob sentiment
            if _TEXTBLOB_AVAILABLE:
                blob = TextBlob(headline)
                textblob_polarities.append(blob.sentiment.polarity)
                textblob_subjectivities.append(blob.sentiment.subjectivity)

        if not vader_scores and not textblob_polarities:
            return _neutral_sentiment()

        # Compute aggregated features
        result = {
            "news_count": len(headlines),
        }

        if vader_scores:
            result["vader_compound_mean"] = sum(vader_scores) / len(vader_scores)
            result["vader_compound_max"] = max(vader_scores)
            result["vader_compound_min"] = min(vader_scores)
            positive = sum(1 for s in vader_scores if s > 0.05)
            negative = sum(1 for s in vader_scores if s < -0.05)
            result["vader_positive_ratio"] = positive / len(vader_scores)
            result["vader_negative_ratio"] = negative / len(vader_scores)
        else:
            result.update({
                "vader_compound_mean": 0.0,
                "vader_compound_max": 0.0,
                "vader_compound_min": 0.0,
                "vader_positive_ratio": 0.0,
                "vader_negative_ratio": 0.0,
            })

        if textblob_polarities:
            result["textblob_polarity"] = sum(textblob_polarities) / len(textblob_polarities)
            result["textblob_subjectivity"] = (
                sum(textblob_subjectivities) / len(textblob_subjectivities)
            )
        else:
            result["textblob_polarity"] = 0.0
            result["textblob_subjectivity"] = 0.0

        return result

    def fetch_and_analyze(
        self, symbol: str, company_name: str, target_date: str = None
    ) -> dict:
        """
        Fetch news and compute sentiment for a single stock.
        Returns a complete sentiment record ready for DB insertion.
        """
        if target_date is None:
            target_date = date.today().isoformat()

        articles = self.fetch_stock_news(symbol, company_name)
        headlines = [a["title"] for a in articles if a.get("title")]
        sentiment = self.analyze_sentiment(headlines)

        return {
            "date": target_date,
            "symbol": symbol,
            **sentiment,
        }

    def fetch_all_sentiment(
        self,
        stock_map: dict[str, str],
        target_date: str = None,
    ) -> list[dict]:
        """
        Fetch sentiment for all stocks with rate limiting.
        stock_map: {symbol: company_name}
        """
        if target_date is None:
            target_date = date.today().isoformat()

        results = []
        total = len(stock_map)

        for i, (symbol, company_name) in enumerate(stock_map.items()):
            logger.debug(f"Fetching news {i + 1}/{total}: {symbol}")
            record = self.fetch_and_analyze(symbol, company_name, target_date)
            results.append(record)

            if i < total - 1:
                time.sleep(self.fetch_delay)

        logger.info(f"Fetched sentiment for {len(results)} stocks")
        return results


def _neutral_sentiment() -> dict:
    """Return neutral sentiment values (no news or analysis unavailable)."""
    return {
        "news_count": 0,
        "vader_compound_mean": 0.0,
        "vader_compound_max": 0.0,
        "vader_compound_min": 0.0,
        "vader_positive_ratio": 0.0,
        "vader_negative_ratio": 0.0,
        "textblob_polarity": 0.0,
        "textblob_subjectivity": 0.0,
    }
