"""
Daily data refresh pipeline.
Runs at 9:05 AM IST before market open.
Fetches previous day's data and current news sentiment.
"""

from datetime import date, timedelta

from loguru import logger

from config.settings import SETTINGS
from config.holidays import is_nse_holiday
from config.nifty50_tickers import get_symbols, get_company_name, NIFTY50_STOCKS
from data.storage.db_manager import DBManager
from data.sources.yahoo_fetcher import YahooFetcher
from data.sources.global_fetcher import GlobalFetcher
from data.sources.nse_fetcher import NSEFetcher
from data.sources.news_fetcher import NewsFetcher


class DailyPipeline:
    """Orchestrates daily data refresh for the prediction system."""

    def __init__(self, db_path=None):
        self.db = DBManager(db_path or SETTINGS.DB_PATH)
        self.yahoo = YahooFetcher()
        self.global_fetcher = GlobalFetcher()
        self.nse = NSEFetcher()
        self.news = NewsFetcher(fetch_delay=SETTINGS.NEWS_FETCH_DELAY_SECS)

    def run(self) -> bool:
        """
        Execute full daily data refresh.
        Returns True if successful, False otherwise.
        """
        today = date.today()

        # Check if today is a trading day
        if is_nse_holiday(today):
            logger.info(f"Today ({today}) is not a trading day. Skipping pipeline.")
            return False

        logger.info(f"Starting daily pipeline for {today}")
        success = True

        # Step 1: Refresh stock OHLCV (last 5 days to catch any gaps)
        try:
            logger.info("Refreshing stock OHLCV data...")
            ohlcv_data = self.yahoo.fetch_recent_ohlcv(period="5d")
            total = 0
            for symbol, df in ohlcv_data.items():
                records = self.yahoo.ohlcv_to_records(symbol, df)
                self.db.insert_ohlcv(records)
                total += len(records)
            logger.info(f"Refreshed OHLCV: {total} records for {len(ohlcv_data)} stocks")
        except Exception as e:
            logger.error(f"OHLCV refresh failed: {e}")
            success = False

        # Step 2: Refresh global indices
        try:
            logger.info("Refreshing global indices...")
            macro_records = self.global_fetcher.fetch_recent_indices(period="5d")
            if macro_records:
                self.db.insert_macro(macro_records)
                logger.info(f"Refreshed {len(macro_records)} macro records")
        except Exception as e:
            logger.error(f"Global indices refresh failed: {e}")

        # Step 3: Refresh FII/DII
        try:
            logger.info("Refreshing FII/DII data...")
            fii_dii = self.nse.fetch_recent_fii_dii(lookback_days=5)
            if fii_dii:
                with self.db.connect() as conn:
                    for record in fii_dii:
                        conn.execute(
                            """UPDATE macro_data
                               SET fii_net_buy = ?, dii_net_buy = ?
                               WHERE date = ?""",
                            (record["fii_net_buy"], record["dii_net_buy"],
                             record["date"]),
                        )
                logger.info(f"Updated {len(fii_dii)} FII/DII records")
        except Exception as e:
            logger.warning(f"FII/DII refresh failed (non-critical): {e}")

        # Step 4: Refresh news sentiment
        try:
            logger.info("Fetching news sentiment...")
            stock_map = {
                sym: get_company_name(sym) for sym in get_symbols()
            }
            sentiment_records = self.news.fetch_all_sentiment(
                stock_map, target_date=today.isoformat()
            )
            self.db.insert_sentiment(sentiment_records)
            logger.info(f"Stored sentiment for {len(sentiment_records)} stocks")
        except Exception as e:
            logger.warning(f"News sentiment fetch failed (non-critical): {e}")

        # Step 5: Refresh fundamentals if Monday
        if today.weekday() == 0:  # Monday
            try:
                logger.info("Refreshing fundamentals (weekly)...")
                fundamentals = self.yahoo.fetch_all_fundamentals(delay=1.0)
                self.db.insert_fundamentals(fundamentals)
                logger.info(f"Updated {len(fundamentals)} fundamental records")
            except Exception as e:
                logger.warning(f"Fundamentals refresh failed (non-critical): {e}")

        if success:
            logger.info("Daily pipeline completed successfully")
        else:
            logger.warning("Daily pipeline completed with some errors")

        return success
