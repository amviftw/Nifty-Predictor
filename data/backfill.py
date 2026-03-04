"""
Historical data backfill module.
Fetches 2+ years of data for all Nifty 50 stocks, global indices, and fundamentals.
Run once before first use, then after each Nifty 50 rebalancing.
"""

from datetime import date

from loguru import logger

from config.settings import SETTINGS
from config.nifty50_tickers import get_symbols, get_company_name, NIFTY50_STOCKS
from data.storage.db_manager import DBManager
from data.sources.yahoo_fetcher import YahooFetcher
from data.sources.global_fetcher import GlobalFetcher
from data.sources.nse_fetcher import NSEFetcher


def run_backfill(start_date: str = None):
    """
    Execute full historical data backfill.

    Steps:
    1. Fetch OHLCV data for all Nifty 50 stocks
    2. Fetch global indices and macro data
    3. Fetch FII/DII data
    4. Fetch fundamentals for all stocks
    """
    if start_date is None:
        start_date = SETTINGS.BACKFILL_START

    end_date = date.today().isoformat()
    db = DBManager(SETTINGS.DB_PATH)

    logger.info(f"Starting backfill from {start_date} to {end_date}")

    # Step 1: Fetch OHLCV data
    logger.info("Step 1/4: Fetching OHLCV data for all Nifty 50 stocks...")
    yahoo = YahooFetcher()
    ohlcv_data = yahoo.fetch_ohlcv_batch(start_date, end_date)

    total_records = 0
    for symbol, df in ohlcv_data.items():
        records = yahoo.ohlcv_to_records(symbol, df)
        db.insert_ohlcv(records)
        total_records += len(records)

    logger.info(f"Inserted {total_records} OHLCV records for {len(ohlcv_data)} stocks")

    # Step 2: Fetch global indices
    logger.info("Step 2/4: Fetching global indices and macro data...")
    global_fetcher = GlobalFetcher()
    macro_df = global_fetcher.fetch_all_indices(start_date, end_date)

    if not macro_df.empty:
        macro_records = macro_df.to_dict("records")
        db.insert_macro(macro_records)
        logger.info(f"Inserted {len(macro_records)} macro data records")

    # Step 3: Fetch FII/DII data
    logger.info("Step 3/4: Fetching FII/DII data...")
    nse = NSEFetcher()
    fii_dii_data = nse.fetch_fii_dii(start_date, end_date)

    if fii_dii_data:
        # Update macro_data table with FII/DII values
        with db.connect() as conn:
            for record in fii_dii_data:
                conn.execute(
                    """UPDATE macro_data
                       SET fii_net_buy = ?, dii_net_buy = ?
                       WHERE date = ?""",
                    (record["fii_net_buy"], record["dii_net_buy"], record["date"]),
                )
        logger.info(f"Updated {len(fii_dii_data)} FII/DII records")

    # Step 4: Fetch fundamentals
    logger.info("Step 4/4: Fetching fundamentals for all stocks...")
    fundamentals = yahoo.fetch_all_fundamentals(delay=1.0)
    db.insert_fundamentals(fundamentals)
    logger.info(f"Inserted {len(fundamentals)} fundamental records")

    # Summary
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"  Stocks: {len(ohlcv_data)}")
    logger.info(f"  OHLCV records: {total_records}")
    logger.info(f"  Macro data days: {len(macro_df) if not macro_df.empty else 0}")
    logger.info(f"  FII/DII records: {len(fii_dii_data)}")
    logger.info(f"  Fundamentals: {len(fundamentals)}")
    logger.info(f"  Database: {SETTINGS.DB_PATH}")
    logger.info("=" * 60)
