#!/usr/bin/env python3
"""
One-time historical data backfill script.
Run this before first use to populate the database with 2+ years of data.

Usage:
    python -m scripts.backfill_data
    python -m scripts.backfill_data --start-date 2023-01-01
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from data.backfill import run_backfill


def main():
    parser = argparse.ArgumentParser(description="Backfill historical data")
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Start date for backfill (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NIFTY 50 PREDICTOR - DATA BACKFILL")
    logger.info("=" * 60)

    run_backfill(start_date=args.start_date)


if __name__ == "__main__":
    main()
