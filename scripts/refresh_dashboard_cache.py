"""
Nightly refresh of the dashboard's precomputed caches.

Run by `.github/workflows/refresh_dashboard_cache.yml` after NSE close on
trading days. Writes:

  - storage/precomputed/fii_dii.parquet      (last 45 days of net flows)
  - storage/precomputed/market_caps.parquet  (current mcap per symbol)

The dashboard's runtime path prefers these files; falling back to live
fetches only when they're missing or too stale. Pre-computing them shaves
~3–6s off cold-start latency for users.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yfinance as yf
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sources.nse_fetcher import NSEFetcher
from dashboard.precomputed_cache import (
    ensure_dir,
    save_fii_dii,
    save_market_caps,
)
from dashboard.universe import EXPANDED_UNIVERSE


FII_DII_LOOKBACK_DAYS = 45


def refresh_fii_dii() -> int:
    """Pull ~45 days of FII/DII flows from nselib and write the parquet.

    Returns the number of records written so the CI step can surface a
    warning if the API blanked out today.
    """
    nf = NSEFetcher(max_retries=3, retry_delay=2.0)
    records = nf.fetch_recent_fii_dii(lookback_days=FII_DII_LOOKBACK_DAYS)
    if not records:
        logger.warning("FII/DII refresh returned 0 records — leaving parquet untouched")
        return 0
    save_fii_dii(records)
    logger.info(f"Wrote {len(records)} FII/DII records to parquet")
    return len(records)


def refresh_market_caps() -> int:
    """Sweep the expanded universe for market caps and write the parquet.

    Runs at CI time so the live dashboard never pays for the 250-ticker
    `fast_info` sweep. Concurrency is bounded — outside the live path Yahoo
    is happy to serve 16 parallel fast_info calls.
    """
    universe: dict[str, str] = {sym: info[0] for sym, info in EXPANDED_UNIVERSE.items()}

    def _one(item):
        sym, tkr = item
        try:
            t = yf.Ticker(tkr)
            mc = None
            try:
                fi = t.fast_info
                mc = getattr(fi, "market_cap", None) or (
                    fi.get("market_cap") if hasattr(fi, "get") else None
                )
            except Exception:
                mc = None
            if not mc:
                try:
                    mc = t.info.get("marketCap")
                except Exception:
                    mc = None
            return sym, float(mc) if mc else None
        except Exception:
            return sym, None

    caps: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for sym, mc in ex.map(_one, universe.items()):
            if mc and mc > 0:
                caps[sym] = mc

    if not caps:
        logger.warning("Market cap refresh returned 0 entries — leaving parquet untouched")
        return 0
    save_market_caps(caps)
    logger.info(f"Wrote market caps for {len(caps)} symbols to parquet")
    return len(caps)


def main() -> int:
    ensure_dir()
    fii = refresh_fii_dii()
    caps = refresh_market_caps()
    # Non-zero exit only when BOTH jobs returned nothing — half-success is
    # still useful and shouldn't fail the workflow.
    return 0 if (fii or caps) else 1


if __name__ == "__main__":
    sys.exit(main())
