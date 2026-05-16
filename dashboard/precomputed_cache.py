"""
Precomputed-data loader for the dashboard.

A nightly GitHub Action (.github/workflows/refresh_dashboard_cache.yml) runs
`scripts/refresh_dashboard_cache.py` and commits two parquet files to
`storage/precomputed/`:

  - fii_dii.parquet      — last ~45 days of FII/DII net flows
  - market_caps.parquet  — market cap per symbol for the expanded universe

Reading these at runtime collapses two of the dashboard's worst cold-start
hogs (a 30-day nselib loop and a 250+ ticker `fast_info` sweep) into a single
pandas read. Live fetches are only triggered for today's incremental data or
when the precomputed file is missing/stale.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

_PRECOMPUTED_DIR = Path(__file__).resolve().parent.parent / "storage" / "precomputed"

FII_DII_PATH = _PRECOMPUTED_DIR / "fii_dii.parquet"
MARKET_CAPS_PATH = _PRECOMPUTED_DIR / "market_caps.parquet"


def ensure_dir() -> Path:
    _PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    return _PRECOMPUTED_DIR


def load_fii_dii(max_age_days: int = 3) -> list[dict] | None:
    """Return precomputed FII/DII records, or None if missing/too stale.

    `max_age_days` guards against running off a parquet file that hasn't been
    refreshed by CI for too long — at that point the runtime should fall back
    to the live nselib loop rather than serve a week-old WoW number.
    """
    if not FII_DII_PATH.exists():
        return None
    try:
        df = pd.read_parquet(FII_DII_PATH)
        if df.empty:
            return None
        latest = pd.to_datetime(df["date"]).max().date()
        if (date.today() - latest).days > max_age_days:
            logger.info(
                f"FII/DII parquet is stale (latest={latest}, age>{max_age_days}d); "
                "falling back to live fetch"
            )
            return None
        df = df.sort_values("date")
        records = df.to_dict("records")
        for r in records:
            d = r.get("date")
            if hasattr(d, "isoformat"):
                r["date"] = d.isoformat()
            else:
                r["date"] = str(d)[:10]
        return records
    except Exception as e:
        logger.warning(f"Failed to read precomputed FII/DII: {e}")
        return None


def save_fii_dii(records: list[dict]) -> None:
    """Persist FII/DII records to the precomputed parquet (used by the cron job)."""
    if not records:
        return
    ensure_dir()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df.to_parquet(FII_DII_PATH, index=False)


def merge_today_into_fii_dii(
    base: list[dict], today_record: dict | None
) -> list[dict]:
    """Splice a freshly fetched today-record onto the precomputed baseline.

    The parquet typically lags by a calendar day (or more on holidays). When
    today's print is available we want the WoW/WTD numbers to reflect it
    without forcing the full 30-day nselib loop on every page load.
    """
    if not today_record:
        return base
    today_iso = today_record.get("date") or date.today().isoformat()
    deduped = [r for r in base if (r.get("date") or "")[:10] != today_iso[:10]]
    deduped.append(today_record)
    deduped.sort(key=lambda r: r.get("date") or "")
    return deduped


def load_market_caps(max_age_days: int = 7) -> dict[str, float] | None:
    """Return precomputed market caps, or None if missing/too stale.

    Staleness is judged off file mtime — `df.attrs` doesn't round-trip through
    parquet, so embedding the refresh timestamp inside the frame would silently
    get dropped on read.
    """
    if not MARKET_CAPS_PATH.exists():
        return None
    try:
        import time
        age_days = (time.time() - MARKET_CAPS_PATH.stat().st_mtime) / 86400.0
        if age_days > max_age_days:
            logger.info(
                f"market_caps parquet is {age_days:.1f}d old (>{max_age_days}d); "
                "falling back to live fetch"
            )
            return None
        df = pd.read_parquet(MARKET_CAPS_PATH)
        if df.empty:
            return None
        out = dict(zip(df["symbol"], df["market_cap"].astype(float)))
        return {s: c for s, c in out.items() if c and c > 0}
    except Exception as e:
        logger.warning(f"Failed to read precomputed market caps: {e}")
        return None


def save_market_caps(caps: dict[str, float]) -> None:
    """Persist market caps to the precomputed parquet (used by the cron job)."""
    if not caps:
        return
    ensure_dir()
    df = pd.DataFrame(
        [{"symbol": s, "market_cap": float(c)} for s, c in caps.items() if c]
    )
    df.to_parquet(MARKET_CAPS_PATH, index=False)
