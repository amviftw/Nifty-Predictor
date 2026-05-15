"""
Lightweight pickle-based disk cache for the dashboard.

Streamlit's `@st.cache_data` is process-local — it's lost on every server
restart and across sessions. Slow-moving data the dashboard depends on
(2y/max historical bars, market caps, sector weekly bars) costs multiple
seconds of yfinance round-trips on every cold start, which is what users
feel as "the dashboard is slow to load".

This module persists those payloads to disk under
`storage/dashboard_cache/`, keyed by a caller-supplied name + TTL bucket
string (typically a calendar-day token like "2026-05-15"). The bucket is
part of the filename, so once a new day starts the old file is simply
ignored — no manual eviction logic needed.

Usage::

    @disk_cached(name="nifty_history_max", ttl_hours=6)
    def _fetch_nifty_history(_bucket: str = "") -> pd.DataFrame:
        ...
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from datetime import date
from functools import wraps
from pathlib import Path

from loguru import logger


_CACHE_DIR = Path(__file__).resolve().parent.parent / "storage" / "dashboard_cache"

# Cache-key arg names ignored by `disk_cached`. The dashboard's fetchers
# accept a `_bucket=<minute-bucket>` (and similar) purely to wire Streamlit's
# in-process cache; on the disk layer we want a stable per-day key instead,
# otherwise every minute-tick during market hours invalidates the cache and
# forces a fresh yfinance round-trip.
_IGNORED_KWARG_NAMES = {"_bucket", "_day_bucket", "_minute_bucket"}


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_path(name: str, key: str) -> Path:
    safe_key = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return _ensure_cache_dir() / f"{name}__{safe_key}.pkl"


def disk_get(name: str, key: str, ttl_seconds: float) -> object | None:
    """Return the cached payload for (name, key) if fresh, else None."""
    path = _cache_path(name, key)
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > ttl_seconds:
            return None
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.debug(f"disk_cache read failed for {name}/{key}: {e}")
        return None


def disk_put(name: str, key: str, payload: object) -> None:
    """Write payload to the disk cache. Best-effort — failures are swallowed."""
    path = _cache_path(name, key)
    try:
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as e:
        logger.debug(f"disk_cache write failed for {name}/{key}: {e}")


def disk_cached(name: str, ttl_hours: float = 6.0):
    """Decorator: cache the function's return value on disk.

    The cache key is built from positional args + filtered keyword args via
    `repr()`. Bucket-style kwargs listed in `_IGNORED_KWARG_NAMES` are
    *stripped* from the key and replaced with today's date — they exist only
    to drive Streamlit's in-process cache, and including them here would
    rotate the disk key every minute during market hours and force a
    redundant yfinance fetch on every tick.
    """
    ttl_seconds = ttl_hours * 3600.0

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                positional = [repr(a) for a in args]
                filtered_kwargs = [
                    f"{k}={v!r}"
                    for k, v in sorted(kwargs.items())
                    if k not in _IGNORED_KWARG_NAMES
                ]
                # The calendar-day token gives us automatic daily rollover
                # without depending on the caller passing a bucket.
                day_token = date.today().isoformat()
                key = "|".join(positional + filtered_kwargs + [f"day={day_token}"]) or "_"
            except Exception:
                # If args aren't repr-able, skip the cache entirely
                return fn(*args, **kwargs)

            cached = disk_get(name, key, ttl_seconds)
            if cached is not None:
                return cached

            value = fn(*args, **kwargs)
            if value is not None:
                disk_put(name, key, value)
            return value

        return wrapper

    return decorator
