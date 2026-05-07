"""
Expanded dashboard universe = Nifty 50 + Nifty Next 50 + Nifty Midcap 150.

The model-training pipeline still anchors on Nifty 50 (config/nifty50_tickers.py)
because feature stability and target backtests are validated on that set. The
*dashboard* surfaces a much wider universe so movers / heatmaps / sector
analytics reflect the broader large+midcap market, which is what the user
actually trades.

Single source of truth: import `EXPANDED_UNIVERSE` (or the helpers below) from
this module instead of mashing the three ticker dicts together inline.
"""

from __future__ import annotations

from config.nifty50_tickers import NIFTY50_STOCKS
from config.largecap_extra_tickers import LARGECAP_NEXT50_STOCKS
from config.midcap_tickers import MIDCAP_STOCKS


def _build_universe() -> dict[str, tuple[str, str, str]]:
    """Merge Nifty 50 + Next 50 + Midcap. Earlier dicts win on symbol overlap.

    Overlap is rare but possible during a rebalance window (a stock newly
    promoted to Nifty 50 may still appear in the curated midcap list). We
    keep the higher-tier metadata so a name like RELIANCE always reports as
    Nifty 50 even if it leaks into a lower list.
    """
    merged: dict[str, tuple[str, str, str]] = {}
    merged.update(MIDCAP_STOCKS)
    merged.update(LARGECAP_NEXT50_STOCKS)
    merged.update(NIFTY50_STOCKS)
    return merged


EXPANDED_UNIVERSE: dict[str, tuple[str, str, str]] = _build_universe()


def get_universe_yahoo_tickers() -> list[str]:
    return [info[0] for info in EXPANDED_UNIVERSE.values()]


def get_universe_symbols() -> list[str]:
    return list(EXPANDED_UNIVERSE.keys())


def universe_tier(symbol: str) -> str:
    """Return one of "Nifty 50", "Nifty Next 50", "Midcap", "Other"."""
    if symbol in NIFTY50_STOCKS:
        return "Nifty 50"
    if symbol in LARGECAP_NEXT50_STOCKS:
        return "Nifty Next 50"
    if symbol in MIDCAP_STOCKS:
        return "Midcap"
    return "Other"


def get_company(symbol: str) -> str:
    return EXPANDED_UNIVERSE.get(symbol, (None, symbol, "Unknown"))[1]


def get_sector(symbol: str) -> str:
    return EXPANDED_UNIVERSE.get(symbol, (None, None, "Unknown"))[2]
