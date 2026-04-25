"""
Dashboard data fetching layer.
Wraps existing fetchers (YahooFetcher, GlobalFetcher, NSEFetcher) and adds
new fetching for sectoral indices and supply chain factors.
All functions use @st.cache_data for smart caching.
"""

import sys
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from loguru import logger

# Add project root to path so we can import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.nifty50_tickers import NIFTY50_STOCKS, GLOBAL_INDICES, get_yahoo_tickers
from config.holidays import is_trading_day, prev_trading_day, days_to_expiry, next_fno_expiry
from data.sources.yahoo_fetcher import YahooFetcher
from data.sources.global_fetcher import GlobalFetcher
from data.sources.nse_fetcher import NSEFetcher
from dashboard.config import (
    SECTORAL_INDICES,
    SUPPLY_CHAIN_TICKERS,
    CACHE_TTL_SECONDS,
    TOP_MOVERS_COUNT,
    PERIOD_DAILY,
    PERIOD_WEEKLY,
)

IST = timezone(timedelta(hours=5, minutes=30))

# NSE regular session
_MARKET_OPEN = (9, 15)
_MARKET_CLOSE = (15, 30)


def is_market_open_now(now: datetime | None = None) -> bool:
    """True iff `now` (IST) falls inside the NSE regular trading session."""
    now = now or datetime.now(IST)
    if not is_trading_day(now.date()):
        return False
    hm = (now.hour, now.minute)
    return _MARKET_OPEN <= hm <= _MARKET_CLOSE


def market_freshness_key(now: datetime | None = None) -> str:
    """
    Cache-key bucket that drives `load_market_snapshot` re-fetches.

    - During NSE market hours: rotates every minute, so prices stay <=60s
      behind the live tape.
    - Outside market hours (after-hours, weekends, holidays): keyed on the
      most recent trading day, so a fresh fetch is forced as soon as a new
      session starts and the previous session's intraday tick can no longer
      masquerade as the close.
    """
    now = now or datetime.now(IST)
    if is_market_open_now(now):
        return f"live-{now:%Y%m%d-%H%M}"
    today = now.date()
    last_session = today if is_trading_day(today) else prev_trading_day(today)
    # Bucket by trading day + AM/PM so a request shortly after market close
    # picks up the freshly settled close instead of the last intraday cache
    # entry written seconds before 15:30.
    half = "am" if now.hour < 12 else "pm"
    return f"closed-{last_session.isoformat()}-{half}"


@dataclass
class MarketSnapshot:
    """All data needed to render the dashboard."""

    timestamp: datetime
    view: str  # "daily" or "weekly"

    # Nifty 50 index
    nifty50_close: float = 0.0
    nifty50_change_pct: float = 0.0
    nifty50_wow_pct: float = 0.0

    # Sectoral indices DataFrame: columns=[close, dod_pct, wow_pct, mom_pct]
    sectoral_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Stock-level data
    stock_changes: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_gainers: list = field(default_factory=list)
    top_losers: list = field(default_factory=list)

    # Macro
    india_vix: float = 0.0
    india_vix_change: float = 0.0
    usdinr: float = 0.0
    usdinr_change: float = 0.0
    fii_net_buy: float = 0.0
    dii_net_buy: float = 0.0

    # Macro time series for charts
    vix_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    usdinr_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    fii_dii_series: list = field(default_factory=list)

    # Global indices: {name: {close, ret_pct}}
    global_indices: dict = field(default_factory=dict)

    # Supply chain factors DataFrame: columns=[close, dod_pct, wow_pct]
    supply_chain: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Market status
    is_market_open: bool = False
    last_trading_date: str = ""

    # Breadth
    advance_count: int = 0
    decline_count: int = 0
    unchanged_count: int = 0


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching live market data...")
def load_market_snapshot(view: str = "daily", _freshness: str = "") -> MarketSnapshot:
    """Master function: fetches all data and returns a MarketSnapshot.

    `_freshness` participates in the cache key only — its value comes from
    `market_freshness_key()` and forces a re-fetch whenever the market state
    transitions (each minute during the session, each new trading day after
    close). It is intentionally unused inside the function body.
    """
    del _freshness
    now = datetime.now(IST)
    today = date.today()
    trading_day = today if is_trading_day(today) else prev_trading_day(today)

    # Always fetch 1mo to have enough data for WoW/MoM computations
    period = "1mo"

    snapshot = MarketSnapshot(
        timestamp=now,
        view=view,
        is_market_open=is_market_open_now(now),
        last_trading_date=trading_day.isoformat(),
    )

    # Fetch all data sources (sequentially to avoid rate-limiting)
    _populate_stock_data(snapshot, period)
    _populate_macro_data(snapshot, period)
    _populate_sectoral_data(snapshot, period)
    _populate_supply_chain(snapshot, period)

    return snapshot


def _populate_stock_data(snapshot: MarketSnapshot, period: str):
    """Fetch Nifty 50 stock prices and compute movers."""
    try:
        fetcher = YahooFetcher(max_retries=2, retry_delay=1.0)
        stock_data = fetcher.fetch_recent_ohlcv(period=period)

        if not stock_data:
            logger.warning("No stock data fetched")
            return

        records = []
        for symbol, df in stock_data.items():
            if df is None or df.empty:
                continue

            close_col = "close" if "close" in df.columns else None
            vol_col = "volume" if "volume" in df.columns else None
            if close_col is None:
                continue

            closes = df[close_col].dropna()
            if len(closes) < 2:
                continue

            latest_close = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2])
            dod_pct = ((latest_close - prev_close) / prev_close) * 100 if prev_close else 0

            # WoW: compare to ~5 trading days ago
            wow_pct = 0.0
            if len(closes) >= 6:
                week_ago_close = float(closes.iloc[-6])
                wow_pct = ((latest_close - week_ago_close) / week_ago_close) * 100 if week_ago_close else 0

            # MoM: first available vs latest
            mom_pct = 0.0
            if len(closes) >= 15:
                month_ago_close = float(closes.iloc[0])
                mom_pct = ((latest_close - month_ago_close) / month_ago_close) * 100 if month_ago_close else 0

            volume = int(df[vol_col].iloc[-1]) if vol_col and pd.notna(df[vol_col].iloc[-1]) else 0

            company = NIFTY50_STOCKS.get(symbol, (None, symbol, "Unknown"))[1]
            sector = NIFTY50_STOCKS.get(symbol, (None, None, "Unknown"))[2]

            records.append({
                "symbol": symbol,
                "company": company,
                "sector": sector,
                "close": latest_close,
                "dod_pct": round(dod_pct, 2),
                "wow_pct": round(wow_pct, 2),
                "mom_pct": round(mom_pct, 2),
                "volume": volume,
            })

        if not records:
            return

        df_all = pd.DataFrame(records)
        snapshot.stock_changes = df_all

        # Determine sort column based on view
        sort_col = "dod_pct" if snapshot.view == "daily" else "wow_pct"

        sorted_df = df_all.sort_values(sort_col, ascending=False)
        snapshot.top_gainers = sorted_df.head(TOP_MOVERS_COUNT).to_dict("records")
        snapshot.top_losers = sorted_df.tail(TOP_MOVERS_COUNT).sort_values(sort_col).to_dict("records")

        # Breadth
        change_col = sort_col
        snapshot.advance_count = int((df_all[change_col] > 0.1).sum())
        snapshot.decline_count = int((df_all[change_col] < -0.1).sum())
        snapshot.unchanged_count = len(df_all) - snapshot.advance_count - snapshot.decline_count

    except Exception as e:
        logger.error(f"Stock data fetch failed: {e}")


def _populate_macro_data(snapshot: MarketSnapshot, period: str):
    """Fetch global indices, VIX, USD/INR, FII/DII."""
    try:
        gf = GlobalFetcher(max_retries=2, retry_delay=1.0)
        records = gf.fetch_recent_indices(period=period)

        if not records:
            logger.warning("No macro data fetched")
            return

        # Latest record for headline metrics
        latest = records[-1]
        prev = records[-2] if len(records) >= 2 else {}

        snapshot.nifty50_close = latest.get("nifty50_close") or 0.0
        snapshot.nifty50_change_pct = (latest.get("nifty50_ret") or 0.0) * 100

        # WoW for Nifty
        if len(records) >= 6:
            week_ago = records[-6]
            n_close = latest.get("nifty50_close") or 0
            w_close = week_ago.get("nifty50_close") or 0
            if w_close:
                snapshot.nifty50_wow_pct = ((n_close - w_close) / w_close) * 100

        snapshot.india_vix = latest.get("india_vix") or 0.0
        snapshot.india_vix_change = (latest.get("india_vix_change") or 0.0) * 100
        snapshot.usdinr = latest.get("usdinr") or 0.0
        snapshot.usdinr_change = (latest.get("usdinr_change") or 0.0) * 100

        # VIX and USD/INR time series for charts
        vix_data = [{"date": r["date"], "India VIX": r.get("india_vix")} for r in records if r.get("india_vix")]
        snapshot.vix_series = pd.DataFrame(vix_data).set_index("date") if vix_data else pd.DataFrame()

        usdinr_data = [{"date": r["date"], "USD/INR": r.get("usdinr")} for r in records if r.get("usdinr")]
        snapshot.usdinr_series = pd.DataFrame(usdinr_data).set_index("date") if usdinr_data else pd.DataFrame()

        # Global indices
        from dashboard.config import GLOBAL_INDEX_DISPLAY
        for key, display_name in GLOBAL_INDEX_DISPLAY.items():
            ret_key = f"{key.lower()}_ret"
            ret_val = latest.get(ret_key)
            snapshot.global_indices[display_name] = {
                "ret_pct": round((ret_val or 0) * 100, 2),
            }

        # FII/DII
        try:
            nf = NSEFetcher(max_retries=2, retry_delay=1.0)
            fii_dii = nf.fetch_recent_fii_dii(lookback_days=10)
            if fii_dii:
                snapshot.fii_dii_series = fii_dii
                latest_fii = fii_dii[-1]
                snapshot.fii_net_buy = latest_fii.get("fii_net_buy", 0.0)
                snapshot.dii_net_buy = latest_fii.get("dii_net_buy", 0.0)
        except Exception as e:
            logger.warning(f"FII/DII fetch failed: {e}")

    except Exception as e:
        logger.error(f"Macro data fetch failed: {e}")


def _populate_sectoral_data(snapshot: MarketSnapshot, period: str):
    """Fetch sectoral index data."""
    try:
        tickers = list(SECTORAL_INDICES.values())
        ticker_str = " ".join(tickers)

        data = yf.download(
            ticker_str,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data.empty:
            logger.warning("No sectoral index data fetched")
            return

        records = []
        for name, ticker in SECTORAL_INDICES.items():
            try:
                if len(SECTORAL_INDICES) == 1:
                    close_series = data["Close"]
                else:
                    # Handle both MultiIndex formats
                    if isinstance(data.columns, pd.MultiIndex):
                        level_0 = data.columns.get_level_values(0).unique().tolist()
                        level_1 = data.columns.get_level_values(1).unique().tolist()
                        if ticker in level_0:
                            close_series = data[ticker]["Close"]
                        elif ticker in level_1:
                            close_series = data.xs(ticker, level=1, axis=1)["Close"] if "Close" in data.xs(ticker, level=1, axis=1).columns else data.xs(ticker, level=1, axis=1).iloc[:, 0]
                        else:
                            continue
                    else:
                        close_series = data["Close"]

                closes = close_series.dropna()
                if len(closes) < 2:
                    continue

                latest = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                dod = ((latest - prev) / prev) * 100 if prev else 0

                wow = 0.0
                if len(closes) >= 6:
                    w = float(closes.iloc[-6])
                    wow = ((latest - w) / w) * 100 if w else 0

                mom = 0.0
                if len(closes) >= 15:
                    m = float(closes.iloc[0])
                    mom = ((latest - m) / m) * 100 if m else 0

                records.append({
                    "Index": name,
                    "Close": round(latest, 2),
                    "DoD %": round(dod, 2),
                    "WoW %": round(wow, 2),
                    "1M %": round(mom, 2),
                })
            except Exception as e:
                logger.debug(f"Sectoral index {name} ({ticker}) failed: {e}")
                continue

        if records:
            snapshot.sectoral_data = pd.DataFrame(records)

    except Exception as e:
        logger.error(f"Sectoral data fetch failed: {e}")


def _populate_supply_chain(snapshot: MarketSnapshot, period: str):
    """Fetch supply chain / international factor data."""
    try:
        tickers = list(SUPPLY_CHAIN_TICKERS.values())
        ticker_str = " ".join(tickers)

        data = yf.download(
            ticker_str,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data.empty:
            logger.warning("No supply chain data fetched")
            return

        records = []
        for name, ticker in SUPPLY_CHAIN_TICKERS.items():
            try:
                if len(SUPPLY_CHAIN_TICKERS) == 1:
                    close_series = data["Close"]
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        level_0 = data.columns.get_level_values(0).unique().tolist()
                        level_1 = data.columns.get_level_values(1).unique().tolist()
                        if ticker in level_0:
                            close_series = data[ticker]["Close"]
                        elif ticker in level_1:
                            close_series = data.xs(ticker, level=1, axis=1)["Close"] if "Close" in data.xs(ticker, level=1, axis=1).columns else data.xs(ticker, level=1, axis=1).iloc[:, 0]
                        else:
                            continue
                    else:
                        close_series = data["Close"]

                closes = close_series.dropna()
                if len(closes) < 2:
                    continue

                latest = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                dod = ((latest - prev) / prev) * 100 if prev else 0

                wow = 0.0
                if len(closes) >= 6:
                    w = float(closes.iloc[-6])
                    wow = ((latest - w) / w) * 100 if w else 0

                records.append({
                    "Factor": name,
                    "Price": round(latest, 2),
                    "DoD %": round(dod, 2),
                    "WoW %": round(wow, 2),
                })
            except Exception as e:
                logger.debug(f"Supply chain {name} ({ticker}) failed: {e}")
                continue

        if records:
            snapshot.supply_chain = pd.DataFrame(records)

    except Exception as e:
        logger.error(f"Supply chain data fetch failed: {e}")
