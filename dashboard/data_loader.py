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

# NSE regular session (IST)
_MARKET_OPEN = (9, 15)
_MARKET_CLOSE = (15, 30)


def is_market_open_now(now: datetime | None = None) -> bool:
    """True iff `now` (IST) falls inside the NSE regular trading session."""
    now = now or datetime.now(IST)
    if not is_trading_day(now.date()):
        return False
    hm = (now.hour, now.minute)
    return _MARKET_OPEN <= hm <= _MARKET_CLOSE


@dataclass
class MarketSnapshot:
    """All data needed to render the dashboard."""

    timestamp: datetime
    view: str  # "daily" or "weekly"

    # Nifty 50 index
    nifty50_close: float = 0.0
    nifty50_change_pct: float = 0.0
    nifty50_wow_pct: float = 0.0
    # Anchor used for the WoW number: the close (and its date) from the last
    # trading day strictly before the current calendar week's Monday.
    # Surfaced so the dashboard can render "WoW: +0.4% (vs Fri 25-Apr 23,892)"
    # and users don't have to guess what the comparison is.
    nifty50_wow_anchor_close: float = 0.0
    nifty50_wow_anchor_date: str = ""

    # Sectoral indices DataFrame: columns=[close, dod_pct, wow_pct, mom_pct]
    sectoral_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Stock-level data
    stock_changes: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_gainers: list = field(default_factory=list)
    top_losers: list = field(default_factory=list)

    # Macro
    india_vix: float = 0.0
    india_vix_change: float = 0.0
    india_vix_wow_pct: float = 0.0
    india_vix_wow_anchor_close: float = 0.0
    india_vix_wow_anchor_date: str = ""
    usdinr: float = 0.0
    usdinr_change: float = 0.0
    usdinr_wow_pct: float = 0.0
    usdinr_wow_anchor_close: float = 0.0
    usdinr_wow_anchor_date: str = ""
    fii_net_buy: float = 0.0
    dii_net_buy: float = 0.0
    # Week-cycle flow aggregates: cumulative ₹ cr from Monday-of-this-week to
    # the latest available print, plus the same metric for the prior calendar
    # week (used as the WoW anchor on the headline chip).
    fii_wtd: float = 0.0
    dii_wtd: float = 0.0
    fii_prev_week: float = 0.0
    dii_prev_week: float = 0.0
    flow_prev_week_label: str = ""

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


def _week_pct_change_full(closes: pd.Series, dates=None) -> tuple[float, float, str]:
    """Return (wow_pct, anchor_close, anchor_date_iso) for the current week.

    Anchor = close on the last trading day strictly before this calendar week's
    Monday. On a Friday close this is the prior Friday. Mid-week it's still the
    prior Friday — which is exactly the "last Friday vs latest" semantics the
    dashboard surfaces as WoW.

    `dates` may be supplied when `closes.index` isn't a DatetimeIndex (e.g. the
    YahooFetcher reset_index() flow). If no calendar information is available,
    falls back to a 5-trading-day rolling anchor so callers never raise; in
    that fallback `anchor_date_iso` is "".
    """
    if len(closes) < 2:
        return 0.0, 0.0, ""
    latest = float(closes.iloc[-1])
    if not latest:
        return 0.0, 0.0, ""

    if dates is None:
        if isinstance(closes.index, pd.DatetimeIndex):
            dates_arr = list(closes.index)
        else:
            anchor_close, pct = _rolling_week_anchor(closes)
            return pct, anchor_close, ""
    else:
        try:
            dates_arr = pd.to_datetime(pd.Series(dates).reset_index(drop=True)).tolist()
        except Exception:
            anchor_close, pct = _rolling_week_anchor(closes)
            return pct, anchor_close, ""

    try:
        latest_dt = pd.Timestamp(dates_arr[-1])
        monday = latest_dt.normalize() - pd.Timedelta(days=latest_dt.weekday())
    except Exception:
        anchor_close, pct = _rolling_week_anchor(closes)
        return pct, anchor_close, ""

    anchor_pos = -1
    for i in range(len(dates_arr) - 1, -1, -1):
        if pd.Timestamp(dates_arr[i]) < monday:
            anchor_pos = i
            break

    if anchor_pos < 0:
        anchor_close, pct = _rolling_week_anchor(closes)
        return pct, anchor_close, ""

    anchor = float(closes.iloc[anchor_pos])
    if not anchor:
        return 0.0, anchor, ""
    pct = ((latest - anchor) / anchor) * 100
    anchor_date = pd.Timestamp(dates_arr[anchor_pos]).date().isoformat()
    return pct, anchor, anchor_date


def _week_pct_change(closes: pd.Series, dates=None) -> float:
    """Percent change vs the prior week's last close. See _week_pct_change_full."""
    pct, _, _ = _week_pct_change_full(closes, dates)
    return pct


def _rolling_week_anchor(closes: pd.Series) -> tuple[float, float]:
    """Fallback: (anchor_close, pct) using a 5-trading-day rolling delta."""
    if len(closes) < 6:
        return 0.0, 0.0
    anchor = float(closes.iloc[-6])
    latest = float(closes.iloc[-1])
    pct = ((latest - anchor) / anchor) * 100 if anchor else 0.0
    return anchor, pct


def _market_minute_bucket() -> str:
    """Cache-key bucket that drives cache invalidation across the dashboard.

    - During NSE market hours (9:15–15:30 IST on trading days): rotates every
      minute, so cached snapshots stay <=60s behind the live tape.
    - Outside market hours (after-hours, weekends, holidays): keyed on the
      most recent trading day plus an AM/PM marker. This guarantees a fresh
      fetch the moment a new session starts, so a viewer on Saturday cannot
      see Friday's last intraday tick masquerading as the close — the bucket
      transitions from `live-...` to `closed-2026-04-24-pm` immediately at
      15:30 IST and forces a re-fetch that picks up the settled close.
    """
    now = datetime.now(IST)
    if is_market_open_now(now):
        return f"live-{now:%Y%m%d-%H%M}"
    today = now.date()
    last_session = today if is_trading_day(today) else prev_trading_day(today)
    half = "am" if now.hour < 12 else "pm"
    return f"closed-{last_session.isoformat()}-{half}"


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching live market data...")
def load_market_snapshot(view: str = "daily", _bucket: str = "") -> MarketSnapshot:
    """Master function: fetches all data and returns a MarketSnapshot.

    The `_bucket` argument is a cache-key only — pass `_market_minute_bucket()`
    so the cache rotates every minute during market hours and forces a fresh
    fetch on every session boundary (so Friday's last intraday tick can never
    leak into Saturday's view).
    """
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

            # WoW: change since prior calendar week's last close.
            # Stock dataframes from YahooFetcher have been reset_index()'d, so the
            # date lives in a "date" column rather than the Series index — pass it
            # through explicitly so _week_pct_change can anchor on Monday boundary.
            date_col = "date" if "date" in df.columns else None
            dates_for_close = df.loc[closes.index, date_col] if date_col else None
            wow_pct = _week_pct_change(closes, dates_for_close)

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
    """Fetch global indices, VIX, USD/INR, FII/DII.

    Critical KPIs (Nifty 50, VIX, USD/INR) are fetched individually for
    reliability — yfinance batch downloads silently drop Indian indices.
    Global indices (S&P, NASDAQ, etc.) still use the batch fetcher.
    """
    # --- Critical KPIs: individual fetches ---
    _fetch_nifty50(snapshot, period)
    _fetch_india_vix(snapshot, period)
    _fetch_usdinr(snapshot, period)

    # --- Global indices: fetch individually for reliable price + DoD + WoW + sparkline.
    # Batch yfinance calls silently drop some international indices, and we need
    # rich per-index data (close, DoD %, WoW %, last-N-day sparkline).
    from dashboard.config import GLOBAL_INDEX_DISPLAY
    from config.nifty50_tickers import GLOBAL_INDICES
    for key, display_name in GLOBAL_INDEX_DISPLAY.items():
        ticker = GLOBAL_INDICES.get(key)
        if not ticker:
            continue
        closes = _fetch_single_ticker(ticker, period)
        if len(closes) < 2:
            continue
        latest = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        dod_pct = ((latest - prev) / prev) * 100 if prev else 0.0
        wow_pct = _week_pct_change(closes)
        # Trailing series for sparkline (last ~10 points)
        spark = closes.tail(10).reset_index(drop=True).tolist()
        snapshot.global_indices[display_name] = {
            "close": round(latest, 2),
            "ret_pct": round(dod_pct, 2),
            "wow_pct": round(wow_pct, 2),
            "spark": [float(v) for v in spark],
        }

    # --- FII/DII --- fetch ~30 days so we can compute WoW and MoM aggregates.
    try:
        nf = NSEFetcher(max_retries=2, retry_delay=1.0)
        fii_dii = nf.fetch_recent_fii_dii(lookback_days=30)
        if fii_dii:
            snapshot.fii_dii_series = fii_dii
            latest_fii = fii_dii[-1]
            snapshot.fii_net_buy = latest_fii.get("fii_net_buy", 0.0)
            snapshot.dii_net_buy = latest_fii.get("dii_net_buy", 0.0)
            _populate_flow_week_aggregates(snapshot, fii_dii)
    except Exception as e:
        logger.warning(f"FII/DII fetch failed: {e}")


def _populate_flow_week_aggregates(snapshot: MarketSnapshot, fii_dii: list[dict]):
    """Compute week-to-date and prior-week net flow sums for FII and DII.

    Anchor weeks on calendar Mondays so the WoW comparison on the FII chip
    matches every other WoW number on the dashboard:
      - WTD = sum of net flows from this week's Monday up to the latest print
      - prior week = sum across the prior calendar week (Mon..Sun)
    """
    if not fii_dii:
        return
    rows = []
    for r in fii_dii:
        d = r.get("date")
        try:
            ts = pd.Timestamp(d)
        except Exception:
            continue
        rows.append((
            ts.normalize(),
            float(r.get("fii_net_buy") or 0.0),
            float(r.get("dii_net_buy") or 0.0),
        ))
    if not rows:
        return

    rows.sort(key=lambda x: x[0])
    latest_dt = rows[-1][0]
    monday_this = latest_dt - pd.Timedelta(days=latest_dt.weekday())
    monday_prev = monday_this - pd.Timedelta(days=7)

    fii_wtd = sum(f for d, f, _ in rows if d >= monday_this)
    dii_wtd = sum(di for d, _, di in rows if d >= monday_this)
    fii_prev = sum(f for d, f, _ in rows if monday_prev <= d < monday_this)
    dii_prev = sum(di for d, _, di in rows if monday_prev <= d < monday_this)

    snapshot.fii_wtd = fii_wtd
    snapshot.dii_wtd = dii_wtd
    snapshot.fii_prev_week = fii_prev
    snapshot.dii_prev_week = dii_prev
    snapshot.flow_prev_week_label = (
        f"{monday_prev.strftime('%d-%b')}–{(monday_this - pd.Timedelta(days=1)).strftime('%d-%b')}"
    )


def _fetch_single_ticker(ticker: str, period: str) -> pd.Series:
    """Fetch close series for a single ticker. Returns empty Series on failure."""
    try:
        data = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
        if data.empty:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception as e:
        logger.warning(f"Single-ticker fetch failed for {ticker}: {e}")
        return pd.Series(dtype=float)


def _fetch_nifty50(snapshot: MarketSnapshot, period: str):
    """Fetch Nifty 50 index close and compute day/week changes."""
    closes = _fetch_single_ticker("^NSEI", period)
    if len(closes) < 2:
        return

    latest = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])
    snapshot.nifty50_close = latest
    snapshot.nifty50_change_pct = ((latest - prev) / prev) * 100 if prev else 0

    wow_pct, anchor_close, anchor_date = _week_pct_change_full(closes)
    snapshot.nifty50_wow_pct = wow_pct
    snapshot.nifty50_wow_anchor_close = anchor_close
    snapshot.nifty50_wow_anchor_date = anchor_date


def _fetch_india_vix(snapshot: MarketSnapshot, period: str):
    """Fetch India VIX."""
    closes = _fetch_single_ticker("^INDIAVIX", period)
    if len(closes) < 2:
        return

    snapshot.india_vix = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])
    if prev:
        snapshot.india_vix_change = ((snapshot.india_vix - prev) / prev) * 100

    wow_pct, anchor_close, anchor_date = _week_pct_change_full(closes)
    snapshot.india_vix_wow_pct = wow_pct
    snapshot.india_vix_wow_anchor_close = anchor_close
    snapshot.india_vix_wow_anchor_date = anchor_date

    vix_data = [{"date": str(d.date()) if hasattr(d, "date") else str(d)[:10],
                 "India VIX": float(v)} for d, v in closes.items()]
    snapshot.vix_series = pd.DataFrame(vix_data).set_index("date") if vix_data else pd.DataFrame()


def _fetch_usdinr(snapshot: MarketSnapshot, period: str):
    """Fetch USD/INR exchange rate."""
    closes = _fetch_single_ticker("INR=X", period)
    if len(closes) < 2:
        return

    snapshot.usdinr = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])
    if prev:
        snapshot.usdinr_change = ((snapshot.usdinr - prev) / prev) * 100

    wow_pct, anchor_close, anchor_date = _week_pct_change_full(closes)
    snapshot.usdinr_wow_pct = wow_pct
    snapshot.usdinr_wow_anchor_close = anchor_close
    snapshot.usdinr_wow_anchor_date = anchor_date

    usdinr_data = [{"date": str(d.date()) if hasattr(d, "date") else str(d)[:10],
                    "USD/INR": float(v)} for d, v in closes.items()]
    snapshot.usdinr_series = pd.DataFrame(usdinr_data).set_index("date") if usdinr_data else pd.DataFrame()


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

                wow = _week_pct_change(closes)

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

                wow = _week_pct_change(closes)

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


# ---------------------------------------------------------------------------
# Heatmap universe loader (Nifty 50 + Midcap, sized by market cap)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_market_caps(_day_bucket: str = "") -> dict[str, float]:
    """Fetch market caps for the heatmap universe. Cached for 24h.

    Market cap is structurally stable intraday — refreshing daily is plenty.
    `fast_info` is preferred (orders of magnitude faster than `.info`); we
    fall back to `.info` only when fast_info doesn't surface a cap.
    """
    del _day_bucket
    from concurrent.futures import ThreadPoolExecutor
    from config.midcap_tickers import MIDCAP_STOCKS

    universe: dict[str, str] = {}
    for sym, (yt, _, _) in NIFTY50_STOCKS.items():
        universe[sym] = yt
    for sym, (yt, _, _) in MIDCAP_STOCKS.items():
        universe.setdefault(sym, yt)

    def _one(item):
        sym, tkr = item
        try:
            t = yf.Ticker(tkr)
            mc = None
            try:
                fi = t.fast_info
                mc = getattr(fi, "market_cap", None) or (fi.get("market_cap") if hasattr(fi, "get") else None)
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

    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for sym, mc in ex.map(_one, universe.items()):
            if mc and mc > 0:
                out[sym] = mc
    return out


def _fetch_universe_ohlcv(period: str = "1mo") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for Nifty 50 + Midcap universe in one batch.

    Returns dict NSE-symbol -> DataFrame (with `close`, `volume`, `date`).
    """
    from config.midcap_tickers import MIDCAP_STOCKS

    fetcher = YahooFetcher(max_retries=2, retry_delay=1.0)
    nifty = fetcher.fetch_recent_ohlcv(period=period) or {}

    mc_map = {sym: yt for sym, (yt, _, _) in MIDCAP_STOCKS.items() if sym not in nifty}
    if not mc_map:
        return nifty

    try:
        data = yf.download(
            " ".join(mc_map.values()),
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.warning(f"Midcap batch fetch failed: {e}")
        return nifty

    if data is None or data.empty:
        return nifty

    for sym, yt in mc_map.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                level_0 = data.columns.get_level_values(0).unique().tolist()
                level_1 = data.columns.get_level_values(1).unique().tolist()
                if yt in level_0:
                    sub = data[yt].copy()
                elif yt in level_1:
                    sub = data.xs(yt, level=1, axis=1).copy()
                else:
                    continue
            else:
                sub = data.copy()
            sub.columns = [str(c).lower() for c in sub.columns]
            if "close" not in sub.columns:
                continue
            sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
            sub.columns = [str(c).lower() for c in sub.columns]
            nifty[sym] = sub
        except Exception:
            continue

    return nifty


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Building market heatmap...")
def load_heatmap_data(view: str = "daily", _bucket: str = "") -> pd.DataFrame:
    """Return a DataFrame for the heatmap: Nifty 50 + Midcap, sized by mcap.

    Columns: symbol, company, sector, close, dod_pct, wow_pct, market_cap,
    volume, size (market_cap with liquidity fallback so a stock without a
    fetched mcap still renders proportionally).
    """
    del _bucket
    from config.midcap_tickers import MIDCAP_STOCKS

    name_sector: dict[str, tuple[str, str]] = {}
    for sym, (_, name, sec) in NIFTY50_STOCKS.items():
        name_sector[sym] = (name, sec)
    for sym, (_, name, sec) in MIDCAP_STOCKS.items():
        name_sector.setdefault(sym, (name, sec))

    today = date.today()
    day_bucket = (today if is_trading_day(today) else prev_trading_day(today)).isoformat()
    mcaps = _fetch_market_caps(_day_bucket=day_bucket)

    universe_data = _fetch_universe_ohlcv(period="1mo")
    if not universe_data:
        return pd.DataFrame()

    rows = []
    for sym, df in universe_data.items():
        if df is None or df.empty:
            continue
        if "close" not in df.columns:
            continue
        closes = df["close"].dropna()
        if len(closes) < 2:
            continue

        latest = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        dod = ((latest - prev) / prev) * 100 if prev else 0.0

        date_col = "date" if "date" in df.columns else None
        dates_for_close = df.loc[closes.index, date_col] if date_col else None
        wow = _week_pct_change(closes, dates_for_close)

        vol_series = df["volume"].dropna() if "volume" in df.columns else pd.Series(dtype=float)
        avg_vol = float(vol_series.tail(20).mean()) if not vol_series.empty else 0.0
        liquidity = latest * avg_vol  # ₹ traded value ~ liquidity proxy

        name, sector = name_sector.get(sym, (sym, "Other"))
        mcap = float(mcaps.get(sym, 0.0))
        size = mcap if mcap > 0 else max(liquidity, 1.0)

        rows.append({
            "symbol": sym,
            "company": name,
            "sector": sector or "Other",
            "close": round(latest, 2),
            "dod_pct": round(dod, 2),
            "wow_pct": round(wow, 2),
            "market_cap": mcap,
            "volume": int(vol_series.iloc[-1]) if not vol_series.empty else 0,
            "size": float(size),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
