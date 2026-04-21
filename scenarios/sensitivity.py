"""
Sensitivity Matrix — how much each Nifty 50 stock has historically moved
for a 1% (or 1-unit) move in each driver.

We compute two complementary statistics over a rolling window of daily
close-to-close returns:

    beta         : OLS slope of stock_return on driver_return.
                   Interpretation: stock moves `beta` units for a 1-unit move in driver.
    up_hit_rate  : P( stock_return > 0 | driver_return > +sigma ).
                   Interpretation: when the driver has a strong up-move, how often
                   does the stock close up? Base-rate for the Varsity explainers.
    down_hit_rate: P( stock_return < 0 | driver_return < -sigma ).

All computations use close-to-close daily log returns. Overnight alignment
(e.g. shifting S&P by +1 day vs Nifty) is applied for drivers whose trading
session ends before NSE opens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _CACHE = st.cache_data(ttl=3600, show_spinner=False)
except ImportError:  # pragma: no cover - streamlit always present in app context
    def _CACHE(fn):
        return fn

import yfinance as yf

from config.nifty50_tickers import NIFTY50_STOCKS, get_sector
from scenarios.drivers import DRIVERS, price_based_drivers


# Drivers whose cash session ends AFTER NSE close (lag them by +1 day so the
# driver's close "leads" the next Indian session). S&P and Nasdaq are the
# canonical overnight leads.
_OVERNIGHT_DRIVERS = {"sp500"}


@dataclass
class SensitivityMatrix:
    """Container for the full Driver x Stock sensitivity table.

    betas         : DataFrame indexed by driver_id, columns = symbol. Beta of
                    stock return vs driver return (both in fractional units).
    up_hit_rates  : Same shape. P(stock up | driver up-shock).
    down_hit_rates: Same shape. P(stock down | driver down-shock).
    n_obs         : Series indexed by driver_id. Number of joint observations.
    lookback_days : How far back the window was.
    """
    betas: pd.DataFrame
    up_hit_rates: pd.DataFrame
    down_hit_rates: pd.DataFrame
    n_obs: pd.Series
    lookback_days: int

    def sectors(self) -> dict[str, str]:
        """symbol -> sector map, aligned to columns."""
        return {sym: get_sector(sym) for sym in self.betas.columns}

    def top_for_driver(
        self,
        driver_id: str,
        side: str = "positive",
        n: int = 10,
    ) -> pd.DataFrame:
        """Return the `n` stocks most positively (or negatively) beta to a driver.

        Returns a DataFrame with columns: symbol, sector, beta, up_hit_rate,
        down_hit_rate — sorted by beta (descending if positive, ascending if negative).
        """
        if driver_id not in self.betas.index:
            return pd.DataFrame()

        row = self.betas.loc[driver_id]
        up_hit = self.up_hit_rates.loc[driver_id]
        down_hit = self.down_hit_rates.loc[driver_id]

        ordered = row.sort_values(ascending=(side != "positive"))
        picked = ordered.head(n).index

        return pd.DataFrame({
            "symbol": picked,
            "sector": [get_sector(s) for s in picked],
            "beta": row.loc[picked].values,
            "up_hit_rate": up_hit.loc[picked].values,
            "down_hit_rate": down_hit.loc[picked].values,
        }).reset_index(drop=True)

    def sector_beta(self, driver_id: str) -> pd.Series:
        """Average beta grouped by sector for a given driver."""
        if driver_id not in self.betas.index:
            return pd.Series(dtype=float)
        betas = self.betas.loc[driver_id].rename("beta").reset_index()
        betas.columns = ["symbol", "beta"]
        betas["sector"] = betas["symbol"].map(get_sector)
        return betas.groupby("sector")["beta"].mean().sort_values(ascending=False)


# -----------------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------------

@_CACHE
def _fetch_history(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    """Batch-fetch close prices from yfinance. Returns wide DF indexed by date.

    Note: `tickers` is a tuple so Streamlit can hash it.
    """
    period = f"{max(lookback_days + 60, 400)}d"
    raw = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex (ticker, field) when multi-download. Flatten to closes.
    closes: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                sub = raw[t]
                if "Close" in sub.columns:
                    closes[t] = sub["Close"].dropna()
    else:
        # Single ticker case — not expected but handle it.
        if "Close" in raw.columns and len(tickers) == 1:
            closes[tickers[0]] = raw["Close"].dropna()

    if not closes:
        return pd.DataFrame()

    df = pd.concat(closes, axis=1)
    df.index = pd.to_datetime(df.index)
    return df


def _log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, NaNs preserved for missing days."""
    return np.log(close / close.shift(1))


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def build_sensitivity_matrix(lookback_days: int = 500) -> SensitivityMatrix:
    """Compute the driver-stock sensitivity matrix from historical data.

    Args:
        lookback_days: How many calendar days of history to request from yfinance.

    Returns:
        SensitivityMatrix. If data fetching fails, matrices are returned empty.
    """
    stock_tickers = [info[0] for info in NIFTY50_STOCKS.values()]
    stock_symbols = list(NIFTY50_STOCKS.keys())

    driver_ids_list = price_based_drivers()
    driver_tickers = [DRIVERS[d].ticker for d in driver_ids_list]

    all_tickers = tuple(stock_tickers + driver_tickers)
    closes = _fetch_history(all_tickers, lookback_days)

    if closes.empty:
        return _empty_matrix(driver_ids_list, stock_symbols, lookback_days)

    # Map yahoo ticker back to NSE symbol for readability.
    yahoo_to_symbol = {info[0]: sym for sym, info in NIFTY50_STOCKS.items()}
    stock_close = closes[[t for t in stock_tickers if t in closes.columns]].rename(
        columns=yahoo_to_symbol
    )
    driver_close = pd.DataFrame({
        d: closes[DRIVERS[d].ticker]
        for d in driver_ids_list
        if DRIVERS[d].ticker in closes.columns
    })

    stock_rets = _log_returns(stock_close)
    driver_rets = _log_returns(driver_close)

    # Overnight alignment: shift driver forward by 1 day so driver close on
    # day T "predicts" Indian session T+1.
    for d in _OVERNIGHT_DRIVERS:
        if d in driver_rets.columns:
            driver_rets[d] = driver_rets[d].shift(1)

    # Keep only the last `lookback_days` rows after alignment.
    stock_rets = stock_rets.tail(lookback_days)
    driver_rets = driver_rets.tail(lookback_days)

    betas = pd.DataFrame(index=driver_ids_list, columns=stock_symbols, dtype=float)
    up_hits = pd.DataFrame(index=driver_ids_list, columns=stock_symbols, dtype=float)
    down_hits = pd.DataFrame(index=driver_ids_list, columns=stock_symbols, dtype=float)
    n_obs = pd.Series(index=driver_ids_list, dtype=float)

    for d in driver_ids_list:
        if d not in driver_rets.columns:
            continue
        dr = driver_rets[d]
        sigma = dr.std(skipna=True)
        if not np.isfinite(sigma) or sigma == 0:
            continue

        up_mask = dr > +sigma
        down_mask = dr < -sigma

        for sym in stock_symbols:
            if sym not in stock_rets.columns:
                continue
            sr = stock_rets[sym]
            joined = pd.concat([sr, dr], axis=1, join="inner").dropna()
            if len(joined) < 30:
                continue
            x = joined.iloc[:, 1].values
            y = joined.iloc[:, 0].values
            # OLS slope (no intercept forcing). beta = cov / var.
            var = np.var(x)
            if var == 0:
                continue
            beta = float(np.cov(x, y, bias=False)[0, 1] / var)
            betas.loc[d, sym] = beta

            # Conditional hit-rates.
            if up_mask.any():
                up_hits.loc[d, sym] = float(((sr > 0) & up_mask).sum() / max(up_mask.sum(), 1))
            if down_mask.any():
                down_hits.loc[d, sym] = float(((sr < 0) & down_mask).sum() / max(down_mask.sum(), 1))

        n_obs.loc[d] = int(dr.dropna().shape[0])

    return SensitivityMatrix(
        betas=betas.astype(float),
        up_hit_rates=up_hits.astype(float),
        down_hit_rates=down_hits.astype(float),
        n_obs=n_obs.astype(float),
        lookback_days=lookback_days,
    )


def _empty_matrix(driver_ids: list[str], symbols: list[str], lookback: int) -> SensitivityMatrix:
    empty = pd.DataFrame(index=driver_ids, columns=symbols, dtype=float)
    return SensitivityMatrix(
        betas=empty.copy(),
        up_hit_rates=empty.copy(),
        down_hit_rates=empty.copy(),
        n_obs=pd.Series(index=driver_ids, dtype=float),
        lookback_days=lookback,
    )
