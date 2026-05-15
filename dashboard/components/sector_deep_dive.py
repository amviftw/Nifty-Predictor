"""Sector deep dive: per-sector EMA/SMA distance trend + relative-to-sector constituents."""

import pandas as pd
import yfinance as yf
import streamlit as st
from loguru import logger

from dashboard.data_loader import MarketSnapshot, _market_minute_bucket
from dashboard.config import (
    SECTORAL_INDICES,
    SECTOR_INDEX_TO_SECTOR,
    SECTOR_SUPPLY_CHAIN,
    CACHE_TTL_SECONDS,
)
from dashboard.disk_cache import disk_cached


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
@disk_cached(name="sector_history_2y", ttl_hours=6)
def _fetch_sector_history(_bucket: str = "") -> dict:
    """Fetch ~2y of daily closes for every sectoral index in one batch.

    `_bucket` is part of the cache key only; passing `_market_minute_bucket()`
    keeps history fresh across session boundaries.
    """
    del _bucket
    tickers = list(SECTORAL_INDICES.values())
    try:
        data = yf.download(
            " ".join(tickers),
            period="2y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"Sector history fetch failed: {e}")
        return {}

    if data.empty:
        return {}

    histories: dict[str, pd.Series] = {}
    for name, ticker in SECTORAL_INDICES.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                level_0 = data.columns.get_level_values(0).unique().tolist()
                level_1 = data.columns.get_level_values(1).unique().tolist()
                if ticker in level_0:
                    close = data[ticker]["Close"]
                elif ticker in level_1:
                    sub = data.xs(ticker, level=1, axis=1)
                    close = sub["Close"] if "Close" in sub.columns else sub.iloc[:, 0]
                else:
                    continue
            else:
                close = data["Close"]

            close = close.dropna()
            if len(close) >= 30:
                histories[name] = close
        except Exception as e:
            logger.debug(f"sector history extract failed for {name}: {e}")
            continue
    return histories


def _ema_sma_deltas(closes: pd.Series) -> pd.DataFrame:
    """% distance of close from EMA 21 / 51 / 100 and SMA 200."""
    df = pd.DataFrame(index=closes.index)
    e21 = closes.ewm(span=21, adjust=False).mean()
    e51 = closes.ewm(span=51, adjust=False).mean()
    e100 = closes.ewm(span=100, adjust=False).mean()
    s200 = closes.rolling(200).mean()
    df["EMA 21"] = (closes - e21) / e21 * 100
    df["EMA 51"] = (closes - e51) / e51 * 100
    df["EMA 100"] = (closes - e100) / e100 * 100
    df["SMA 200"] = (closes - s200) / s200 * 100
    return df


def _render_sector_ema_chart(closes: pd.Series, view: str):
    """Distance-from-MA line chart for a single sector index."""
    deltas = _ema_sma_deltas(closes).dropna()
    if deltas.empty:
        st.caption("Not enough history to compute moving averages")
        return

    if view == "weekly":
        deltas = deltas.resample("W-FRI").last().dropna()
        window = 52
    else:
        window = 120

    chart_data = deltas.tail(window)
    if chart_data.empty:
        st.caption("Not enough history yet")
        return

    st.line_chart(chart_data, height=240, use_container_width=True)

    latest = deltas.iloc[-1]
    cols = st.columns(4)
    for col, label in zip(cols, ["EMA 21", "EMA 51", "EMA 100", "SMA 200"]):
        val = float(latest[label])
        with col:
            color = "green" if val > 0 else "red"
            st.markdown(f"**{label}**: :{color}[{val:+.1f}%]")


def render_sector_deep_dive(snapshot: MarketSnapshot):
    """Render expandable sector-by-sector breakdown."""
    st.subheader("Sector Deep Dive")

    if snapshot.sectoral_data.empty or snapshot.stock_changes.empty:
        st.info("Sector data unavailable")
        return

    sector_history = _fetch_sector_history(_bucket=_market_minute_bucket())

    change_col = "dod_pct" if snapshot.view == "daily" else "wow_pct"
    sort_col = "DoD %" if snapshot.view == "daily" else "WoW %"
    change_label = "DoD %" if snapshot.view == "daily" else "WoW %"

    sorted_sectors = snapshot.sectoral_data.sort_values(sort_col, ascending=False)

    for _, row in sorted_sectors.iterrows():
        index_name = row["Index"]
        sector_name = SECTOR_INDEX_TO_SECTOR.get(index_name, index_name)
        change_val = float(row[sort_col])

        indicator = "🟢" if change_val > 0 else "🔴" if change_val < 0 else "⚪"

        with st.expander(
            f"{indicator}  **{index_name}** — {change_val:+.1f}%  |  Close: {row['Close']:,.0f}"
        ):
            chart_col, info_col = st.columns([2.2, 1])

            with chart_col:
                st.markdown("**Distance from Moving Averages (%)**")
                closes = sector_history.get(index_name)
                if closes is not None and not closes.empty:
                    _render_sector_ema_chart(closes, snapshot.view)
                else:
                    st.caption("History unavailable")

            with info_col:
                st.markdown("**Index Performance**")
                st.metric("Day", f"{row['DoD %']:+.1f}%")
                st.metric("Week", f"{row['WoW %']:+.1f}%")
                st.metric("Month", f"{row['1M %']:+.1f}%")

                sc_info = SECTOR_SUPPLY_CHAIN.get(sector_name)
                if sc_info:
                    st.markdown("**Key Drivers**")
                    st.caption(f"Linked to: {', '.join(sc_info['factors'])}")
                    st.caption(sc_info["note"])

            sector_stocks = snapshot.stock_changes[
                snapshot.stock_changes["sector"] == sector_name
            ].copy()

            if not sector_stocks.empty:
                # vs Sector = stock's move minus the sector index's move on the
                # same horizon. Positive => stock outperforming the sector.
                sector_stocks["rel_pct"] = (sector_stocks[change_col] - change_val).round(2)
                sector_stocks = sector_stocks.sort_values("rel_pct", ascending=False)

                st.markdown(f"**Constituents — relative move vs {index_name}**")
                display_df = sector_stocks[
                    ["symbol", "company", change_col, "rel_pct", "close", "volume"]
                ].copy()
                display_df.columns = [
                    "Symbol",
                    "Company",
                    change_label,
                    "vs Sector",
                    "Close",
                    "Volume",
                ]
                st.dataframe(
                    display_df,
                    column_config={
                        change_label: st.column_config.NumberColumn(format="%+.1f%%"),
                        "vs Sector": st.column_config.NumberColumn(
                            help=(
                                f"Stock {change_label} minus {index_name} {change_label}. "
                                "Positive = outperforming the sector."
                            ),
                            format="%+.1f%%",
                        ),
                        "Close": st.column_config.NumberColumn(format="%.2f"),
                        "Volume": st.column_config.NumberColumn(format="%d"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption(f"No Nifty 50 stocks in '{sector_name}' sector")


def render_sector_rotation(snapshot: MarketSnapshot):
    """Render sector rotation table showing leadership over different timeframes."""
    st.subheader("Sector Rotation")

    if snapshot.sectoral_data.empty:
        st.info("Sector data unavailable")
        return

    df = snapshot.sectoral_data.copy()

    df["Day Rank"] = df["DoD %"].rank(ascending=False).astype(int)
    df["Week Rank"] = df["WoW %"].rank(ascending=False).astype(int)
    df["Month Rank"] = df["1M %"].rank(ascending=False).astype(int)

    display_df = df[["Index", "DoD %", "Day Rank", "WoW %", "Week Rank", "1M %", "Month Rank"]]
    display_df = display_df.sort_values("Week Rank")

    st.dataframe(
        display_df,
        column_config={
            "DoD %": st.column_config.NumberColumn("Day %", format="%+.1f%%"),
            "WoW %": st.column_config.NumberColumn("Week %", format="%+.1f%%"),
            "1M %": st.column_config.NumberColumn("Month %", format="%+.1f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )

    if len(df) >= 3:
        day_leaders = df.nsmallest(3, "Day Rank")["Index"].tolist()
        week_leaders = df.nsmallest(3, "Week Rank")["Index"].tolist()
        month_leaders = df.nsmallest(3, "Month Rank")["Index"].tolist()

        day_set = set(day_leaders)
        week_set = set(week_leaders)
        month_set = set(month_leaders)

        if day_set != week_set:
            new_leaders = day_set - week_set
            if new_leaders:
                st.caption(f"Emerging leadership: {', '.join(new_leaders)} (leading today, not in weekly top 3)")

        fading = month_set - day_set
        if fading:
            st.caption(f"Fading momentum: {', '.join(fading)} (monthly leader but lagging today)")
