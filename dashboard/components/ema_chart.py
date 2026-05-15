"""Nifty 50 price chart with EMA / SMA overlays and a timeframe toggle.

Renders the spot Nifty 50 close as a primary line with EMA 21 / 50 / 100 and
SMA 200 plotted on the same axis. The timeframe toggle (1M / 3M / 6M / YTD /
1Y / 3Y / 5Y / 10Y / All) re-slices the same underlying series — moving
averages are always computed on the full history (so EMA-200 is meaningful
even on a 1M view) and trimmed to the selected window for display.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from loguru import logger

from dashboard.config import CACHE_TTL_SECONDS
from dashboard.data_loader import _market_minute_bucket
from dashboard.disk_cache import disk_cached


# Colour palette — matches the Groww-style dashboard CSS in app.py.
PRICE = "#00d09c"
EMA21 = "#f5a623"
EMA50 = "#5b8def"
EMA100 = "#b97aff"
SMA200 = "#eb5757"
TEXT_PRIMARY = "#e8ecf1"
TEXT_MUTED = "#7a8294"
BORDER = "#232834"
PANEL = "#0f131c"


# Display window → trading-days approximation. None = no slicing (All).
_TIMEFRAMES: dict[str, int | None] = {
    "1M": 22,
    "3M": 66,
    "6M": 132,
    "YTD": -1,        # special-case: from Jan 1 of current year
    "1Y": 252,
    "3Y": 756,
    "5Y": 1260,
    "7Y": 1764,
    "10Y": 2520,
    "All": None,
}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
@disk_cached(name="nifty_history", ttl_hours=6)
def _fetch_nifty_history(period: str = "max", _bucket: str = "") -> pd.DataFrame:
    """Fetch Nifty 50 daily close history. Cached on the minute bucket.

    We fetch a long history (`max` by default) so a 200-day SMA is well
    defined even on the shortest timeframes. Slicing happens on display.

    The disk-cache layer (6h TTL) survives Streamlit server restarts, so a
    cold-loaded dashboard no longer pays the ~3s Yahoo round-trip for two
    decades of Nifty closes — only the deltas matter intraday and they're
    already plumbed through the live-quote overlay elsewhere.
    """
    del _bucket
    try:
        data = yf.download("^NSEI", period=period, interval="1d", auto_adjust=True,
                           progress=False, threads=False)
        if data.empty:
            return pd.DataFrame()
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna().to_frame(name="Close")
    except Exception as e:
        logger.error(f"Nifty history fetch failed: {e}")
        return pd.DataFrame()


def _compute_overlays(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    out = df.copy()
    out["EMA 21"] = close.ewm(span=21, adjust=False).mean()
    out["EMA 50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA 100"] = close.ewm(span=100, adjust=False).mean()
    out["SMA 200"] = close.rolling(200).mean()
    return out


def _slice_for_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    n = _TIMEFRAMES[window]
    if n is None:
        return df
    if window == "YTD":
        year_start = pd.Timestamp(date.today().year, 1, 1)
        return df[df.index >= year_start]
    return df.tail(n)


def _build_figure(df: pd.DataFrame, window: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Nifty 50",
        mode="lines",
        line=dict(color=PRICE, width=2.4),
        hovertemplate="%{x|%d %b %Y}<br>Nifty: <b>%{y:,.0f}</b><extra></extra>",
    ))

    overlay_specs = [
        ("EMA 21", EMA21, 1.4),
        ("EMA 50", EMA50, 1.4),
        ("EMA 100", EMA100, 1.4),
        ("SMA 200", SMA200, 1.6),
    ]
    for col, color, width in overlay_specs:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                mode="lines",
                line=dict(color=color, width=width, dash="solid"),
                hovertemplate=col + ": %{y:,.0f}<extra></extra>",
            ))

    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=420,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, sans-serif", color="#c9cfd9"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0b0e14",
            bordercolor="#2f3645",
            font=dict(family="Inter, sans-serif", size=12, color=TEXT_PRIMARY),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        xaxis=dict(
            showgrid=False, showline=False,
            tickfont=dict(size=10, color=TEXT_MUTED),
            type="date",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=10, color=TEXT_MUTED),
            showline=False, zeroline=False,
            tickformat=",.0f",
        ),
    )
    return fig


def _summary_strip(latest_row: pd.Series, prev_close: float | None) -> str:
    close = float(latest_row["Close"])
    parts = []

    chg = ""
    if prev_close:
        delta = close - prev_close
        pct = (delta / prev_close) * 100
        color = "#00d09c" if delta >= 0 else "#eb5757"
        sign = "+" if delta >= 0 else ""
        chg = (
            f'<span style="color:{color};font-weight:600;margin-left:10px;font-size:0.95rem;">'
            f'{sign}{delta:,.1f} ({sign}{pct:.2f}%)</span>'
        )

    parts.append(
        f'<div style="font-size:1.6rem;font-weight:700;color:{TEXT_PRIMARY};letter-spacing:-0.02em;">'
        f'{close:,.0f}{chg}</div>'
    )

    chips: list[str] = []
    chip_specs = [
        ("EMA 21", EMA21),
        ("EMA 50", EMA50),
        ("EMA 100", EMA100),
        ("SMA 200", SMA200),
    ]
    for col, color in chip_specs:
        val = latest_row.get(col)
        if val is None or pd.isna(val):
            continue
        gap_pct = ((close - float(val)) / float(val)) * 100 if val else 0.0
        gap_color = "#00d09c" if gap_pct > 0 else "#eb5757"
        sign = "+" if gap_pct > 0 else ""
        chips.append(
            f'<div style="display:inline-flex;flex-direction:column;gap:2px;'
            f'padding:6px 12px;border:1px solid {BORDER};border-radius:8px;'
            f'background:#151922;min-width:108px;">'
            f'<span style="font-size:0.62rem;color:{TEXT_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.06em;"><span style="display:inline-block;width:8px;height:8px;'
            f'background:{color};border-radius:2px;margin-right:6px;vertical-align:middle;">'
            f'</span>{col}</span>'
            f'<span style="font-size:0.92rem;font-weight:600;color:{TEXT_PRIMARY};">{float(val):,.0f}</span>'
            f'<span style="font-size:0.7rem;color:{gap_color};font-weight:500;">'
            f'{sign}{gap_pct:.2f}% vs spot</span>'
            f'</div>'
        )

    parts.append(
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;">{"".join(chips)}</div>'
    )

    return "".join(parts)


def render_ema_chart(view: str = "daily"):
    """Render Nifty 50 price line with EMA / SMA overlays and timeframe toggle."""
    del view  # the timeframe is selected inside this widget itself

    st.markdown("#### Nifty 50 — Price & Moving Averages")

    df_full = _fetch_nifty_history("max", _bucket=_market_minute_bucket())
    if df_full.empty or len(df_full) < 30:
        st.info("Nifty 50 history unavailable")
        return

    df_with_ma = _compute_overlays(df_full)

    timeframe_options = list(_TIMEFRAMES.keys())
    default_idx = timeframe_options.index("1Y")
    window = st.radio(
        "Timeframe",
        timeframe_options,
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="nifty_chart_timeframe",
    )

    sliced = _slice_for_window(df_with_ma, window)
    if sliced.empty:
        st.caption("Not enough history for this window.")
        return

    latest_row = df_with_ma.iloc[-1]
    prev_close = float(df_with_ma["Close"].iloc[-2]) if len(df_with_ma) >= 2 else None
    st.markdown(_summary_strip(latest_row, prev_close), unsafe_allow_html=True)

    fig = _build_figure(sliced, window)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Brief regime read so the chart isn't just shapes.
    msg = _regime_caption(latest_row)
    if msg:
        st.caption(msg)


def _regime_caption(latest: pd.Series) -> str:
    close = float(latest["Close"])
    e21 = latest.get("EMA 21")
    e50 = latest.get("EMA 50")
    e100 = latest.get("EMA 100")
    s200 = latest.get("SMA 200")

    if any(pd.isna(v) for v in (e21, e50, e100, s200)):
        return ""

    above_all = close > e21 > e50 > e100 > s200
    below_all = close < e21 < e50 < e100 < s200
    if above_all:
        return "Stacked bullishly — price above EMA 21 > 50 > 100 > SMA 200. Trend intact."
    if below_all:
        return "Stacked bearishly — price below all key averages. Wait for reversal."
    if close > s200 and close < e21:
        return "Long-term uptrend (above SMA 200) with a short-term pullback below EMA 21 — watch for support."
    if close < s200 and close > e21:
        return "Short-term recovery above EMA 21 in a longer-term downtrend — needs follow-through above EMA 50 / 100."
    return ""
