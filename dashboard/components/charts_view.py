"""
TradingView-style candlestick charts view.

A standalone mode that lets users pick any tracked instrument, choose a
timeframe / interval, toggle common indicators, and see live-refreshing
candles with volume, RSI and MACD subplots.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.nifty50_tickers import NIFTY50_STOCKS, GLOBAL_INDICES, SECTOR_INDICES
from dashboard.chart_data import (
    TIMEFRAMES,
    INTRADAY_INTERVALS,
    build_bundle,
    ChartBundle,
)

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False


# Explicit interval override list — superset of per-timeframe defaults.
ALL_INTERVALS = ["1m", "5m", "15m", "30m", "60m", "1d", "1wk"]

# Color palette tuned for the dark theme.
BULL = "#26a69a"
BEAR = "#ef5350"
ACCENT_1 = "#42a5f5"
ACCENT_2 = "#ab47bc"
ACCENT_3 = "#ffa726"
GRID = "rgba(255, 255, 255, 0.06)"
TEXT = "#d7d7d7"


# ---------------------------------------------------------------------------
# Instrument universe
# ---------------------------------------------------------------------------

def _build_universe() -> dict[str, str]:
    """Return {display_label: yahoo_ticker} for the picker."""
    universe: dict[str, str] = {}
    # Indices first
    for name, tkr in GLOBAL_INDICES.items():
        universe[f"Index · {name}"] = tkr
    for name, tkr in SECTOR_INDICES.items():
        universe[f"Sector · {name}"] = tkr
    # Stocks
    for sym, (tkr, company, sector) in sorted(NIFTY50_STOCKS.items()):
        universe[f"{sym} · {company}"] = tkr
    return universe


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_figure(bundle: ChartBundle, indicators: dict[str, bool]) -> go.Figure:
    df = bundle.df

    show_rsi = indicators.get("rsi", False)
    show_macd = indicators.get("macd", False)

    rows = 2  # price + volume baseline
    row_heights = [0.65, 0.15]
    specs_extra = []
    if show_rsi:
        rows += 1
        row_heights.append(0.10)
        specs_extra.append("rsi")
    if show_macd:
        rows += 1
        row_heights.append(0.10)
        specs_extra.append("macd")
    # Normalize row heights to sum to 1
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    # --- Price: candles ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color=BULL, decreasing_line_color=BEAR,
            increasing_fillcolor=BULL, decreasing_fillcolor=BEAR,
            name="Price", showlegend=False,
        ),
        row=1, col=1,
    )

    # --- Overlays ---
    if indicators.get("ema20") and "EMA_20" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_20"], mode="lines",
            line=dict(color=ACCENT_1, width=1.2), name="EMA 20",
        ), row=1, col=1)

    if indicators.get("ema50") and "EMA_50" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_50"], mode="lines",
            line=dict(color=ACCENT_2, width=1.2), name="EMA 50",
        ), row=1, col=1)

    if indicators.get("sma200") and "SMA_200" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_200"], mode="lines",
            line=dict(color=ACCENT_3, width=1.4), name="SMA 200",
        ), row=1, col=1)

    if indicators.get("bbands") and "BB_Upper" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], mode="lines",
            line=dict(color="rgba(120,144,156,0.5)", width=1, dash="dot"),
            name="BB Upper",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], mode="lines",
            line=dict(color="rgba(120,144,156,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(120,144,156,0.08)",
            name="BB Lower",
        ), row=1, col=1)

    if indicators.get("vwap") and "VWAP" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], mode="lines",
            line=dict(color="#ffee58", width=1.4, dash="dash"),
            name="VWAP",
        ), row=1, col=1)

    # --- Volume ---
    vol_colors = [
        BULL if c >= o else BEAR
        for o, c in zip(df["Open"].to_numpy(), df["Close"].to_numpy())
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=vol_colors, marker_line_width=0,
            name="Volume", showlegend=False, opacity=0.7,
        ),
        row=2, col=1,
    )

    next_row = 3
    # --- RSI ---
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], mode="lines",
            line=dict(color=ACCENT_1, width=1.3), name="RSI 14",
        ), row=next_row, col=1)
        for level, dash in [(70, "dot"), (30, "dot")]:
            fig.add_hline(
                y=level, line=dict(color="rgba(255,255,255,0.25)", width=1, dash=dash),
                row=next_row, col=1,
            )
        fig.update_yaxes(range=[0, 100], row=next_row, col=1, title_text="RSI")
        next_row += 1

    # --- MACD ---
    if show_macd:
        hist_colors = [BULL if v >= 0 else BEAR for v in df["Hist"].fillna(0).to_numpy()]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Hist"], marker_color=hist_colors,
            marker_line_width=0, name="Hist", opacity=0.55,
        ), row=next_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], mode="lines",
            line=dict(color=ACCENT_1, width=1.3), name="MACD",
        ), row=next_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Signal"], mode="lines",
            line=dict(color=ACCENT_3, width=1.3), name="Signal",
        ), row=next_row, col=1)
        fig.update_yaxes(title_text="MACD", row=next_row, col=1)

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        height=760,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color=TEXT, size=12),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
        dragmode="pan",
        bargap=0,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text=f"Price ({bundle.currency})", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    # Hide non-trading gaps for intraday
    if df.index.inferred_freq is None and len(df) > 2:
        delta = (df.index[-1] - df.index[-2]).total_seconds()
        if delta < 24 * 3600:  # intraday
            fig.update_xaxes(rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[15.5, 9.25], pattern="hour"),
            ])

    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_charts_view():
    """Top-level render for the Charts mode."""
    st.title("Charts")
    st.caption("TradingView-style candles with live refresh and common indicators.")

    universe = _build_universe()
    labels = list(universe.keys())
    default_idx = labels.index("Index · NIFTY50") if "Index · NIFTY50" in labels else 0

    # --- Controls row 1: symbol / timeframe / interval / refresh ---
    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.6])
    with c1:
        label = st.selectbox("Instrument", labels, index=default_idx, key="chart_symbol")
    with c2:
        tf_key = st.selectbox(
            "Timeframe", list(TIMEFRAMES.keys()), index=2, key="chart_tf",
            help="Controls the lookback window.",
        )
    with c3:
        default_interval = TIMEFRAMES[tf_key][1]
        interval_options = ["(auto)"] + ALL_INTERVALS
        interval_choice = st.selectbox(
            "Interval", interval_options,
            index=0,
            help=f"Auto picks {default_interval} for this timeframe.",
            key="chart_interval",
        )
    with c4:
        auto_refresh = st.toggle(
            "Live refresh (60s)",
            value=False,
            help="Re-fetches quotes every 60 seconds while the Charts tab is open.",
            key="chart_autorefresh",
        )

    override = None if interval_choice == "(auto)" else interval_choice
    ticker = universe[label]

    # --- Controls row 2: indicator toggles ---
    with st.expander("Indicators", expanded=True):
        i1, i2, i3, i4, i5, i6, i7 = st.columns(7)
        indicators = {
            "ema20": i1.checkbox("EMA 20", value=True),
            "ema50": i2.checkbox("EMA 50", value=True),
            "sma200": i3.checkbox("SMA 200", value=False),
            "bbands": i4.checkbox("Bollinger", value=False),
            "vwap": i5.checkbox("VWAP", value=True),
            "rsi": i6.checkbox("RSI", value=True),
            "macd": i7.checkbox("MACD", value=False),
        }

    # --- Fetch + render ---
    with st.spinner(f"Loading {label}..."):
        bundle = build_bundle(ticker, tf_key, override_interval=override)

    if bundle.df.empty:
        st.error(f"No data returned for {ticker}. Try a different timeframe or interval.")
        return

    _render_stats_strip(bundle, label)

    # Warn for VWAP on daily bars
    effective_interval = override or TIMEFRAMES[tf_key][1]
    if indicators["vwap"] and effective_interval not in INTRADAY_INTERVALS:
        st.caption("VWAP is only meaningful on intraday intervals; hidden on this timeframe.")
        indicators["vwap"] = False

    fig = _build_figure(bundle, indicators)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

    # --- Live refresh ---
    if auto_refresh:
        if _HAS_AUTOREFRESH:
            st_autorefresh(interval=60_000, key="chart_autorefresh_tick")
        else:
            st.info(
                "`streamlit-autorefresh` is not installed. "
                "Run `pip install streamlit-autorefresh` to enable live tick updates."
            )


def _render_stats_strip(bundle: ChartBundle, label: str):
    """Top-of-chart headline stats."""
    change_color = BULL if bundle.day_change_pct >= 0 else BEAR
    arrow = "▲" if bundle.day_change_pct >= 0 else "▼"

    cols = st.columns([2.2, 1, 1, 1, 1, 1])
    with cols[0]:
        st.markdown(
            f"### {label.split(' · ', 1)[-1] if ' · ' in label else label}"
            f"  <span style='color:{change_color};font-size:0.7em'>"
            f"{arrow} {bundle.day_change_pct:+.2f}%</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"`{bundle.symbol}` · {bundle.currency}")
    cols[1].metric("Last", f"{bundle.last_price:,.2f}")
    cols[2].metric("Prev Close", f"{bundle.prev_close:,.2f}")
    cols[3].metric("Day High", f"{bundle.day_high:,.2f}")
    cols[4].metric("Day Low", f"{bundle.day_low:,.2f}")
    range_52w = (
        f"{bundle.week52_low:,.0f} – {bundle.week52_high:,.0f}"
        if bundle.week52_high == bundle.week52_high  # not NaN
        else "—"
    )
    cols[5].metric("52W Range", range_52w)
