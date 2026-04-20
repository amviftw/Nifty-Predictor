"""
Charts view — TradingView-quality candles via Lightweight Charts.

Uses streamlit-lightweight-charts (a wrapper around TradingView's OSS
lightweight-charts library). Rendering engine is identical to the one
on tradingview.com, but the data comes from yfinance, so NSE/BSE
instruments are not paywalled.

Layout:
    - Instrument / timeframe / indicator controls
    - Quote strip (last, prev close, day high/low, 52W range)
    - Stacked panes:
        1. Price candles  (+ EMA/SMA/Bollinger overlays)
        2. Volume histogram
        3. RSI (optional, with 30/70 guides)
        4. MACD (optional, line + signal + histogram)
"""

from __future__ import annotations

import streamlit as st

from config.nifty50_tickers import NIFTY50_STOCKS, GLOBAL_INDICES, SECTOR_INDICES
from dashboard.chart_data import (
    TIMEFRAMES,
    INTRADAY_INTERVALS,
    build_bundle,
    ChartBundle,
    candles_payload,
    volume_payload,
    line_payload,
    histogram_payload,
)

try:
    from streamlit_lightweight_charts import renderLightweightCharts
    _HAS_LWC = True
except ImportError:  # pragma: no cover
    _HAS_LWC = False


ALL_INTERVALS = ["1m", "5m", "15m", "30m", "60m", "1d", "1wk"]

# Color palette (matches the rest of the dashboard)
BULL = "#26a69a"
BEAR = "#ef5350"
EMA20_COL = "#42a5f5"
EMA50_COL = "#ab47bc"
SMA200_COL = "#ffa726"
BB_COL = "rgba(120, 144, 156, 0.7)"

BG = "#0b0e14"
TEXT = "#c9cfd9"
GRID = "#1a1f2b"


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def _build_universe() -> dict[str, str]:
    universe: dict[str, str] = {}
    for name, tkr in GLOBAL_INDICES.items():
        universe[f"Index · {name}"] = tkr
    for name, tkr in SECTOR_INDICES.items():
        universe[f"Sector · {name}"] = tkr
    for sym, (tkr, company, _sector) in sorted(NIFTY50_STOCKS.items()):
        universe[f"{sym} · {company}"] = tkr
    return universe


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _chart_options(height: int, show_time_scale: bool = True) -> dict:
    return {
        "height": height,
        "layout": {
            "background": {"type": "solid", "color": BG},
            "textColor": TEXT,
            "fontSize": 11,
        },
        "grid": {
            "vertLines": {"color": GRID},
            "horzLines": {"color": GRID},
        },
        "crosshair": {"mode": 1},  # magnet
        "rightPriceScale": {"borderColor": GRID},
        "timeScale": {
            "borderColor": GRID,
            "timeVisible": True,
            "secondsVisible": False,
            "visible": show_time_scale,
        },
    }


def _price_series(bundle: ChartBundle, indicators: dict[str, bool]) -> list[dict]:
    df = bundle.df
    intraday = bundle.is_intraday

    series: list[dict] = [{
        "type": "Candlestick",
        "data": candles_payload(df, intraday),
        "options": {
            "upColor": BULL, "downColor": BEAR,
            "borderVisible": False,
            "wickUpColor": BULL, "wickDownColor": BEAR,
            "priceLineWidth": 1,
        },
    }]

    if indicators["ema20"]:
        series.append({
            "type": "Line",
            "data": line_payload(df["EMA_20"], intraday),
            "options": {"color": EMA20_COL, "lineWidth": 1, "title": "EMA 20"},
        })
    if indicators["ema50"]:
        series.append({
            "type": "Line",
            "data": line_payload(df["EMA_50"], intraday),
            "options": {"color": EMA50_COL, "lineWidth": 1, "title": "EMA 50"},
        })
    if indicators["sma200"]:
        series.append({
            "type": "Line",
            "data": line_payload(df["SMA_200"], intraday),
            "options": {"color": SMA200_COL, "lineWidth": 2, "title": "SMA 200"},
        })
    if indicators["bbands"]:
        series.append({
            "type": "Line",
            "data": line_payload(df["BB_Upper"], intraday),
            "options": {"color": BB_COL, "lineWidth": 1, "title": "BB Upper"},
        })
        series.append({
            "type": "Line",
            "data": line_payload(df["BB_Lower"], intraday),
            "options": {"color": BB_COL, "lineWidth": 1, "title": "BB Lower"},
        })
    return series


def _volume_series(bundle: ChartBundle) -> list[dict]:
    return [{
        "type": "Histogram",
        "data": volume_payload(bundle.df, bundle.is_intraday),
        "options": {
            "priceFormat": {"type": "volume"},
            "priceScaleId": "",  # overlay on its own scale
        },
        "priceScale": {
            "scaleMargins": {"top": 0.1, "bottom": 0},
        },
    }]


def _rsi_series(bundle: ChartBundle) -> list[dict]:
    return [{
        "type": "Line",
        "data": line_payload(bundle.df["RSI"], bundle.is_intraday),
        "options": {
            "color": EMA20_COL, "lineWidth": 1, "title": "RSI 14",
            "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
        },
    }]


def _macd_series(bundle: ChartBundle) -> list[dict]:
    intraday = bundle.is_intraday
    return [
        {
            "type": "Histogram",
            "data": histogram_payload(bundle.df["Hist"], intraday),
            "options": {"title": "Hist", "priceFormat": {"type": "price", "precision": 3, "minMove": 0.001}},
        },
        {
            "type": "Line",
            "data": line_payload(bundle.df["MACD"], intraday),
            "options": {"color": EMA20_COL, "lineWidth": 1, "title": "MACD"},
        },
        {
            "type": "Line",
            "data": line_payload(bundle.df["Signal"], intraday),
            "options": {"color": SMA200_COL, "lineWidth": 1, "title": "Signal"},
        },
    ]


def _render_charts(bundle: ChartBundle, indicators: dict[str, bool]):
    """Stack price + volume + optional RSI + optional MACD panes."""
    charts: list[dict] = []

    charts.append({
        "chart": _chart_options(420, show_time_scale=False),
        "series": _price_series(bundle, indicators),
    })
    charts.append({
        "chart": _chart_options(120, show_time_scale=not (indicators["rsi"] or indicators["macd"])),
        "series": _volume_series(bundle),
    })
    if indicators["rsi"]:
        charts.append({
            "chart": _chart_options(140, show_time_scale=not indicators["macd"]),
            "series": _rsi_series(bundle),
        })
    if indicators["macd"]:
        charts.append({
            "chart": _chart_options(160, show_time_scale=True),
            "series": _macd_series(bundle),
        })

    renderLightweightCharts(charts, key=f"lwc_{bundle.symbol}_{len(charts)}")


# ---------------------------------------------------------------------------
# Quote strip
# ---------------------------------------------------------------------------

def _render_quote_strip(label: str, bundle: ChartBundle):
    change = bundle.day_change_pct
    color = BULL if (change == change and change >= 0) else BEAR
    arrow = "▲" if (change == change and change >= 0) else "▼"
    title = label.split(" · ", 1)[-1] if " · " in label else label

    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
          <div style="font-size:1.2rem;font-weight:700;color:#e8ecf1;">{title}</div>
          <div style="font-size:0.9rem;color:{color};font-weight:600;">
            {arrow} {change:+.2f}%
          </div>
        </div>
        <div style="font-size:0.78rem;color:#7a8294;margin-bottom:10px;">
          <code style="background:#151922;padding:2px 6px;border-radius:4px;">{bundle.symbol}</code>
          &nbsp;·&nbsp; {bundle.currency}
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _fmt(x: float) -> str:
        return f"{x:,.2f}" if x == x else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last", _fmt(bundle.last_price))
    c2.metric("Prev Close", _fmt(bundle.prev_close))
    c3.metric("Day High", _fmt(bundle.day_high))
    c4.metric("Day Low", _fmt(bundle.day_low))
    c5.metric(
        "52W Range",
        f"{bundle.week52_low:,.0f} – {bundle.week52_high:,.0f}"
        if bundle.week52_high == bundle.week52_high else "—",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_charts_view():
    st.markdown("### Charts")
    st.caption(
        "Interactive candles powered by TradingView's open-source Lightweight "
        "Charts library. Data from Yahoo Finance — NSE stocks and indices "
        "work without any paywall. Drag to pan, scroll to zoom, hover for OHLC."
    )

    if not _HAS_LWC:
        st.error(
            "`streamlit-lightweight-charts` is not installed. "
            "Add it to `requirements.txt` (already done) and rebuild the app."
        )
        return

    universe = _build_universe()
    labels = list(universe.keys())
    default_idx = labels.index("Index · NIFTY50") if "Index · NIFTY50" in labels else 0

    # --- Controls row 1: symbol / timeframe / interval ---
    c1, c2, c3 = st.columns([3, 1.2, 1.4])
    with c1:
        label = st.selectbox("Instrument", labels, index=default_idx, key="lwc_symbol")
    with c2:
        tf_key = st.selectbox(
            "Timeframe", list(TIMEFRAMES.keys()),
            index=list(TIMEFRAMES.keys()).index("6M"),
            key="lwc_tf",
        )
    with c3:
        default_interval = TIMEFRAMES[tf_key][1]
        interval_options = ["(auto)"] + ALL_INTERVALS
        interval_choice = st.selectbox(
            "Interval", interval_options,
            index=0,
            help=f"Auto picks {default_interval} for this timeframe.",
            key="lwc_interval",
        )

    override = None if interval_choice == "(auto)" else interval_choice
    ticker = universe[label]

    # --- Controls row 2: indicator toggles ---
    with st.expander("Indicators", expanded=True):
        i1, i2, i3, i4, i5, i6 = st.columns(6)
        indicators = {
            "ema20":  i1.checkbox("EMA 20", value=True),
            "ema50":  i2.checkbox("EMA 50", value=True),
            "sma200": i3.checkbox("SMA 200", value=False),
            "bbands": i4.checkbox("Bollinger", value=False),
            "rsi":    i5.checkbox("RSI (pane)", value=True),
            "macd":   i6.checkbox("MACD (pane)", value=False),
        }

    # --- Fetch ---
    with st.spinner(f"Loading {label}..."):
        bundle = build_bundle(ticker, tf_key, override_interval=override)

    if bundle.df.empty:
        st.error(
            f"No data returned for `{ticker}`. Try a different timeframe or "
            f"interval (Yahoo limits very short intervals to short lookback windows)."
        )
        return

    # --- Render ---
    _render_quote_strip(label, bundle)
    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    _render_charts(bundle, indicators)

    bars = len(bundle.df)
    st.caption(
        f"{bars} bars · {bundle.symbol} · "
        f"interval {override or TIMEFRAMES[tf_key][1]} · "
        f"data cached 60s · source: Yahoo Finance"
    )
