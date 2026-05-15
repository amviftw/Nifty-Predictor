"""
Stock View — single-stock deep-dive inspired by the Apple Stocks app.

A user picks any ticker (Nifty 50 / Next 50 / Midcap autocompletes from the
dashboard's `EXPANDED_UNIVERSE`; anything else is forwarded to Yahoo as a
free-form symbol so US/global tickers like AAPL or BTC-USD work too) and
gets a single full-bleed panel with:

    - Big price headline + signed day change in a colored pill
    - Smooth area chart with a 1D / 5D / 1M / 3M / 6M / 1Y / 5Y / Max toggle
    - 52-week range bar with a live "where we are now" marker
    - Key stats grid (open / prev close / day range / 52W range / market
      cap / P/E / dividend yield / beta / avg volume)
    - Latest headlines pulled via `yfinance.Ticker(...).news`

The component is intentionally read-only — no model predictions, no
broker-target overlays. Everything else on the dashboard speaks the Indian
universe; this surface is where a user goes when they want to look at any
single name (Indian or otherwise) with low ceremony.
"""

from __future__ import annotations

from datetime import datetime, date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from loguru import logger

from dashboard.config import CACHE_TTL_SECONDS
from dashboard.data_loader import _market_minute_bucket
from dashboard.disk_cache import disk_cached
from dashboard.universe import EXPANDED_UNIVERSE


# Colors aligned with the rest of the dashboard
PRICE_UP = "#00d09c"
PRICE_DN = "#eb5757"
PANEL = "#0f131c"
CARD = "#151922"
BORDER = "#232834"
TEXT_PRIMARY = "#e8ecf1"
TEXT_MUTED = "#7a8294"
TEXT_SECONDARY = "#c9cfd9"


# (label, yfinance period, yfinance interval). Intervals chosen to mirror
# the Apple Stocks app's density at each zoom level — 1D is intraday,
# 5D / 1M move to hourly+daily, longer frames stick to daily / weekly.
_TIMEFRAMES: list[tuple[str, str, str]] = [
    ("1D", "1d", "5m"),
    ("5D", "5d", "30m"),
    ("1M", "1mo", "1d"),
    ("3M", "3mo", "1d"),
    ("6M", "6mo", "1d"),
    ("1Y", "1y", "1d"),
    ("5Y", "5y", "1wk"),
    ("Max", "max", "1mo"),
]


def _normalize_input_ticker(raw: str) -> str:
    """Map a user input string to a Yahoo ticker.

    The autocomplete options encode "SYMBOL — Company" so we strip the
    pretty suffix first. Anything that resolves to a known NSE symbol from
    `EXPANDED_UNIVERSE` is rewritten to its Yahoo form (`.NS` suffix);
    everything else is uppercased and passed through unmodified so US /
    crypto tickers (AAPL, MSFT, BTC-USD, ^GSPC) work as-is.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    # Strip " — Company" suffix from the autocomplete options
    if "—" in s:
        s = s.split("—", 1)[0].strip()
    if " - " in s:
        s = s.split(" - ", 1)[0].strip()

    # NSE symbol shortcut
    info = EXPANDED_UNIVERSE.get(s.upper())
    if info:
        return info[0]
    return s.upper()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_history(ticker: str, period: str, interval: str, _bucket: str = "") -> pd.DataFrame:
    """Fetch OHLC history for a single ticker at the requested resolution."""
    del _bucket
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna(subset=["Close"])
    except Exception as e:
        logger.warning(f"stock_view history fetch failed for {ticker} {period}/{interval}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
@disk_cached(name="stock_meta", ttl_hours=6)
def _fetch_meta(ticker: str, _bucket: str = "") -> dict:
    """Fetch the slow-moving / structural metadata for a ticker.

    Disk-cached for 6h because the values here (market cap, 52w range,
    dividend yield, beta, sector / exchange labels) move on a daily — not
    intraday — cadence. Streamlit's in-process cache layers on top at 15
    minutes so live sessions never block on this call.
    """
    del _bucket
    out: dict = {
        "long_name": ticker,
        "short_name": ticker,
        "currency": "",
        "exchange": "",
        "sector": "",
        "industry": "",
        "market_cap": None,
        "pe": None,
        "forward_pe": None,
        "dividend_yield": None,
        "beta": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "avg_volume": None,
        "previous_close": None,
        "open": None,
        "day_high": None,
        "day_low": None,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Prefer fast_info for price-adjacent fields (cheap to hit, always fresh)
        fi = None
        try:
            fi = t.fast_info
        except Exception:
            fi = None

        def _from_fast_info(key):
            if fi is None:
                return None
            try:
                if hasattr(fi, "get"):
                    v = fi.get(key)
                    if v is not None:
                        return v
            except Exception:
                pass
            return getattr(fi, key, None)

        out["long_name"] = info.get("longName") or info.get("shortName") or ticker
        out["short_name"] = info.get("shortName") or info.get("longName") or ticker
        out["currency"] = info.get("currency") or _from_fast_info("currency") or ""
        out["exchange"] = info.get("fullExchangeName") or info.get("exchange") or _from_fast_info("exchange") or ""
        out["sector"] = info.get("sector") or ""
        out["industry"] = info.get("industry") or ""
        out["market_cap"] = info.get("marketCap") or _from_fast_info("market_cap")
        out["pe"] = info.get("trailingPE")
        out["forward_pe"] = info.get("forwardPE")
        out["dividend_yield"] = info.get("dividendYield")
        out["beta"] = info.get("beta")
        out["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh") or _from_fast_info("year_high")
        out["fifty_two_week_low"] = info.get("fiftyTwoWeekLow") or _from_fast_info("year_low")
        out["avg_volume"] = info.get("averageVolume") or _from_fast_info("ten_day_average_volume")
        out["previous_close"] = info.get("regularMarketPreviousClose") or _from_fast_info("previous_close")
        out["open"] = info.get("regularMarketOpen") or _from_fast_info("open")
        out["day_high"] = info.get("dayHigh") or _from_fast_info("day_high")
        out["day_low"] = info.get("dayLow") or _from_fast_info("day_low")
    except Exception as e:
        logger.warning(f"stock_view meta fetch failed for {ticker}: {e}")
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_news(ticker: str, _bucket: str = "") -> list[dict]:
    """Pull recent headlines via `Ticker.news`. Best-effort — failures are silent."""
    del _bucket
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out: list[dict] = []
    for it in items[:8]:
        # yfinance is in transition between two news payload shapes
        title = it.get("title") or it.get("content", {}).get("title")
        publisher = it.get("publisher") or it.get("content", {}).get("provider", {}).get("displayName", "")
        link = it.get("link") or it.get("content", {}).get("canonicalUrl", {}).get("url")
        ts = it.get("providerPublishTime") or 0
        if not title or not link:
            continue
        out.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "published": ts,
        })
    return out


def _format_currency(value: float | None, currency: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    symbol = "₹" if currency.upper() == "INR" else ("$" if currency.upper() == "USD" else "")
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"{symbol}{value / 1e12:,.2f}T"
    if abs_v >= 1e9:
        return f"{symbol}{value / 1e9:,.2f}B"
    if abs_v >= 1e7:
        return f"{symbol}{value / 1e7:,.2f}Cr"
    if abs_v >= 1e5:
        return f"{symbol}{value / 1e5:,.2f}L"
    if abs_v >= 1e3:
        return f"{symbol}{value / 1e3:,.1f}K"
    return f"{symbol}{value:,.2f}"


def _format_volume(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{int(value):,}"


def _format_pct(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    # yfinance dividend_yield comes as either 0.012 (fraction) or 1.2 (already percent)
    return f"{value:.{decimals}f}%"


def _format_price(value: float | None, currency: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    symbol = "₹" if currency.upper() == "INR" else ("$" if currency.upper() == "USD" else "")
    return f"{symbol}{value:,.2f}"


def _ticker_options() -> list[str]:
    """Autocomplete options: 'SYMBOL — Company' for the expanded universe."""
    items = []
    for sym, (_, company, _sector) in sorted(EXPANDED_UNIVERSE.items()):
        items.append(f"{sym} — {company}")
    return items


def _build_chart(df: pd.DataFrame, up: bool, intraday: bool) -> go.Figure:
    """Apple-Stocks-style smooth area + line chart."""
    color = PRICE_UP if up else PRICE_DN
    fill_color = (
        "rgba(0,208,156,0.12)" if up else "rgba(235,87,87,0.12)"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines",
        line=dict(color=color, width=2.0, shape="spline", smoothing=0.6),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate=(
            "%{x|%d %b %Y · %H:%M}<br><b>%{y:,.2f}</b><extra></extra>"
            if intraday
            else "%{x|%d %b %Y}<br><b>%{y:,.2f}</b><extra></extra>"
        ),
    ))

    # Lower-bound the y-axis just below the series min so the fill reads
    # like a price band rather than a flat slab from zero.
    y_min = float(df["Close"].min())
    y_max = float(df["Close"].max())
    pad = (y_max - y_min) * 0.08 or y_max * 0.005
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=300,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0b0e14",
            bordercolor="#2f3645",
            font=dict(family="Inter, sans-serif", size=12, color=TEXT_PRIMARY),
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
            tickformat=",.2f",
            range=[max(0, y_min - pad), y_max + pad],
        ),
        showlegend=False,
    )
    return fig


def _stat_card_html(label: str, value: str) -> str:
    return (
        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;'
        f'padding:12px 14px;flex:1 1 130px;min-width:130px;">'
        f'<div style="font-size:0.62rem;color:{TEXT_MUTED};text-transform:uppercase;'
        f'letter-spacing:0.06em;font-weight:500;">{label}</div>'
        f'<div style="font-size:0.95rem;color:{TEXT_PRIMARY};font-weight:600;'
        f'margin-top:4px;letter-spacing:-0.01em;">{value}</div>'
        f'</div>'
    )


def _render_52w_range(low: float | None, high: float | None, current: float, currency: str):
    """Apple-Stocks-style 52-week range bar with a live marker."""
    if low is None or high is None or high <= low:
        return
    pos_pct = max(0.0, min(100.0, (current - low) / (high - low) * 100))
    st.markdown(
        f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;'
        f'padding:14px 16px;margin-top:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:0.66rem;color:{TEXT_MUTED};text-transform:uppercase;'
        f'letter-spacing:0.06em;font-weight:500;">52-week range</span>'
        f'<span style="font-size:0.75rem;color:{TEXT_SECONDARY};">'
        f'{_format_price(current, currency)} &middot; '
        f'<span style="color:{TEXT_MUTED};">{pos_pct:.0f}% of range</span></span>'
        f'</div>'
        f'<div style="position:relative;height:6px;background:linear-gradient(90deg,'
        f'rgba(235,87,87,0.4) 0%, rgba(255,255,255,0.06) 50%, rgba(0,208,156,0.4) 100%);'
        f'border-radius:3px;margin-top:10px;">'
        f'<div style="position:absolute;left:{pos_pct:.1f}%;top:-4px;width:3px;height:14px;'
        f'background:{TEXT_PRIMARY};border-radius:2px;transform:translateX(-1px);'
        f'box-shadow:0 0 0 2px rgba(232,236,241,0.15);"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:8px;'
        f'font-size:0.72rem;color:{TEXT_MUTED};font-variant-numeric:tabular-nums;">'
        f'<span>{_format_price(low, currency)}</span>'
        f'<span>{_format_price(high, currency)}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_headline(meta: dict, latest: float, prev: float, ticker: str):
    """Big-price headline strip — name, ticker, price, signed change pill."""
    currency = meta.get("currency") or ""
    delta = latest - prev if prev else 0.0
    pct = (delta / prev * 100) if prev else 0.0
    up = delta >= 0
    color = PRICE_UP if up else PRICE_DN
    sign = "+" if up else ""

    sub_bits = [b for b in (meta.get("exchange"), meta.get("sector")) if b]
    sub = " &middot; ".join(sub_bits)

    st.markdown(
        f'<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:14px;">'
        f'<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">'
        f'<span style="font-size:1.4rem;font-weight:700;color:{TEXT_PRIMARY};'
        f'letter-spacing:-0.02em;">{meta.get("long_name", ticker)}</span>'
        f'<span style="font-size:0.85rem;color:{TEXT_MUTED};font-weight:500;">{ticker}</span>'
        f'</div>'
        f'<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">'
        f'<span style="font-size:2.6rem;font-weight:700;color:{TEXT_PRIMARY};'
        f'letter-spacing:-0.03em;line-height:1.05;">{_format_price(latest, currency)}</span>'
        f'<span style="display:inline-flex;align-items:center;gap:6px;padding:6px 12px;'
        f'border-radius:8px;background:{color}22;color:{color};font-weight:600;'
        f'font-size:0.95rem;">'
        f'{sign}{delta:,.2f} &nbsp; {sign}{pct:.2f}%'
        f'</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:{TEXT_MUTED};">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_news(news: list[dict]):
    if not news:
        return
    st.markdown(
        f'<div style="margin-top:18px;font-size:0.78rem;color:{TEXT_MUTED};'
        f'text-transform:uppercase;letter-spacing:0.06em;font-weight:500;">'
        f'Latest headlines</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for it in news:
        ts = it.get("published") or 0
        try:
            when = datetime.fromtimestamp(ts).strftime("%d %b · %H:%M") if ts else ""
        except Exception:
            when = ""
        rows.append(
            f'<a href="{it["link"]}" target="_blank" rel="noopener" '
            f'style="display:block;background:{CARD};border:1px solid {BORDER};'
            f'border-radius:10px;padding:10px 14px;margin-top:8px;'
            f'text-decoration:none;color:{TEXT_SECONDARY};">'
            f'<div style="font-size:0.82rem;font-weight:500;color:{TEXT_PRIMARY};'
            f'line-height:1.35;">{it["title"]}</div>'
            f'<div style="font-size:0.7rem;color:{TEXT_MUTED};margin-top:4px;">'
            f'{it.get("publisher", "")}{" &middot; " + when if when else ""}</div>'
            f'</a>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_stock_view():
    """Top-level entry point for the Stock View mode."""
    st.markdown("#### Stock View")
    st.caption(
        "Look up any ticker — autocomplete covers Nifty 50 + Next 50 + Midcap; "
        "anything else is forwarded to Yahoo Finance, so US, ETF, and crypto tickers "
        "(AAPL, BTC-USD, ^GSPC) work as-is."
    )

    options = [""] + _ticker_options()
    sel_col, manual_col = st.columns([2, 1])
    with sel_col:
        choice = st.selectbox(
            "Search",
            options,
            index=options.index("RELIANCE — Reliance Industries")
            if "RELIANCE — Reliance Industries" in options
            else 0,
            label_visibility="collapsed",
            key="stock_view_select",
        )
    with manual_col:
        manual = st.text_input(
            "Or type a Yahoo ticker",
            value="",
            placeholder="e.g. AAPL, BTC-USD, ^GSPC",
            label_visibility="collapsed",
            key="stock_view_manual",
        )

    ticker = _normalize_input_ticker(manual or choice)
    if not ticker:
        st.info("Pick a stock from the dropdown or type a Yahoo ticker to begin.")
        return

    # Timeframe pill row
    tf_labels = [tf[0] for tf in _TIMEFRAMES]
    default_idx = tf_labels.index("1M")
    tf_label = st.radio(
        "Timeframe",
        tf_labels,
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
        key=f"stock_view_tf_{ticker}",
    )
    tf = next(t for t in _TIMEFRAMES if t[0] == tf_label)
    _, period, interval = tf

    bucket = _market_minute_bucket()
    df = _fetch_history(ticker, period, interval, _bucket=bucket)
    if df.empty:
        st.warning(f"No data returned for `{ticker}` — try a different symbol or timeframe.")
        return

    meta = _fetch_meta(ticker, _bucket=date.today().isoformat())
    news = _fetch_news(ticker, _bucket=bucket)

    latest = float(df["Close"].iloc[-1])
    # For 1D / 5D intraday charts the "prev close" is the daily prior-close
    # field from meta, not the first bar (which would be today's open). For
    # daily+ resolutions, the chart's first bar is a clean anchor.
    if interval in ("5m", "30m", "1h") and meta.get("previous_close"):
        prev = float(meta["previous_close"])
    else:
        prev = float(df["Close"].iloc[0])

    _render_headline(meta, latest, prev, ticker)

    chart_col, stats_col = st.columns([2.2, 1])

    with chart_col:
        fig = _build_chart(df, up=latest >= prev, intraday=interval in ("5m", "30m", "1h"))
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        # 52w range bar lives under the chart so it shares the wider column
        _render_52w_range(
            meta.get("fifty_two_week_low"),
            meta.get("fifty_two_week_high"),
            latest,
            meta.get("currency") or "",
        )

    with stats_col:
        st.markdown(
            f'<div style="font-size:0.66rem;color:{TEXT_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.06em;font-weight:500;margin-bottom:8px;">Key stats</div>',
            unsafe_allow_html=True,
        )

        currency = meta.get("currency") or ""
        cards = [
            ("Open", _format_price(meta.get("open"), currency)),
            ("Prev Close", _format_price(meta.get("previous_close"), currency)),
            ("Day Low", _format_price(meta.get("day_low"), currency)),
            ("Day High", _format_price(meta.get("day_high"), currency)),
            ("52W Low", _format_price(meta.get("fifty_two_week_low"), currency)),
            ("52W High", _format_price(meta.get("fifty_two_week_high"), currency)),
            ("Market Cap", _format_currency(meta.get("market_cap"), currency)),
            ("Avg Volume", _format_volume(meta.get("avg_volume"))),
            ("P/E (TTM)", f"{meta['pe']:.1f}" if meta.get("pe") else "—"),
            ("Fwd P/E", f"{meta['forward_pe']:.1f}" if meta.get("forward_pe") else "—"),
            ("Beta", f"{meta['beta']:.2f}" if meta.get("beta") else "—"),
            (
                "Div Yield",
                _format_pct(
                    meta["dividend_yield"] * 100
                    if meta.get("dividend_yield") and meta["dividend_yield"] < 1
                    else meta.get("dividend_yield"),
                )
                if meta.get("dividend_yield") is not None
                else "—",
            ),
        ]
        cards_html = "".join(_stat_card_html(label, value) for label, value in cards)
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{cards_html}</div>',
            unsafe_allow_html=True,
        )

    _render_news(news)
