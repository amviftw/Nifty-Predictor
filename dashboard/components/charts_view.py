"""
Charts view — embeds TradingView's free Advanced Chart Widget.

The widget is loaded via <script src="https://s3.tradingview.com/tv.js">
and rendered inside an iframe via streamlit.components.v1.html. It is
free, requires no API key, and ships the full TradingView UI: indicators,
drawing tools, timeframe switcher, crosshair, volume, compare, alerts.

We keep a thin Streamlit layer on top:
    - Instrument picker (Nifty 50 stocks, global & sector indices)
    - A small quote strip (last / change / day range / 52W) via yfinance
    - The widget fills the rest of the page

Symbol mapping: Yahoo Finance tickers -> TradingView symbols.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from config.nifty50_tickers import NIFTY50_STOCKS, GLOBAL_INDICES, SECTOR_INDICES


# ---------------------------------------------------------------------------
# Symbol mapping: Yahoo ticker -> TradingView symbol
# ---------------------------------------------------------------------------

# Global/index overrides. TradingView has its own naming scheme.
_TV_OVERRIDES: dict[str, str] = {
    "^NSEI": "NSE:NIFTY",
    "^NSEBANK": "NSE:BANKNIFTY",
    "^INDIAVIX": "NSE:INDIAVIX",
    "^GSPC": "SP:SPX",
    "^IXIC": "NASDAQ:IXIC",
    "^DJI": "TVC:DJI",
    "^FTSE": "TVC:UKX",
    "^N225": "TVC:NI225",
    "^HSI": "TVC:HSI",
    "INR=X": "FX_IDC:USDINR",
    # Sector indices
    "^CNXIT": "NSE:CNXIT",
    "^CNXPHARMA": "NSE:CNXPHARMA",
    "^CNXAUTO": "NSE:CNXAUTO",
    "^CNXFMCG": "NSE:CNXFMCG",
    "^CNXMETAL": "NSE:CNXMETAL",
    "^CNXINFRA": "NSE:CNXINFRA",
    "^CNXENERGY": "NSE:CNXENERGY",
    "^CNXFIN": "NSE:CNXFIN",
}


def _to_tv_symbol(yahoo_ticker: str) -> str:
    """Convert a Yahoo Finance ticker to a TradingView symbol."""
    if yahoo_ticker in _TV_OVERRIDES:
        return _TV_OVERRIDES[yahoo_ticker]
    if yahoo_ticker.endswith(".NS"):
        return f"NSE:{yahoo_ticker[:-3]}"
    # Fallback — TradingView will show a search prompt if unresolved.
    return yahoo_ticker


def _build_universe() -> dict[str, tuple[str, str]]:
    """
    Return {display_label: (yahoo_ticker, tradingview_symbol)} for the picker.
    """
    universe: dict[str, tuple[str, str]] = {}
    for name, tkr in GLOBAL_INDICES.items():
        universe[f"Index · {name}"] = (tkr, _to_tv_symbol(tkr))
    for name, tkr in SECTOR_INDICES.items():
        universe[f"Sector · {name}"] = (tkr, _to_tv_symbol(tkr))
    for sym, (tkr, company, _sector) in sorted(NIFTY50_STOCKS.items()):
        universe[f"{sym} · {company}"] = (tkr, _to_tv_symbol(tkr))
    return universe


# ---------------------------------------------------------------------------
# Quote strip (yfinance fast_info — free, no key)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_quote(yahoo_ticker: str) -> dict:
    """Fetch the tiny quote payload shown above the chart."""
    try:
        fi = yf.Ticker(yahoo_ticker).fast_info
        last = float(fi.get("last_price") or float("nan"))
        prev = float(fi.get("previous_close") or float("nan"))
        return {
            "last": last,
            "prev_close": prev,
            "day_high": float(fi.get("day_high") or float("nan")),
            "day_low": float(fi.get("day_low") or float("nan")),
            "year_high": float(fi.get("year_high") or float("nan")),
            "year_low": float(fi.get("year_low") or float("nan")),
            "currency": str(fi.get("currency") or "INR"),
            "change_pct": (last / prev - 1.0) * 100 if prev else float("nan"),
        }
    except Exception:
        return {
            "last": float("nan"), "prev_close": float("nan"),
            "day_high": float("nan"), "day_low": float("nan"),
            "year_high": float("nan"), "year_low": float("nan"),
            "currency": "INR", "change_pct": float("nan"),
        }


def _render_quote_strip(label: str, yahoo_ticker: str, tv_symbol: str, quote: dict):
    """Render the quote header above the widget."""
    change = quote["change_pct"]
    color = "#00d09c" if (change == change and change >= 0) else "#ef5350"
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
          <code style="background:#151922;padding:2px 6px;border-radius:4px;">{tv_symbol}</code>
          &nbsp;·&nbsp; Yahoo: <code style="background:#151922;padding:2px 6px;border-radius:4px;">{yahoo_ticker}</code>
          &nbsp;·&nbsp; {quote["currency"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _fmt(x: float) -> str:
        return f"{x:,.2f}" if x == x else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last", _fmt(quote["last"]))
    c2.metric("Prev Close", _fmt(quote["prev_close"]))
    c3.metric("Day High", _fmt(quote["day_high"]))
    c4.metric("Day Low", _fmt(quote["day_low"]))
    c5.metric(
        "52W Range",
        f"{quote['year_low']:,.0f} – {quote['year_high']:,.0f}"
        if quote["year_high"] == quote["year_high"] else "—",
    )


# ---------------------------------------------------------------------------
# TradingView widget
# ---------------------------------------------------------------------------

_TV_WIDGET_HTML = """
<div class="tradingview-widget-container" style="height:{height}px;width:100%;">
  <div id="tv_chart_advanced" style="height:100%;width:100%;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
    new TradingView.widget({{
      "autosize": true,
      "symbol": "{symbol}",
      "interval": "{interval}",
      "timezone": "Asia/Kolkata",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#0b0e14",
      "enable_publishing": false,
      "withdateranges": true,
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "save_image": false,
      "details": true,
      "studies": {studies},
      "show_popup_button": true,
      "popup_width": "1200",
      "popup_height": "760",
      "container_id": "tv_chart_advanced"
    }});
  </script>
</div>
"""


_DEFAULT_STUDIES = [
    "MAExp@tv-basicstudies",   # EMA (user can configure length in widget)
    "Volume@tv-basicstudies",  # Volume with MA
    "RSI@tv-basicstudies",     # RSI 14
]


def _render_tv_widget(tv_symbol: str, interval: str, studies: list[str], height: int):
    """Embed the TradingView Advanced Chart widget in an iframe."""
    import json
    html = _TV_WIDGET_HTML.format(
        symbol=tv_symbol,
        interval=interval,
        studies=json.dumps(studies),
        height=height,
    )
    components.html(html, height=height + 20, scrolling=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_TV_INTERVALS = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240",
    "1D": "D", "1W": "W", "1M": "M",
}


def render_charts_view():
    """Top-level render for the Charts mode."""
    st.markdown("### Charts")
    st.caption(
        "Full TradingView Advanced Chart — click anywhere on the chart to use "
        "crosshair, drawing tools, compare, alerts, and the complete indicator "
        "library. The symbol search inside the chart supports any TradingView ticker."
    )

    universe = _build_universe()
    labels = list(universe.keys())
    default_idx = labels.index("Index · NIFTY50") if "Index · NIFTY50" in labels else 0

    # --- Controls ---
    c1, c2, c3 = st.columns([3, 1.2, 1.5])
    with c1:
        label = st.selectbox("Instrument", labels, index=default_idx, key="tv_symbol_pick")
    with c2:
        interval_label = st.selectbox(
            "Interval",
            list(_TV_INTERVALS.keys()),
            index=list(_TV_INTERVALS.keys()).index("1D"),
            key="tv_interval_pick",
            help="Initial interval. Change inside the chart anytime.",
        )
    with c3:
        studies_choice = st.multiselect(
            "Pre-loaded indicators",
            ["EMA", "RSI", "MACD", "Bollinger", "Volume"],
            default=["EMA", "RSI", "Volume"],
            key="tv_studies_pick",
            help="More indicators available via the widget toolbar.",
        )

    yahoo_ticker, tv_symbol = universe[label]
    interval = _TV_INTERVALS[interval_label]

    # Map user-friendly names -> TradingView study ids
    study_map = {
        "EMA": "MAExp@tv-basicstudies",
        "RSI": "RSI@tv-basicstudies",
        "MACD": "MACD@tv-basicstudies",
        "Bollinger": "BB@tv-basicstudies",
        "Volume": "Volume@tv-basicstudies",
    }
    studies = [study_map[s] for s in studies_choice if s in study_map]

    # --- Quote strip ---
    quote = _fetch_quote(yahoo_ticker)
    _render_quote_strip(label, yahoo_ticker, tv_symbol, quote)

    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)

    # --- TradingView widget ---
    _render_tv_widget(tv_symbol, interval, studies, height=720)

    st.caption(
        "Powered by [TradingView](https://www.tradingview.com/) — free embedded widget. "
        "Quote strip data via Yahoo Finance."
    )
