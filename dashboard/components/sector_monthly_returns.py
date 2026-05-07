"""
Sector monthly returns heatmap — year × month grid for any sectoral index.

Modelled on the FundLens "Monthly Returns Heatmap" view: rows are calendar
years, columns are months, cells are the calendar-month return in % computed
from the index's first close in the month vs the last close in the month.
The right-hand TOTAL column is the calendar-year return (compounded across
filled months).

A single dropdown (selectbox) at the top lets the user toggle between the
broad market (Nifty 50) and each sectoral index — that's the directional
insight the rest of the dashboard wasn't giving at-a-glance.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from loguru import logger

from dashboard.config import SECTORAL_INDICES


# Palette — green = positive, red = negative, dark = no data.
POS_STRONG = "#0f5132"
POS = "#1b6e3a"
POS_LIGHT = "#225b3b"
NEG_LIGHT = "#5a1f1f"
NEG = "#8a2424"
NEG_STRONG = "#7a1d1d"
NEUTRAL = "#1a1f2b"
EMPTY = "#0f131c"

PANEL = "#151922"
BORDER = "#232834"
TEXT_PRIMARY = "#e8ecf1"
TEXT_SECONDARY = "#c9cfd9"
TEXT_MUTED = "#7a8294"

MONTH_LABELS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


# Index choices for the dropdown. Nifty 50 is broad-market default; the rest
# are the sectoral indices already wired up elsewhere on the dashboard.
_INDEX_CHOICES: dict[str, str] = {
    "Nifty 50 (Broad market)": "^NSEI",
    **{f"{name} (Sector)": ticker for name, ticker in SECTORAL_INDICES.items()},
}


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _fetch_index_history(ticker: str) -> pd.DataFrame:
    """Fetch up to ~max-period daily history for a single index ticker.

    Cached for 6 hours — monthly returns don't move intraday, so this is
    plenty. Returns a DataFrame indexed by date with a single 'Close' column.
    """
    try:
        data = yf.download(ticker, period="max", interval="1d",
                           auto_adjust=True, progress=False, threads=False)
        if data.empty:
            return pd.DataFrame()
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna().to_frame(name="Close")
    except Exception as e:
        logger.warning(f"Index history fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def _compute_monthly_grid(closes: pd.Series) -> pd.DataFrame:
    """Pivot daily closes into a year × month % return matrix."""
    if closes.empty:
        return pd.DataFrame()

    # Resample to month-end, compute pct change. The first observation will be
    # NaN — we fill it from the first available daily close in that month vs
    # the prior month's last close (which is what the resample naturally does
    # except for the very first month, where we leave it as NaN).
    monthly_last = closes.resample("ME").last()
    monthly_returns = monthly_last.pct_change() * 100

    grid = monthly_returns.to_frame(name="ret")
    grid["year"] = grid.index.year
    grid["month"] = grid.index.month

    pivot = grid.pivot(index="year", columns="month", values="ret")
    # Ensure all 12 month columns exist even if the early years lack history.
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[list(range(1, 13))]
    pivot.columns = MONTH_LABELS

    # Yearly compounded return across populated months.
    def _compound(row: pd.Series) -> float:
        vals = row.dropna()
        if vals.empty:
            return np.nan
        return (np.prod(1 + vals.to_numpy() / 100) - 1) * 100

    pivot["TOTAL"] = pivot[MONTH_LABELS].apply(_compound, axis=1)
    return pivot.sort_index()


def _cell_bg(val: float, scale: float) -> str:
    """Map a return value to a background colour, opacity scaled by magnitude."""
    if pd.isna(val):
        return EMPTY
    abs_val = min(abs(val), scale)
    intensity = abs_val / scale if scale else 0.0
    # Discretise into 3 bands for visual grouping.
    if val > 0:
        if intensity > 0.66:
            return POS_STRONG
        if intensity > 0.33:
            return POS
        return POS_LIGHT
    if val < 0:
        if intensity > 0.66:
            return NEG_STRONG
        if intensity > 0.33:
            return NEG
        return NEG_LIGHT
    return NEUTRAL


def _format_cell(val: float) -> str:
    if pd.isna(val):
        return "--"
    return f"{val:+.1f}" if abs(val) < 100 else f"{val:+.0f}"


def render_sector_monthly_returns():
    """Render the sector monthly returns heatmap with a sector toggle."""
    st.markdown("#### Sector Monthly Returns")

    label_to_ticker = _INDEX_CHOICES
    labels = list(label_to_ticker.keys())

    selector_col, info_col = st.columns([2, 3])
    with selector_col:
        selected_label = st.selectbox(
            "Index",
            labels,
            index=0,
            label_visibility="collapsed",
            key="sector_monthly_index",
        )
    with info_col:
        st.markdown(
            f'<div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:6px;line-height:1.5;">'
            f'Calendar-month % returns &middot; rows = years &middot; '
            f'<span style="color:{TEXT_SECONDARY};">TOTAL</span> = compounded yearly return &middot; '
            f'opacity scales with magnitude.'
            f'</div>',
            unsafe_allow_html=True,
        )

    ticker = label_to_ticker[selected_label]
    history = _fetch_index_history(ticker)
    if history.empty:
        st.info(f"History unavailable for {selected_label}")
        return

    grid = _compute_monthly_grid(history["Close"])
    if grid.empty:
        st.info("Not enough history to build a monthly grid yet.")
        return

    # Trim leading rows that are entirely NaN (e.g. partial first year of data).
    months_only = grid[MONTH_LABELS]
    non_empty_mask = months_only.notna().any(axis=1)
    grid = grid.loc[non_empty_mask]

    # Pick a colour scale anchored on the historical magnitude so a single
    # blow-out month doesn't flatten the whole grid.
    flat = months_only.to_numpy().flatten()
    flat = flat[~pd.isna(flat)]
    if len(flat):
        scale = float(np.percentile(np.abs(flat), 90))
    else:
        scale = 5.0
    scale = max(scale, 3.0)  # floor so quiet months still show colour

    css = f"""
    <style>
    .smr-wrap {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 16px 12px;
        overflow-x: auto;
    }}
    .smr {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 2px;
        font: 13px/1.4 'Inter', -apple-system, Segoe UI, sans-serif;
        color: {TEXT_SECONDARY};
        font-variant-numeric: tabular-nums;
    }}
    .smr thead th {{
        font-size: 0.66rem;
        font-weight: 500;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 8px 6px;
        background: transparent;
        text-align: center;
        white-space: nowrap;
    }}
    .smr thead th.year-h {{ text-align: left; padding-left: 6px; }}
    .smr thead th.total-h {{ color: {TEXT_SECONDARY}; }}
    .smr td {{
        padding: 10px 6px;
        text-align: center;
        font-size: 0.78rem;
        font-weight: 500;
        border-radius: 4px;
        min-width: 52px;
    }}
    .smr td.year {{
        text-align: left;
        padding-left: 6px;
        color: {TEXT_PRIMARY};
        font-weight: 600;
        font-size: 0.82rem;
        background: transparent;
        min-width: 64px;
    }}
    .smr td.total {{
        font-size: 0.82rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        min-width: 72px;
    }}
    .smr td.empty {{ color: {TEXT_MUTED}; opacity: 0.45; }}
    </style>
    """

    parts = [css, '<div class="smr-wrap"><table class="smr">']
    parts.append("<thead><tr>")
    parts.append('<th class="year-h">YEAR</th>')
    for m in MONTH_LABELS:
        parts.append(f'<th>{m}</th>')
    parts.append('<th class="total-h">TOTAL</th>')
    parts.append("</tr></thead><tbody>")

    for year, row in grid.iterrows():
        parts.append(f'<tr><td class="year">{year}</td>')
        for m in MONTH_LABELS:
            val = row[m]
            bg = _cell_bg(val, scale)
            text = _format_cell(val)
            cls = "empty" if pd.isna(val) else ""
            color = TEXT_PRIMARY if not pd.isna(val) else TEXT_MUTED
            parts.append(
                f'<td class="{cls}" style="background:{bg};color:{color};">{text}</td>'
            )
        total = row["TOTAL"]
        if pd.isna(total):
            parts.append('<td class="total empty">--</td>')
        else:
            tcolor = "#00d09c" if total > 0 else "#eb5757" if total < 0 else TEXT_PRIMARY
            parts.append(
                f'<td class="total" style="color:{tcolor};">{total:+.1f}</td>'
            )
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

    # Quick stats strip — handy for directional read-throughs.
    _render_summary_stats(months_only.loc[grid.index], selected_label)


def _render_summary_stats(months: pd.DataFrame, label: str):
    flat = months.to_numpy().flatten()
    flat = flat[~pd.isna(flat)]
    if not len(flat):
        return

    pos_pct = (flat > 0).mean() * 100
    avg_ret = flat.mean()
    best = flat.max()
    worst = flat.min()

    # Best / worst calendar months (which month name is strongest on average).
    monthly_avg = months.mean(axis=0)
    best_month = monthly_avg.idxmax()
    best_month_avg = monthly_avg.max()
    worst_month = monthly_avg.idxmin()
    worst_month_avg = monthly_avg.min()

    chip_specs = [
        ("Hit rate", f"{pos_pct:.0f}%", f"of {len(flat)} months positive", "#00d09c" if pos_pct >= 50 else "#eb5757"),
        ("Avg month", f"{avg_ret:+.1f}%", "across all history", "#00d09c" if avg_ret > 0 else "#eb5757"),
        ("Best month", f"{best:+.1f}%", "single-month peak", "#00d09c"),
        ("Worst month", f"{worst:+.1f}%", "single-month trough", "#eb5757"),
        ("Strongest", best_month, f"avg {best_month_avg:+.1f}% in {best_month}", "#00d09c"),
        ("Weakest", worst_month, f"avg {worst_month_avg:+.1f}% in {worst_month}", "#eb5757"),
    ]

    chips = []
    for title, headline, sub, color in chip_specs:
        chips.append(
            f'<div style="background:{PANEL};border:1px solid {BORDER};border-radius:10px;'
            f'padding:10px 12px;flex:1 1 150px;min-width:140px;">'
            f'<div style="font-size:0.62rem;color:{TEXT_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.06em;font-weight:500;">{title}</div>'
            f'<div style="font-size:1.0rem;font-weight:600;color:{color};margin-top:3px;">{headline}</div>'
            f'<div style="font-size:0.66rem;color:{TEXT_MUTED};margin-top:1px;">{sub}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"Stats computed over {months.index.min()}–{months.index.max()} for {label}.")
