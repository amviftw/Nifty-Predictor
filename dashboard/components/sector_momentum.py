"""
Sectoral momentum heatmap: 16-week rolling view of sector performance.

Inspired by the "momentum grid" pattern — each cell represents one week,
filled/green when the sector was positive, hollow/grey when negative.
Shows at-a-glance which sectors are gaining or losing momentum.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from loguru import logger

from dashboard.config import SECTORAL_INDICES, CACHE_TTL_SECONDS

NUM_WEEKS = 16


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_sector_weekly() -> pd.DataFrame:
    """Fetch ~1 year of weekly pct changes for all sectoral indices."""
    tickers = list(SECTORAL_INDICES.values())
    try:
        data = yf.download(
            " ".join(tickers),
            period="1y",
            interval="1wk",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"Sector weekly fetch failed: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    records = {}
    for name, ticker in SECTORAL_INDICES.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.get_level_values(0):
                    close = data[ticker]["Close"].dropna()
                else:
                    continue
            else:
                close = data["Close"].dropna()
            if len(close) < 3:
                continue
            pct = close.pct_change() * 100
            records[name] = pct
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df.iloc[1:]  # drop first NaN row from pct_change; keep full year


def _momentum_score(series: pd.Series) -> int:
    """Count how many of the last 5 weeks were positive."""
    recent = series.tail(5).dropna()
    return int((recent > 0).sum())


def render_sector_momentum():
    """Render the sectoral momentum heatmap grid."""
    st.markdown("#### Sectoral Momentum — Weekly Performance Grid")
    st.caption(
        f"Last {NUM_WEEKS} weeks. Each block = one week. "
        "Filled = positive week, hollow = negative. "
        "Sorted by recent 5-week hit rate, then 4-week cumulative return."
    )

    full_df = _fetch_sector_weekly()
    if full_df.empty:
        st.info("Sector weekly data unavailable")
        return

    # Last NUM_WEEKS for the grid; full year kept for cum % (1Y)
    df = full_df.tail(NUM_WEEKS)

    sectors = list(df.columns)
    weeks = list(range(1, len(df) + 1))

    # Compute per-sector stats
    stats = {}
    for s in sectors:
        col = df[s].dropna()
        score = _momentum_score(col)
        cum_1y = ((1 + full_df[s].dropna() / 100).prod() - 1) * 100
        cum_4w = ((1 + col.tail(4) / 100).prod() - 1) * 100
        stats[s] = {"score": score, "cum_1y": cum_1y, "cum_4w": cum_4w}

    # Sort by momentum score descending, then by 4-week cumulative return
    sorted_sectors = sorted(sectors, key=lambda s: (stats[s]["score"], stats[s]["cum_4w"]), reverse=True)

    # Build the grid as HTML for rich rendering
    week_labels = [f"W{w}" for w in weeks]
    recent_start = max(0, len(weeks) - 5)

    html_parts = ['<div style="overflow-x:auto;">']
    html_parts.append(
        '<table style="border-collapse:collapse; width:100%; font-size:13px; font-family:sans-serif;">'
    )

    # Header row
    html_parts.append("<tr>")
    html_parts.append('<th style="text-align:left; padding:6px 10px; border-bottom:2px solid #333;">Sector</th>')
    for i, wl in enumerate(week_labels):
        bg = "rgba(255,255,255,0.05)" if i >= recent_start else ""
        style = f"background:{bg};" if bg else ""
        weight = "font-weight:700;" if i >= recent_start else "font-weight:400; color:#888;"
        html_parts.append(
            f'<th style="text-align:center; padding:4px 2px; border-bottom:2px solid #333; {style}{weight}">{wl}</th>'
        )
    html_parts.append('<th style="text-align:center; padding:6px 8px; border-bottom:2px solid #333; border-left:2px solid #555;">W{0}+</th>'.format(NUM_WEEKS - 4))
    html_parts.append('<th style="text-align:right; padding:6px 8px; border-bottom:2px solid #333;">4W Cum %</th>')
    html_parts.append('<th style="text-align:right; padding:6px 8px; border-bottom:2px solid #333;">1Y Cum %</th>')
    html_parts.append("</tr>")

    for sector in sorted_sectors:
        col = df[sector]
        score = stats[sector]["score"]
        cum_4w = stats[sector]["cum_4w"]
        cum_1y = stats[sector]["cum_1y"]

        # Sector name styling based on score
        if score >= 4:
            name_color = "#4ade80"
        elif score >= 3:
            name_color = "#fbbf24"
        else:
            name_color = "#f87171"

        html_parts.append("<tr>")
        label = sector.replace("Nifty ", "").upper()
        html_parts.append(
            f'<td style="padding:6px 10px; font-weight:600; color:{name_color}; '
            f'border-bottom:1px solid #222; white-space:nowrap;">{label}</td>'
        )

        for i, val in enumerate(col):
            is_recent = i >= recent_start
            if pd.isna(val):
                cell = '<span style="color:#555;">·</span>'
            elif val > 1.5:
                cell = '<span style="color:#22c55e; font-size:16px;">&#9632;</span>'
            elif val > 0:
                cell = '<span style="color:#4ade80; font-size:14px;">&#9632;</span>'
            elif val > -1.5:
                cell = '<span style="color:#f87171; font-size:14px;">&#9632;</span>'
            else:
                cell = '<span style="color:#ef4444; font-size:16px;">&#9632;</span>'

            bg = "rgba(255,255,255,0.03)" if is_recent else ""
            style = f"background:{bg};" if bg else ""
            html_parts.append(
                f'<td style="text-align:center; padding:4px 2px; border-bottom:1px solid #222; {style}">{cell}</td>'
            )

        # Score column
        html_parts.append(
            f'<td style="text-align:center; padding:6px 8px; border-bottom:1px solid #222; '
            f'border-left:2px solid #555; font-weight:700;">{score}/5</td>'
        )

        # 4-week cumulative %
        c4_color = "#4ade80" if cum_4w > 0 else "#f87171"
        html_parts.append(
            f'<td style="text-align:right; padding:6px 8px; border-bottom:1px solid #222; '
            f'color:{c4_color}; font-weight:600;">{cum_4w:+.1f}%</td>'
        )

        # 1-year cumulative %
        c1_color = "#4ade80" if cum_1y > 0 else "#f87171"
        html_parts.append(
            f'<td style="text-align:right; padding:6px 8px; border-bottom:1px solid #222; '
            f'color:{c1_color};">{cum_1y:+.1f}%</td>'
        )
        html_parts.append("</tr>")

    # Divider between high and low momentum
    # (already sorted, visual separation via color is enough)

    html_parts.append("</table></div>")

    st.markdown("".join(html_parts), unsafe_allow_html=True)

    # Legend
    st.caption(
        "&#9632; Large green = >1.5% up · Small green = 0–1.5% up · "
        "Small red = 0–1.5% down · Large red = >1.5% down · "
        "Score = positive weeks in last 5"
    )

    # Commentary
    strong = [s for s in sorted_sectors if stats[s]["score"] >= 4]
    weak = [s for s in sorted_sectors if stats[s]["score"] <= 1]

    if strong:
        names = ", ".join(s.replace("Nifty ", "") for s in strong)
        st.success(f"Strong momentum: **{names}** — 4+ positive weeks in last 5")
    if weak:
        names = ", ".join(s.replace("Nifty ", "") for s in weak)
        st.error(f"Weak momentum: **{names}** — 1 or fewer positive weeks in last 5")
