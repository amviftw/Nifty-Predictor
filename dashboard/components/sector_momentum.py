"""
Sectoral momentum heatmap: 16-week rolling view of sector performance.

Each cell = one week's return: filled green square = positive week, hollow red
square = negative week, with opacity scaling to magnitude. Column headers show
the actual week-start date pulled from the weekly bars themselves, so as new
weekly bars print on Yahoo Finance the right-most column auto-advances on the
next cache rotation — no manual update needed.
"""

import pandas as pd
import yfinance as yf
import streamlit as st
from loguru import logger

from dashboard.config import SECTORAL_INDICES, CACHE_TTL_SECONDS
from dashboard.data_loader import _market_minute_bucket
from dashboard.disk_cache import disk_cached

NUM_WEEKS = 16
RECENT_WINDOW = 5

# Palette aligned with the Groww-style dashboard CSS in app.py
POS = "#00d09c"
NEG = "#eb5757"
NEUTRAL = "#7a8294"
TEXT_PRIMARY = "#e8ecf1"
TEXT_SECONDARY = "#c9cfd9"
TEXT_MUTED = "#7a8294"
PANEL = "#151922"
BORDER = "#232834"


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
@disk_cached(name="sector_weekly_1y", ttl_hours=4)
def _fetch_sector_weekly(_bucket: str = "") -> pd.DataFrame:
    """Fetch ~1 year of weekly pct changes for all sectoral indices.

    `_bucket` is part of the cache key only; passing `_market_minute_bucket()`
    rotates the cache every minute during NSE hours and at every session
    boundary, so a freshly-printed weekly bar shows up on the next refresh.
    """
    del _bucket
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
            records[name] = close.pct_change() * 100
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).iloc[1:]


def _hit_rate(series: pd.Series, window: int = RECENT_WINDOW) -> int:
    return int((series.tail(window).dropna() > 0).sum())


def _current_streak(series: pd.Series) -> int:
    """Signed run-length of consecutive same-sign weeks ending at the latest week.

    Positive return value = current up-streak length, negative = down-streak.
    """
    vals = series.dropna().to_numpy()
    if len(vals) == 0 or vals[-1] == 0:
        return 0
    sign = 1 if vals[-1] > 0 else -1
    streak = 0
    for v in vals[::-1]:
        if (sign == 1 and v > 0) or (sign == -1 and v < 0):
            streak += 1
        else:
            break
    return streak * sign


def _cum_return(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    return ((1 + s / 100).prod() - 1) * 100


def _format_week_label(idx) -> str:
    """yfinance weekly bars are indexed by the Monday of the week."""
    if hasattr(idx, "strftime"):
        return idx.strftime("%b %d")
    return str(idx)


def render_sector_momentum():
    """Render the sectoral momentum grid."""
    full_df = _fetch_sector_weekly(_bucket=_market_minute_bucket())
    if full_df.empty:
        st.markdown("#### Sectoral Momentum — Weekly Performance Grid")
        st.info("Sector weekly data unavailable")
        return

    df = full_df.tail(NUM_WEEKS)
    sectors = list(df.columns)
    week_dates = list(df.index)
    date_labels = [_format_week_label(d) for d in week_dates]

    latest_label = date_labels[-1] if date_labels else "—"

    st.markdown("#### Sectoral Momentum — Weekly Performance Grid")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{TEXT_MUTED};margin:-4px 0 10px;line-height:1.55;">'
        f'Last {NUM_WEEKS} weeks &middot; week-ending '
        f'<span style="color:{TEXT_SECONDARY};">{latest_label}</span> &middot; '
        f'<span style="display:inline-block;width:8px;height:8px;background:{POS};'
        f'border-radius:1px;vertical-align:middle;margin-right:4px;"></span>filled = up week '
        f'&middot; '
        f'<span style="display:inline-block;width:8px;height:8px;border:1.5px solid {NEG};'
        f'border-radius:1px;vertical-align:middle;margin-right:4px;"></span>hollow = down week '
        f'&middot; opacity scales with magnitude &middot; '
        f'sorted by recent {RECENT_WINDOW}-week hit-rate then 4-week cumulative return &middot; '
        f'<span style="color:{POS};">auto-refreshes</span> as new weekly bars print on Yahoo.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Per-sector stats
    stats = {}
    for s in sectors:
        col = df[s].dropna()
        stats[s] = {
            "hit": _hit_rate(col, RECENT_WINDOW),
            "cum_4w": _cum_return(col.tail(4)),
            "cum_13w": _cum_return(col.tail(13)),
            "cum_1y": _cum_return(full_df[s]),
            "streak": _current_streak(col),
        }

    sorted_sectors = sorted(
        sectors,
        key=lambda s: (stats[s]["hit"], stats[s]["cum_4w"]),
        reverse=True,
    )

    recent_start = max(0, len(week_dates) - RECENT_WINDOW)

    # Magnitude normaliser — clamp so a single outlier doesn't wash everything out
    abs_vals = df.abs().to_numpy().flatten()
    abs_vals = abs_vals[~pd.isna(abs_vals)]
    if len(abs_vals):
        # Use 90th percentile so typical weeks read as full-intensity
        mag_norm = float(pd.Series(abs_vals).quantile(0.90)) or 1.0
    else:
        mag_norm = 1.0

    css = f"""
    <style>
    .smt-wrap {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 16px 10px;
        overflow-x: auto;
    }}
    .smt {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font: 13px/1.4 'Inter', -apple-system, Segoe UI, sans-serif;
        color: {TEXT_SECONDARY};
    }}
    .smt thead th {{
        font-size: 0.66rem;
        font-weight: 500;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 6px 4px 10px;
        border-bottom: 1px solid {BORDER};
        text-align: center;
        white-space: nowrap;
    }}
    .smt thead th.sector-h {{ text-align: left; padding-left: 4px; }}
    .smt thead th.dt {{
        font-size: 0.62rem;
        letter-spacing: 0.02em;
        text-transform: none;
        font-variant-numeric: tabular-nums;
    }}
    .smt thead th.dt.recent {{ color: {TEXT_SECONDARY}; font-weight: 600; }}
    .smt thead th.dt.current {{
        color: {POS};
        background: rgba(0,208,156,0.06);
        border-radius: 4px 4px 0 0;
    }}
    .smt thead th.divider {{ border-left: 1px solid {BORDER}; padding-left: 12px; }}
    .smt tbody td {{
        padding: 8px 4px;
        border-bottom: 1px solid rgba(35,40,52,0.6);
    }}
    .smt tbody tr:last-child td {{ border-bottom: none; }}
    .smt tbody tr:hover td {{ background: rgba(99,102,241,0.04); }}
    .smt td.sector {{
        text-align: left;
        padding-left: 4px;
        color: {TEXT_PRIMARY};
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.01em;
        white-space: nowrap;
    }}
    .smt td.cell {{ text-align: center; }}
    .smt td.cell.recent {{ background: rgba(255,255,255,0.018); }}
    .smt td.cell.current {{ background: rgba(0,208,156,0.06); }}
    .smt td.num {{
        text-align: right;
        padding: 8px 10px;
        font-variant-numeric: tabular-nums;
        font-weight: 500;
    }}
    .smt td.streak {{ text-align: center; font-variant-numeric: tabular-nums; }}
    .smt td.divider {{ border-left: 1px solid {BORDER}; padding-left: 12px; }}
    .smt .mark {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 2px;
        vertical-align: middle;
    }}
    .smt .mark.up   {{ background: {POS}; }}
    .smt .mark.down {{ background: transparent; border: 1.5px solid {NEG}; }}
    .smt .hit-bar {{
        display: inline-block;
        width: 32px;
        height: 5px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        position: relative;
        vertical-align: middle;
        margin-right: 8px;
    }}
    .smt .hit-bar > i {{
        position: absolute;
        left: 0; top: 0; bottom: 0;
        border-radius: 3px;
        display: block;
    }}
    </style>
    """

    parts = [css, '<div class="smt-wrap"><table class="smt">']

    # Header row
    parts.append("<thead><tr>")
    parts.append('<th class="sector-h">Sector</th>')
    for i, lbl in enumerate(date_labels):
        cls = ["dt"]
        if i >= recent_start:
            cls.append("recent")
        if i == len(date_labels) - 1:
            cls.append("current")
        parts.append(f'<th class="{" ".join(cls)}">{lbl}</th>')
    parts.append('<th class="divider">5W Hit</th>')
    parts.append("<th>Streak</th>")
    parts.append("<th>4W</th>")
    parts.append("<th>13W</th>")
    parts.append("<th>1Y</th>")
    parts.append("</tr></thead>")

    parts.append("<tbody>")
    for sector in sorted_sectors:
        col = df[sector]
        s = stats[sector]
        label = sector.replace("Nifty ", "").upper()

        parts.append(f'<tr><td class="sector">{label}</td>')

        # Per-week dot cells
        for i, val in enumerate(col):
            cls = ["cell"]
            if i >= recent_start:
                cls.append("recent")
            if i == len(col) - 1:
                cls.append("current")

            if pd.isna(val):
                marker = f'<span style="color:{TEXT_MUTED};opacity:0.4;">·</span>'
                tip = f"{date_labels[i]}: n/a"
            else:
                opacity = max(0.32, min(1.0, abs(val) / mag_norm))
                kind = "up" if val >= 0 else "down"
                marker = f'<span class="mark {kind}" style="opacity:{opacity:.2f};"></span>'
                tip = f"{date_labels[i]}: {val:+.1f}%"

            parts.append(f'<td class="{" ".join(cls)}" title="{tip}">{marker}</td>')

        # 5W Hit cell with bar
        hit = s["hit"]
        hit_pct = hit / RECENT_WINDOW * 100
        if hit >= 4:
            bar_color, text_color = POS, POS
        elif hit >= 3:
            bar_color, text_color = "#f0b034", "#f0b034"
        else:
            bar_color, text_color = NEG, NEG
        bar_html = (
            f'<span class="hit-bar"><i style="width:{hit_pct:.0f}%;background:{bar_color};"></i></span>'
            f'<span style="color:{text_color};font-weight:600;">{hit}/{RECENT_WINDOW}</span>'
        )
        parts.append(f'<td class="num divider" style="text-align:left;">{bar_html}</td>')

        # Streak
        streak = s["streak"]
        if streak > 0:
            streak_html = f'<span style="color:{POS};font-weight:600;">▲ {streak}</span>'
        elif streak < 0:
            streak_html = f'<span style="color:{NEG};font-weight:600;">▼ {abs(streak)}</span>'
        else:
            streak_html = f'<span style="color:{NEUTRAL};">—</span>'
        parts.append(f'<td class="streak">{streak_html}</td>')

        # 4W / 13W / 1Y cumulative
        for val, weight in ((s["cum_4w"], 600), (s["cum_13w"], 500), (s["cum_1y"], 400)):
            color = POS if val > 0 else NEG if val < 0 else NEUTRAL
            parts.append(
                f'<td class="num" style="color:{color};font-weight:{weight};">{val:+.1f}%</td>'
            )

        parts.append("</tr>")

    parts.append("</tbody></table></div>")

    st.markdown("".join(parts), unsafe_allow_html=True)
