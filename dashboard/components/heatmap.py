"""Market heatmap — Finviz-style treemap of the Nifty 50 + Midcap universe.

Sectors form the parent clusters (correlation buckets); individual stocks are
sized by market cap and coloured by % change. Daily / weekly is driven by the
top-level view toggle in app.py.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import load_heatmap_data, MarketSnapshot


# Symmetric red ↔ neutral ↔ green colourscale, anchored at 0%.
# Mid-grey at 0 keeps unchanged stocks visually quiet so movers pop.
_COLORSCALE = [
    [0.00, "#7a1d1d"],   # deep red
    [0.25, "#c53030"],
    [0.45, "#5a2424"],
    [0.50, "#2a2f3a"],   # neutral / unchanged
    [0.55, "#1f4d3a"],
    [0.75, "#1ea36a"],
    [1.00, "#00d09c"],   # vivid green
]


def render_heatmap(view: str = "daily", snapshot: MarketSnapshot | None = None):
    """Render the full-market heatmap inside a tab.

    `snapshot` carries the Nifty 50 stock changes already computed by
    `load_market_snapshot` — reusing them guarantees the heatmap shows the
    same DoD / WoW numbers as Top Movers and the rest of the dashboard.
    """
    st.markdown("#### Market Heatmap")
    as_of = snapshot.last_trading_date if snapshot else ""
    as_of_html = (
        f' &middot; <span style="color:#c9cfd9;">As of {as_of}</span>'
        if as_of else ""
    )
    st.markdown(
        '<div style="font-size:0.78rem;color:#7a8294;margin-top:-4px;margin-bottom:10px;">'
        "Nifty 50 + curated Midcap universe, clustered by sector. "
        "Tile size = market cap, colour = "
        f"{'day' if view == 'daily' else 'week'} % change."
        f"{as_of_html}"
        "</div>",
        unsafe_allow_html=True,
    )

    df = load_heatmap_data(view=view, snapshot=snapshot)
    if df is None or df.empty:
        st.info("Heatmap data unavailable — try refreshing in a moment.")
        return

    # Per-view metric the heatmap visualises
    chg_col = "dod_pct" if view == "daily" else "wow_pct"

    # ---- Filters: minimum-cap floor, sector multi-select ------------------
    sectors_all = sorted(df["sector"].dropna().unique().tolist())

    f1, f2, f3 = st.columns([1.4, 2.2, 1.2])
    with f1:
        size_metric = st.radio(
            "Tile size",
            ["Market cap", "Equal weight"],
            horizontal=True,
            label_visibility="visible",
            key="heatmap_size_metric",
        )
    with f2:
        selected_sectors = st.multiselect(
            "Sectors",
            sectors_all,
            default=sectors_all,
            key="heatmap_sectors",
        )
    with f3:
        clip = st.slider(
            "Colour range (±%)",
            min_value=2.0, max_value=10.0, value=4.0, step=0.5,
            key="heatmap_clip",
            help="Caps the colour gradient. Smaller = more sensitive to small moves.",
        )

    if selected_sectors:
        df = df[df["sector"].isin(selected_sectors)].copy()

    if df.empty:
        st.info("No stocks match the current filter.")
        return

    if size_metric == "Equal weight":
        df["size"] = 1.0

    # Sort sectors by aggregate (mcap-weighted) % change so the strongest
    # cluster sits in the top-left, matching the Finviz convention.
    sector_score = (
        df.assign(_w=df[chg_col] * df["size"])
        .groupby("sector")
        .agg(_w=("_w", "sum"), _s=("size", "sum"))
    )
    sector_score["score"] = (sector_score["_w"] / sector_score["_s"]).fillna(0.0)
    df["sector_score"] = df["sector"].map(sector_score["score"])

    fig = _build_treemap(df, chg_col, view, clip)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    _render_summary_strip(df, chg_col, view)


def _build_treemap(df: pd.DataFrame, chg_col: str, view: str, clip: float) -> go.Figure:
    """Two-level treemap: root → sector → stock."""

    # Build hierarchy rows: one per stock + one per sector + a single root.
    sector_agg = (
        df.groupby("sector")
        .agg(size=("size", "sum"), score=("sector_score", "first"))
        .reset_index()
    )

    root_label = "Indian Market"

    labels: list[str] = [root_label]
    parents: list[str] = [""]
    ids: list[str] = [root_label]
    values: list[float] = [float(df["size"].sum())]
    colors: list[float] = [0.0]
    # customdata[0] = pre-rendered hover text. Keeps hover useful at every
    # level (root / sector / stock) with a single hovertemplate.
    chg_label = "DoD" if view == "daily" else "WoW"
    custom: list[list] = [[
        f"<b>Indian Market</b><br>"
        f"{len(df)} stocks across {len(sector_agg)} sectors"
    ]]

    # Sector parents
    sector_agg = sector_agg.sort_values("score", ascending=False)
    for _, r in sector_agg.iterrows():
        sec = str(r["sector"])
        score = float(r["score"])
        labels.append(sec)
        parents.append(root_label)
        ids.append(f"sec::{sec}")
        values.append(float(r["size"]))
        colors.append(score)
        sec_count = int((df["sector"] == sec).sum())
        custom.append([
            f"<b>{sec}</b><br>"
            f"{sec_count} stocks &middot; mcap-weighted {chg_label}: "
            f"{score:+.1f}%"
        ])

    # Stock leaves — sort within sector by size desc so big names dominate
    df_sorted = df.sort_values(["sector_score", "size"], ascending=[False, False])
    for _, r in df_sorted.iterrows():
        sec = str(r["sector"])
        sym = str(r["symbol"])
        labels.append(sym)
        parents.append(f"sec::{sec}")
        ids.append(f"stk::{sym}")
        values.append(float(r["size"]))
        colors.append(float(r[chg_col]))
        mcap_str = _format_inr_crore(float(r.get("market_cap", 0) or 0))
        custom.append([
            f"<b>{sym}</b> &nbsp; {r['company']}<br>"
            f"Sector: {sec}<br>"
            f"Close: ₹{r['close']:,.2f}<br>"
            f"DoD: {r['dod_pct']:+.1f}% &nbsp; WoW: {r['wow_pct']:+.1f}%<br>"
            f"Mcap: {mcap_str}"
        ])

    # Tile text: "SYMBOL\n+1.2%" — only on stock leaves (sectors get their name)
    chg_by_id = dict(zip(ids, colors))
    text = []
    for lbl, parent, _id in zip(labels, parents, ids):
        if parent == root_label or lbl == root_label:
            text.append(f"<b>{lbl}</b>")
        else:
            stk_chg = chg_by_id.get(_id, 0.0)
            sign = "+" if stk_chg >= 0 else ""
            text.append(f"<b>{lbl}</b><br>{sign}{stk_chg:.1f}%")

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        ids=ids,
        values=values,
        text=text,
        textinfo="text",
        textposition="middle center",
        textfont=dict(family="Inter, sans-serif", size=13, color="#f3f5f9"),
        insidetextfont=dict(family="Inter, sans-serif", size=13, color="#f3f5f9"),
        branchvalues="total",
        marker=dict(
            colors=colors,
            colorscale=_COLORSCALE,
            cmin=-clip,
            cmid=0,
            cmax=clip,
            line=dict(color="#0b0e14", width=1.5),
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=("Day %" if view == "daily" else "Week %"),
                    font=dict(color="#c9cfd9", size=11),
                ),
                tickfont=dict(color="#c9cfd9", size=10),
                len=0.6, thickness=12, x=1.01,
                outlinewidth=0,
                ticksuffix="%",
            ),
        ),
        customdata=custom,
        hovertemplate="%{customdata[0]}<extra></extra>",
        hoverlabel=dict(
            bgcolor="#0f131c",
            bordercolor="#2f3645",
            font=dict(family="Inter, sans-serif", size=12, color="#e8ecf1"),
        ),
        pathbar=dict(
            visible=True,
            thickness=22,
            textfont=dict(color="#c9cfd9", size=11),
        ),
        tiling=dict(packing="squarify", squarifyratio=1.4, pad=2),
    ))

    fig.update_layout(
        margin=dict(l=4, r=4, t=8, b=4),
        height=720,
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0b0e14",
        font=dict(family="Inter, sans-serif", color="#c9cfd9"),
        uniformtext=dict(minsize=9, mode="hide"),
    )
    return fig


def _render_summary_strip(df: pd.DataFrame, chg_col: str, view: str):
    """Three small cards under the heatmap: breadth, top sector, top stock."""
    total = len(df)
    advancers = int((df[chg_col] > 0.1).sum())
    decliners = int((df[chg_col] < -0.1).sum())

    sec_score = (
        df.assign(_w=df[chg_col] * df["size"])
        .groupby("sector")
        .agg(_w=("_w", "sum"), _s=("size", "sum"))
    )
    sec_score["pct"] = (sec_score["_w"] / sec_score["_s"]).fillna(0.0)
    if not sec_score.empty:
        best_sec = sec_score["pct"].idxmax()
        best_sec_pct = sec_score["pct"].max()
        worst_sec = sec_score["pct"].idxmin()
        worst_sec_pct = sec_score["pct"].min()
    else:
        best_sec = worst_sec = ""
        best_sec_pct = worst_sec_pct = 0.0

    top_stock = df.sort_values(chg_col, ascending=False).iloc[0]
    bot_stock = df.sort_values(chg_col, ascending=True).iloc[0]

    label = "DoD" if view == "daily" else "WoW"
    cols = st.columns(4)
    cols[0].markdown(_pill_card(
        "Breadth",
        f"{advancers} ▲ &nbsp; {decliners} ▼",
        f"of {total} stocks",
        accent="#00d09c" if advancers >= decliners else "#eb5757",
    ), unsafe_allow_html=True)

    cols[1].markdown(_pill_card(
        f"Strongest sector ({label})",
        best_sec or "—",
        f"{best_sec_pct:+.1f}% mcap-weighted",
        accent="#00d09c" if best_sec_pct >= 0 else "#eb5757",
    ), unsafe_allow_html=True)

    cols[2].markdown(_pill_card(
        f"Weakest sector ({label})",
        worst_sec or "—",
        f"{worst_sec_pct:+.1f}% mcap-weighted",
        accent="#eb5757" if worst_sec_pct < 0 else "#00d09c",
    ), unsafe_allow_html=True)

    cols[3].markdown(_pill_card(
        f"Top mover ({label})",
        f"{top_stock['symbol']} &nbsp; {top_stock[chg_col]:+.1f}%",
        f"vs {bot_stock['symbol']} {bot_stock[chg_col]:+.1f}%",
        accent="#00d09c",
    ), unsafe_allow_html=True)


def _pill_card(title: str, headline: str, sub: str, accent: str = "#00d09c") -> str:
    return f"""
        <div style="background:#151922;border:1px solid #232834;border-radius:10px;
                    padding:12px 14px;">
          <div style="font-size:0.66rem;color:#7a8294;font-weight:500;
                      text-transform:uppercase;letter-spacing:0.06em;">{title}</div>
          <div style="font-size:1.05rem;font-weight:600;color:{accent};
                      margin-top:4px;">{headline}</div>
          <div style="font-size:0.72rem;color:#7a8294;margin-top:2px;">{sub}</div>
        </div>
    """


def _format_inr_crore(mcap: float) -> str:
    """Format a market cap (in INR) as crore / lakh-crore."""
    if not mcap:
        return "—"
    crore = mcap / 1e7
    if crore >= 1e5:
        return f"₹{crore / 1e5:,.2f} L Cr"
    return f"₹{crore:,.0f} Cr"
