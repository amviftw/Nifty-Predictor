"""Market overview: Nifty 50 KPIs and sectoral chip cards."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot
from dashboard.config import SECTOR_INDEX_TO_SECTOR


def render_key_metrics(snapshot: MarketSnapshot):
    """Render the headline KPI metric cards."""
    cols = st.columns(5)

    with cols[0]:
        nifty_delta = snapshot.nifty50_change_pct if snapshot.view == "daily" else snapshot.nifty50_wow_pct
        label = "Nifty 50 (DoD)" if snapshot.view == "daily" else "Nifty 50 (WoW)"
        st.metric(label, f"{snapshot.nifty50_close:,.0f}", f"{nifty_delta:+.1f}%")

    with cols[1]:
        st.metric("India VIX", f"{snapshot.india_vix:.1f}", f"{snapshot.india_vix_change:+.1f}%",
                  delta_color="inverse")

    with cols[2]:
        st.metric("USD/INR", f"{snapshot.usdinr:.2f}", f"{snapshot.usdinr_change:+.1f}%",
                  delta_color="inverse")

    with cols[3]:
        st.metric("FII Net (Cr)", f"{snapshot.fii_net_buy:,.0f}",
                  "Buying" if snapshot.fii_net_buy > 0 else "Selling")

    with cols[4]:
        total = snapshot.advance_count + snapshot.decline_count + snapshot.unchanged_count
        if total > 0:
            adv_pct = snapshot.advance_count / total * 100
            breadth_label = f"{adv_pct:.0f}% advancing"
        else:
            breadth_label = ""
        st.metric(
            "Advance / Decline",
            f"{snapshot.advance_count} / {snapshot.decline_count}",
            breadth_label,
            delta_color="off",
        )


def render_sectoral_heatmap(snapshot: MarketSnapshot):
    """Render sectoral indices as colored chip cards with per-sector context."""
    st.markdown("#### Sectors")

    if snapshot.sectoral_data.empty:
        st.info("Sectoral index data unavailable")
        return

    df = snapshot.sectoral_data.copy()
    sort_col = "DoD %" if snapshot.view == "daily" else "WoW %"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Per-sector stock-level stats
    sector_stats = _compute_sector_stats(snapshot)

    change_label = "Day" if snapshot.view == "daily" else "Week"
    st.markdown(
        f'<div style="font-size:0.72rem;color:#7a8294;margin-bottom:6px;">'
        f'Sorted by {change_label} change &middot; '
        f'Green = positive momentum &middot; '
        f'<span style="color:#00d09c;">&#9650;</span> advancing / '
        f'<span style="color:#eb5757;">&#9660;</span> declining stocks in sector</div>',
        unsafe_allow_html=True,
    )

    chips = []
    for _, row in df.iterrows():
        idx_name = row["Index"]
        change = row[sort_col]
        close = row["Close"]
        wow = row["WoW %"]
        mom = row["1M %"]
        stats = sector_stats.get(idx_name, {})

        # Gradient + border based on performance intensity
        if change > 1.5:
            bg = "linear-gradient(135deg, #0a2e1c 0%, #151922 100%)"
            border = "rgba(0,208,156,0.4)"
        elif change > 0:
            bg = "linear-gradient(135deg, #0f2018 0%, #151922 100%)"
            border = "rgba(0,208,156,0.2)"
        elif change > -1.5:
            bg = "linear-gradient(135deg, #2a1515 0%, #151922 100%)"
            border = "rgba(235,87,87,0.2)"
        else:
            bg = "linear-gradient(135deg, #3a1515 0%, #151922 100%)"
            border = "rgba(235,87,87,0.4)"

        chg_color = "#00d09c" if change > 0 else "#eb5757" if change < 0 else "#c9cfd9"
        wow_color = "#00d09c" if wow > 0 else "#eb5757" if wow < 0 else "#c9cfd9"
        mom_color = "#00d09c" if mom > 0 else "#eb5757" if mom < 0 else "#c9cfd9"

        # Advance / decline row
        ad_html = ""
        if stats:
            top_sym = stats["top_symbol"]
            top_chg = stats["top_change"]
            top_color = "#00d09c" if top_chg > 0 else "#eb5757"
            ad_html = (
                f'<div style="margin-top:8px;font-size:0.66rem;color:#7a8294;">'
                f'<span style="color:#00d09c;">&#9650;{stats["adv"]}</span>'
                f' &middot; <span style="color:#eb5757;">&#9660;{stats["dec"]}</span>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'<span style="color:#e8ecf1;">{top_sym}</span> '
                f'<span style="color:{top_color};">{top_chg:+.1f}%</span>'
                f'</div>'
            )

        # Momentum verdict
        verdict = _sector_verdict(change, wow, mom)
        verdict_html = ""
        if verdict:
            verdict_html = (
                f'<div style="margin-top:7px;padding-top:6px;border-top:1px solid #232834;'
                f'font-size:0.64rem;color:#7a8294;line-height:1.3;">{verdict}</div>'
            )

        chip = (
            f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
            f'padding:14px 16px;min-width:210px;flex:1 1 210px;max-width:280px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-weight:700;font-size:0.88rem;color:#e8ecf1;">{idx_name}</span>'
            f'<span style="color:{chg_color};font-weight:600;font-size:0.88rem;">{change:+.1f}%</span>'
            f'</div>'
            f'<div style="font-size:1.0rem;font-weight:500;color:#c9cfd9;margin-top:4px;">'
            f'{close:,.0f}</div>'
            f'<div style="display:flex;gap:18px;margin-top:10px;font-size:0.72rem;">'
            f'<div><span style="color:#7a8294;font-size:0.62rem;text-transform:uppercase;'
            f'letter-spacing:0.04em;">Week</span>'
            f'<br><span style="color:{wow_color};font-weight:500;">{wow:+.1f}%</span></div>'
            f'<div><span style="color:#7a8294;font-size:0.62rem;text-transform:uppercase;'
            f'letter-spacing:0.04em;">Month</span>'
            f'<br><span style="color:{mom_color};font-weight:500;">{mom:+.1f}%</span></div>'
            f'</div>'
            f'{ad_html}{verdict_html}'
            f'</div>'
        )
        chips.append(chip)

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def _compute_sector_stats(snapshot: MarketSnapshot) -> dict:
    """Advance/decline counts and top mover per sector index."""
    if snapshot.stock_changes.empty:
        return {}

    change_col = "dod_pct" if snapshot.view == "daily" else "wow_pct"
    result = {}

    for idx_name, sector_name in SECTOR_INDEX_TO_SECTOR.items():
        stocks = snapshot.stock_changes[snapshot.stock_changes["sector"] == sector_name]
        if stocks.empty:
            continue
        adv = int((stocks[change_col] > 0.1).sum())
        dec = int((stocks[change_col] < -0.1).sum())
        top = stocks.sort_values(change_col, ascending=False).iloc[0]
        result[idx_name] = {
            "adv": adv,
            "dec": dec,
            "top_symbol": top["symbol"],
            "top_change": top[change_col],
        }
    return result


def _sector_verdict(day_chg, wow, mom):
    """One-line momentum verdict for a sector chip."""
    if wow > 3 and mom > 5:
        return "Strong multi-week rally — momentum intact"
    if wow > 2 and day_chg > 1:
        return "Accelerating momentum this week"
    if mom > 5 and day_chg < -1:
        return "Monthly gainer pulling back — dip opportunity?"
    if wow < -3 and mom < -3:
        return "Sustained weakness — avoid or wait for reversal"
    if wow < -2 and day_chg > 1:
        return "Bouncing from weekly low — watch for follow-through"
    if mom > 3 and wow < 0:
        return "Monthly uptrend, short-term pause"
    if mom < -3 and wow > 0:
        return "Short-term bounce in a downtrend"
    return ""
