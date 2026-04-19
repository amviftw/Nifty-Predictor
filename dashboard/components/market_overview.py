"""Market overview: Nifty 50 KPIs and sectoral indices heatmap."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot


def render_key_metrics(snapshot: MarketSnapshot):
    """Render the headline KPI metric cards."""
    cols = st.columns(5)

    with cols[0]:
        nifty_delta = snapshot.nifty50_change_pct if snapshot.view == "daily" else snapshot.nifty50_wow_pct
        label = "Nifty 50 (DoD)" if snapshot.view == "daily" else "Nifty 50 (WoW)"
        st.metric(label, f"{snapshot.nifty50_close:,.0f}", f"{nifty_delta:+.2f}%")

    with cols[1]:
        st.metric("India VIX", f"{snapshot.india_vix:.1f}", f"{snapshot.india_vix_change:+.2f}%",
                  delta_color="inverse")

    with cols[2]:
        st.metric("USD/INR", f"{snapshot.usdinr:.2f}", f"{snapshot.usdinr_change:+.2f}%",
                  delta_color="inverse")

    with cols[3]:
        fii_label = "FII Net (Cr)"
        st.metric(fii_label, f"{snapshot.fii_net_buy:,.0f}",
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
    """Render sectoral indices sorted by performance with bar chart."""
    st.markdown("#### Sectoral Indices")

    if snapshot.sectoral_data.empty:
        st.info("Sectoral index data unavailable")
        return

    df = snapshot.sectoral_data.copy()
    sort_col = "DoD %" if snapshot.view == "daily" else "WoW %"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    st.dataframe(
        df,
        column_config={
            "Index": st.column_config.TextColumn("Sector", width="medium"),
            "Close": st.column_config.NumberColumn("Close", format="%.0f", width="small"),
            "DoD %": st.column_config.NumberColumn("Day %", format="%.2f%%", width="small"),
            "WoW %": st.column_config.NumberColumn("Week %", format="%.2f%%", width="small"),
            "1M %": st.column_config.NumberColumn("Month %", format="%.2f%%", width="small"),
        },
        use_container_width=True,
        hide_index=True,
        height=min(38 + 35 * len(df), 500),
    )

    if not df.empty:
        chart_data = df.set_index("Index")[[sort_col]].sort_values(sort_col)
        st.bar_chart(chart_data, horizontal=True, height=max(250, len(df) * 30))
