"""Market overview: Nifty 50 KPIs and sectoral indices heatmap."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot


def render_key_metrics(snapshot: MarketSnapshot):
    """Render the 4 headline KPI metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        nifty_delta = snapshot.nifty50_change_pct if snapshot.view == "daily" else snapshot.nifty50_wow_pct
        label = "Nifty 50 (DoD)" if snapshot.view == "daily" else "Nifty 50 (WoW)"
        st.metric(label, f"{snapshot.nifty50_close:,.0f}", f"{nifty_delta:+.2f}%")

    with col2:
        st.metric("India VIX", f"{snapshot.india_vix:.1f}", f"{snapshot.india_vix_change:+.2f}%",
                   delta_color="inverse")  # Higher VIX = bad

    with col3:
        st.metric("USD/INR", f"{snapshot.usdinr:.2f}", f"{snapshot.usdinr_change:+.2f}%",
                   delta_color="inverse")  # Weaker rupee = bad

    with col4:
        fii_label = "FII Net" + (" (Cr)" if abs(snapshot.fii_net_buy) < 1e6 else "")
        st.metric(fii_label, f"{snapshot.fii_net_buy:,.0f}",
                   "Buying" if snapshot.fii_net_buy > 0 else "Selling")

    with col5:
        st.metric(
            "Advance / Decline",
            f"{snapshot.advance_count} / {snapshot.decline_count}",
            f"{snapshot.unchanged_count} unchanged",
            delta_color="off",
        )


def render_sectoral_heatmap(snapshot: MarketSnapshot):
    """Render the sectoral indices table sorted by performance."""
    st.subheader("Sectoral Indices")

    if snapshot.sectoral_data.empty:
        st.info("Sectoral index data unavailable")
        return

    df = snapshot.sectoral_data.copy()

    # Sort by the primary change column based on view
    sort_col = "DoD %" if snapshot.view == "daily" else "WoW %"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    st.dataframe(
        df,
        column_config={
            "Index": st.column_config.TextColumn("Sector Index", width="medium"),
            "Close": st.column_config.NumberColumn("Close", format="%.2f"),
            "DoD %": st.column_config.NumberColumn("Day %", format="%.2f%%"),
            "WoW %": st.column_config.NumberColumn("Week %", format="%.2f%%"),
            "1M %": st.column_config.NumberColumn("Month %", format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Horizontal bar chart for quick visual
    if not df.empty:
        chart_data = df.set_index("Index")[[sort_col]].sort_values(sort_col)
        st.bar_chart(chart_data, horizontal=True, height=max(250, len(df) * 30))
