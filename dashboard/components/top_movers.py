"""Top movers component: gainers and losers tables."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot


def render_top_movers(snapshot: MarketSnapshot):
    """Render top gainers and losers in a two-column layout."""
    st.subheader("Top Movers")

    if not snapshot.top_gainers and not snapshot.top_losers:
        st.info("Stock data unavailable")
        return

    change_label = "DoD %" if snapshot.view == "daily" else "WoW %"
    change_key = "dod_pct" if snapshot.view == "daily" else "wow_pct"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**:green[Top Gainers]**")
        if snapshot.top_gainers:
            gdf = pd.DataFrame(snapshot.top_gainers)
            display_df = gdf[["symbol", "company", "sector", change_key, "close", "volume"]].copy()
            display_df.columns = ["Symbol", "Company", "Sector", change_label, "Close", "Volume"]
            st.dataframe(
                display_df,
                column_config={
                    change_label: st.column_config.NumberColumn(format="%.2f%%"),
                    "Close": st.column_config.NumberColumn(format="%.2f"),
                    "Volume": st.column_config.NumberColumn(format="%d"),
                },
                use_container_width=True,
                hide_index=True,
            )

    with col2:
        st.markdown("**:red[Top Losers]**")
        if snapshot.top_losers:
            ldf = pd.DataFrame(snapshot.top_losers)
            display_df = ldf[["symbol", "company", "sector", change_key, "close", "volume"]].copy()
            display_df.columns = ["Symbol", "Company", "Sector", change_label, "Close", "Volume"]
            st.dataframe(
                display_df,
                column_config={
                    change_label: st.column_config.NumberColumn(format="%.2f%%"),
                    "Close": st.column_config.NumberColumn(format="%.2f"),
                    "Volume": st.column_config.NumberColumn(format="%d"),
                },
                use_container_width=True,
                hide_index=True,
            )
