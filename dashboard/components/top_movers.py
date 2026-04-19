"""Top movers component: gainers and losers in a two-column tabbed layout."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot


def render_top_movers(snapshot: MarketSnapshot):
    """Render top gainers and losers."""
    st.markdown("#### Top Movers")

    if not snapshot.top_gainers and not snapshot.top_losers:
        st.info("Stock data unavailable")
        return

    change_label = "DoD %" if snapshot.view == "daily" else "WoW %"
    change_key = "dod_pct" if snapshot.view == "daily" else "wow_pct"

    col_config = {
        change_label: st.column_config.NumberColumn(format="%.2f%%"),
        "Close": st.column_config.NumberColumn(format="%.2f"),
        "Volume": st.column_config.NumberColumn(format="%d"),
    }

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**:green[Gainers]**")
        if snapshot.top_gainers:
            gdf = pd.DataFrame(snapshot.top_gainers)
            display = gdf[["symbol", "company", "sector", change_key, "close", "volume"]].copy()
            display.columns = ["Symbol", "Company", "Sector", change_label, "Close", "Volume"]
            st.dataframe(display, column_config=col_config,
                         use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**:red[Losers]**")
        if snapshot.top_losers:
            ldf = pd.DataFrame(snapshot.top_losers)
            display = ldf[["symbol", "company", "sector", change_key, "close", "volume"]].copy()
            display.columns = ["Symbol", "Company", "Sector", change_label, "Close", "Volume"]
            st.dataframe(display, column_config=col_config,
                         use_container_width=True, hide_index=True)
