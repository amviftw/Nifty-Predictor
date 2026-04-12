"""Sector deep dive: expandable per-sector breakdown."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot
from dashboard.config import SECTOR_INDEX_TO_SECTOR, SECTOR_SUPPLY_CHAIN


def render_sector_deep_dive(snapshot: MarketSnapshot):
    """Render expandable sector-by-sector breakdown."""
    st.subheader("Sector Deep Dive")

    if snapshot.sectoral_data.empty or snapshot.stock_changes.empty:
        st.info("Sector data unavailable")
        return

    change_col = "dod_pct" if snapshot.view == "daily" else "wow_pct"
    change_label = "DoD %" if snapshot.view == "daily" else "WoW %"

    # Sort sectors by performance
    sorted_sectors = snapshot.sectoral_data.sort_values(
        "DoD %" if snapshot.view == "daily" else "WoW %", ascending=False
    )

    for _, row in sorted_sectors.iterrows():
        index_name = row["Index"]
        sector_name = SECTOR_INDEX_TO_SECTOR.get(index_name, index_name)
        change_val = row["DoD %"] if snapshot.view == "daily" else row["WoW %"]

        # Color indicator
        indicator = ":green_circle:" if change_val > 0 else ":red_circle:" if change_val < 0 else ":white_circle:"

        with st.expander(f"{indicator} **{index_name}** — {change_val:+.2f}% | Close: {row['Close']:,.0f}"):
            c1, c2 = st.columns([2, 1])

            with c1:
                # Constituent stocks in this sector
                sector_stocks = snapshot.stock_changes[
                    snapshot.stock_changes["sector"] == sector_name
                ].sort_values(change_col, ascending=False)

                if not sector_stocks.empty:
                    display_df = sector_stocks[["symbol", "company", change_col, "close", "volume"]].copy()
                    display_df.columns = ["Symbol", "Company", change_label, "Close", "Volume"]
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
                else:
                    st.caption(f"No Nifty 50 stocks in '{sector_name}' sector")

            with c2:
                # Index performance summary
                st.markdown("**Index Performance**")
                st.metric("Day", f"{row['DoD %']:+.2f}%")
                st.metric("Week", f"{row['WoW %']:+.2f}%")
                st.metric("Month", f"{row['1M %']:+.2f}%")

                # Supply chain context
                sc_info = SECTOR_SUPPLY_CHAIN.get(sector_name)
                if sc_info:
                    st.markdown("**Key Drivers**")
                    st.caption(f"Linked to: {', '.join(sc_info['factors'])}")
                    st.caption(sc_info["note"])


def render_sector_rotation(snapshot: MarketSnapshot):
    """Render sector rotation table showing leadership over different timeframes."""
    st.subheader("Sector Rotation")

    if snapshot.sectoral_data.empty:
        st.info("Sector data unavailable")
        return

    df = snapshot.sectoral_data.copy()

    # Compute ranks (1 = best performer)
    df["Day Rank"] = df["DoD %"].rank(ascending=False).astype(int)
    df["Week Rank"] = df["WoW %"].rank(ascending=False).astype(int)
    df["Month Rank"] = df["1M %"].rank(ascending=False).astype(int)

    display_df = df[["Index", "DoD %", "Day Rank", "WoW %", "Week Rank", "1M %", "Month Rank"]]
    display_df = display_df.sort_values("Week Rank")

    st.dataframe(
        display_df,
        column_config={
            "DoD %": st.column_config.NumberColumn("Day %", format="%.2f%%"),
            "WoW %": st.column_config.NumberColumn("Week %", format="%.2f%%"),
            "1M %": st.column_config.NumberColumn("Month %", format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Highlight rotation patterns
    if len(df) >= 3:
        day_leaders = df.nsmallest(3, "Day Rank")["Index"].tolist()
        week_leaders = df.nsmallest(3, "Week Rank")["Index"].tolist()
        month_leaders = df.nsmallest(3, "Month Rank")["Index"].tolist()

        # Check for rotation (day leaders different from week/month leaders)
        day_set = set(day_leaders)
        week_set = set(week_leaders)
        month_set = set(month_leaders)

        if day_set != week_set:
            new_leaders = day_set - week_set
            if new_leaders:
                st.caption(f"Emerging leadership: {', '.join(new_leaders)} (leading today, not in weekly top 3)")

        fading = month_set - day_set
        if fading:
            st.caption(f"Fading momentum: {', '.join(fading)} (monthly leader but lagging today)")
