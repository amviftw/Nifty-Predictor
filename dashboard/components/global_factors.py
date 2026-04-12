"""Global indices and supply chain factors."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot
from dashboard.config import SECTOR_SUPPLY_CHAIN, SUPPLY_CHAIN_TICKERS


def render_global_indices(snapshot: MarketSnapshot):
    """Render global index metric cards."""
    st.subheader("Global Indices")

    if not snapshot.global_indices:
        st.info("Global index data unavailable")
        return

    indices = list(snapshot.global_indices.items())
    cols = st.columns(len(indices))

    for col, (name, data) in zip(cols, indices):
        with col:
            ret = data.get("ret_pct", 0)
            st.metric(name, f"{ret:+.2f}%", delta=None)


def render_supply_chain(snapshot: MarketSnapshot):
    """Render supply chain factors table with sector impact annotations."""
    st.subheader("Supply Chain & International Factors")

    if snapshot.supply_chain.empty:
        st.info("Supply chain data unavailable")
        return

    # Display the data table
    df = snapshot.supply_chain.copy()
    st.dataframe(
        df,
        column_config={
            "Factor": st.column_config.TextColumn("Factor", width="medium"),
            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
            "DoD %": st.column_config.NumberColumn("Day %", format="%.2f%%"),
            "WoW %": st.column_config.NumberColumn("Week %", format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Sector impact annotations
    st.markdown("**Sector Impact Analysis**")

    # Build a mapping from factor name to its current movement
    factor_moves = {}
    for _, row in df.iterrows():
        factor_name = row["Factor"]
        dod = row.get("DoD %", 0)
        factor_moves[factor_name] = dod

    # For each sector that has supply chain links, show the impact
    impacted_sectors = []
    for sector, info in SECTOR_SUPPLY_CHAIN.items():
        relevant_factors = []
        for f in info["factors"]:
            if f in factor_moves:
                move = factor_moves[f]
                if abs(move) > 0.5:  # Only show meaningful moves
                    direction = "up" if move > 0 else "down"
                    relevant_factors.append(f"{f} {move:+.1f}%")

        if relevant_factors:
            factors_str = ", ".join(relevant_factors)
            impacted_sectors.append({
                "Sector": sector,
                "Active Factors": factors_str,
                "Context": info["note"],
            })

    if impacted_sectors:
        impact_df = pd.DataFrame(impacted_sectors)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No significant supply chain movements today (>0.5% threshold)")
