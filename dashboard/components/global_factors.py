"""Global indices and supply chain factors."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot
from dashboard.config import SECTOR_SUPPLY_CHAIN, SUPPLY_CHAIN_TICKERS


def render_global_indices(snapshot: MarketSnapshot):
    """Render global index metric cards."""
    st.markdown("#### Global Indices")

    if not snapshot.global_indices:
        st.info("Global index data unavailable")
        return

    indices = list(snapshot.global_indices.items())
    cols = st.columns(len(indices))

    for col, (name, data) in zip(cols, indices):
        with col:
            ret = data.get("ret_pct", 0)
            color = "green" if ret > 0 else "red" if ret < 0 else "grey"
            st.metric(name, f"{ret:+.2f}%")


def render_supply_chain(snapshot: MarketSnapshot):
    """Render supply chain factors with sector impact analysis."""
    st.markdown("#### Supply Chain & International Factors")

    if snapshot.supply_chain.empty:
        st.info("Supply chain data unavailable")
        return

    df = snapshot.supply_chain.copy()

    c1, c2 = st.columns([3, 2])

    with c1:
        st.dataframe(
            df,
            column_config={
                "Factor": st.column_config.TextColumn("Factor", width="medium"),
                "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                "DoD %": st.column_config.NumberColumn("Day", format="%.2f%%"),
                "WoW %": st.column_config.NumberColumn("Week", format="%.2f%%"),
            },
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        factor_moves = {}
        for _, row in df.iterrows():
            factor_moves[row["Factor"]] = row.get("DoD %", 0)

        impacted = []
        for sector, info in SECTOR_SUPPLY_CHAIN.items():
            active = []
            for f in info["factors"]:
                if f in factor_moves and abs(factor_moves[f]) > 0.5:
                    active.append(f"{f} {factor_moves[f]:+.1f}%")
            if active:
                impacted.append({
                    "Sector": sector,
                    "Movers": ", ".join(active),
                })

        if impacted:
            st.markdown("**Sector impact**")
            st.dataframe(pd.DataFrame(impacted), use_container_width=True, hide_index=True)
        else:
            st.caption("No significant supply chain moves today (>0.5%)")
