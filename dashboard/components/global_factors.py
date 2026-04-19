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
            st.metric(name, f"{ret:+.2f}%")


def render_supply_chain(snapshot: MarketSnapshot):
    """Render supply chain factors with sector impact analysis — stacked, full-width."""
    st.markdown("#### Supply Chain & International Factors")

    if snapshot.supply_chain.empty:
        st.info("Supply chain data unavailable")
        return

    df = snapshot.supply_chain.copy()

    # Full-width factor table with centered numeric columns
    st.dataframe(
        df,
        column_config={
            "Factor": st.column_config.TextColumn("Factor", width="medium"),
            "Price": st.column_config.NumberColumn("Price", format="%.2f", width="small"),
            "DoD %": st.column_config.NumberColumn("Day %", format="%.2f%%", width="small"),
            "WoW %": st.column_config.NumberColumn("Week %", format="%.2f%%", width="small"),
        },
        use_container_width=True,
        hide_index=True,
        height=min(38 + 35 * len(df), 420),
    )

    # Sector impact — only show sectors with active movements
    factor_moves = {row["Factor"]: row.get("DoD %", 0) for _, row in df.iterrows()}

    impacted = []
    for sector, info in SECTOR_SUPPLY_CHAIN.items():
        active = [
            f"{f} {factor_moves[f]:+.1f}%"
            for f in info["factors"]
            if f in factor_moves and abs(factor_moves[f]) > 0.5
        ]
        if active:
            impacted.append({
                "Sector": sector,
                "Active Factors": "  ·  ".join(active),
                "Context": info["note"],
            })

    if impacted:
        st.markdown(
            '<div style="margin-top:24px; margin-bottom:8px; font-size:0.85rem; '
            'color:#c9cfd9; font-weight:600;">Sector Impact</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{len(impacted)} sectors with supply-chain factors moving >0.5% today")
        impact_df = pd.DataFrame(impacted)
        st.dataframe(
            impact_df,
            column_config={
                "Sector": st.column_config.TextColumn("Sector", width="small"),
                "Active Factors": st.column_config.TextColumn("Active Factors", width="medium"),
                "Context": st.column_config.TextColumn("Context", width="large"),
            },
            use_container_width=True,
            hide_index=True,
            height=min(38 + 35 * len(impacted), 420),
        )
    else:
        st.caption("No significant supply chain moves today (>0.5%)")
