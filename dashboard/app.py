"""
Indian Stock Market Dashboard — Main Streamlit Application.

Run with:
    streamlit run dashboard/app.py

Provides Daily and Weekly views of:
- Nifty 50 and sectoral index movements
- Top gainers and losers
- Macro indicators (VIX, USD/INR, FII/DII flows)
- Global indices
- Supply chain / international factors with sector impact analysis
- Sector-by-sector deep dive
- ML predictions (if available from the existing prediction system)
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from dashboard.data_loader import load_market_snapshot
from dashboard.components.header import render_header
from dashboard.components.market_overview import render_key_metrics, render_sectoral_heatmap
from dashboard.components.top_movers import render_top_movers
from dashboard.components.macro_panel import render_macro_panel
from dashboard.components.global_factors import render_global_indices, render_supply_chain
from dashboard.components.sector_deep_dive import render_sector_deep_dive, render_sector_rotation
from dashboard.components.predictions_panel import render_predictions_panel
from dashboard.components.charts_view import render_charts_view


def main():
    st.set_page_config(
        page_title="Nifty Market Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Sidebar ---
    with st.sidebar:
        st.title("Nifty Market Dashboard")
        st.markdown("Live Indian equity market fundamentals")

        st.divider()

        mode = st.radio(
            "Mode",
            ["Market Overview", "Charts"],
            index=0,
            help=(
                "Market Overview: macro, sectoral and stock-level fundamentals. "
                "Charts: TradingView-style candles with indicators."
            ),
        )

        view = st.radio(
            "View",
            ["Daily", "Weekly"],
            index=0,
            help="Daily: Day-over-day changes. Weekly: Week-over-week changes.",
            disabled=(mode == "Charts"),
        )

        st.divider()

        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        st.markdown("**About**")
        st.caption(
            "Data sourced from Yahoo Finance, NSE India, and Google News. "
            "All sources are free and require no API keys. "
            "Data refreshes automatically every 5 minutes."
        )

    # --- Route to mode ---
    if mode == "Charts":
        render_charts_view()
        return

    _render_market_overview(view.lower())


def _render_market_overview(view_key: str):
    """Render the original Market Overview dashboard."""
    snapshot = load_market_snapshot(view=view_key)

    st.title("Indian Equity Market Dashboard")
    render_header(snapshot)

    st.divider()
    render_key_metrics(snapshot)

    st.divider()
    render_sectoral_heatmap(snapshot)

    st.divider()
    render_top_movers(snapshot)

    st.divider()
    render_macro_panel(snapshot)

    st.divider()
    render_global_indices(snapshot)

    st.divider()
    render_supply_chain(snapshot)

    st.divider()
    render_sector_rotation(snapshot)

    st.divider()
    render_sector_deep_dive(snapshot)

    st.divider()
    render_predictions_panel()


if __name__ == "__main__":
    main()
