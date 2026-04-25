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

from dashboard.data_loader import load_market_snapshot, market_freshness_key
from dashboard.components.header import render_header
from dashboard.components.market_overview import render_key_metrics, render_sectoral_heatmap
from dashboard.components.top_movers import render_top_movers
from dashboard.components.macro_panel import render_macro_panel
from dashboard.components.global_factors import render_global_indices, render_supply_chain
from dashboard.components.sector_deep_dive import render_sector_deep_dive, render_sector_rotation
from dashboard.components.predictions_panel import render_predictions_panel
from dashboard.components.charts_view import render_charts_view


_TOP_BAR_CSS = """
<style>
/* Tight top-bar spacing */
.block-container { padding-top: 1.2rem; }

/* Compact, horizontal radios styled like segmented pills */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div { gap: 0.35rem; }
div[data-testid="stRadio"] label {
    background: #1e2130;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    cursor: pointer;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #1a237e;
    border-color: #3f51b5;
}
</style>
"""


def _render_top_bar() -> tuple[str, str]:
    """Render the top-of-page mode + view toggles and refresh button.

    Returns
    -------
    (mode, view_lower)
    """
    st.markdown(_TOP_BAR_CSS, unsafe_allow_html=True)

    left, center, right = st.columns([2.2, 2, 1])

    with left:
        mode = st.radio(
            "Mode",
            ["Market Overview", "Charts"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="app_mode",
        )

    with center:
        view = st.radio(
            "View",
            ["Daily", "Weekly"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="app_view",
            disabled=(mode == "Charts"),
        )

    with right:
        if st.button("↻ Refresh", use_container_width=True, key="app_refresh"):
            st.cache_data.clear()
            st.rerun()

    st.divider()
    return mode, view.lower()


def main():
    st.set_page_config(
        page_title="Nifty Market Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- Sidebar: kept for About / reference only ---
    with st.sidebar:
        st.title("Nifty Market Dashboard")
        st.caption(
            "Data sourced from Yahoo Finance, NSE India, and Google News. "
            "All sources are free and require no API keys. "
            "During NSE market hours data refreshes every minute; "
            "outside market hours it refreshes at the start of each new session. "
            "Use the ↻ Refresh button to force an immediate fetch."
        )

    # --- Top bar: mode / view / refresh ---
    mode, view_key = _render_top_bar()

    # --- Route to mode ---
    if mode == "Charts":
        render_charts_view()
        return

    _render_market_overview(view_key)


def _render_market_overview(view_key: str):
    """Render the original Market Overview dashboard."""
    snapshot = load_market_snapshot(view=view_key, _freshness=market_freshness_key())

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
