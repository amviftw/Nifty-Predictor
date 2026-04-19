"""
Indian Stock Market Dashboard — Main Streamlit Application.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import os

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
from dashboard.components.target_hunter import render_target_hunter
from dashboard.components.ema_chart import render_ema_chart
from dashboard.components.sector_momentum import render_sector_momentum


_CUSTOM_CSS = """
<style>
/* Tighter metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2130 0%, #262a3d 100%);
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
[data-testid="stMetric"] label { font-size: 0.78rem; color: #aaa; }
[data-testid="stMetricValue"] { font-size: 1.4rem; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161825;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1a237e !important;
}

/* Expander polish */
.streamlit-expanderHeader { font-size: 0.92rem; }

/* Better dataframe density */
[data-testid="stDataFrame"] { font-size: 0.85rem; }

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #131620 100%);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem;
}

/* Section header styling */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 1px solid #333;
}
</style>
"""


def main():
    st.set_page_config(
        page_title="Nifty Market Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.title("Nifty Market Dashboard")
        st.caption("Live Indian equity market fundamentals")

        st.divider()

        mode = st.radio(
            "Mode",
            ["Market Overview", "Target Hunter"],
            index=0,
            help=(
                "Market Overview: live Nifty 50 + sector + macro snapshot. "
                "Target Hunter: find stocks with high analyst upside + technical confirmation."
            ),
        )

        view = st.radio(
            "View",
            ["Daily", "Weekly"],
            index=0,
            help="Daily: Day-over-day changes. Weekly: Week-over-week changes.",
            disabled=(mode == "Target Hunter"),
        )

        st.divider()

        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        st.caption(
            "Data: Yahoo Finance, NSE India, Google News. "
            "Free — no API keys. Auto-refreshes every 5 min."
        )

    # --- Target Hunter mode ---
    if mode == "Target Hunter":
        snapshot = load_market_snapshot(view="daily")
        render_header(snapshot)
        st.divider()
        render_target_hunter()
        return

    # --- Market Overview mode ---
    view_key = view.lower()
    snapshot = load_market_snapshot(view=view_key)

    render_header(snapshot)

    # Key metrics bar (always visible)
    render_key_metrics(snapshot)

    st.divider()

    # Main content in tabs
    tab_market, tab_sectors, tab_macro, tab_signals = st.tabs([
        "Market",
        "Sectors",
        "Macro & Global",
        "Signals",
    ])

    with tab_market:
        # EMA chart
        render_ema_chart(view=view_key)

        st.divider()

        # Top movers
        render_top_movers(snapshot)

    with tab_sectors:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            render_sectoral_heatmap(snapshot)

        with col_right:
            render_sector_rotation(snapshot)

        st.divider()

        # Momentum grid (weekly data — always relevant)
        render_sector_momentum()

        st.divider()

        render_sector_deep_dive(snapshot)

    with tab_macro:
        render_macro_panel(snapshot)

        st.divider()

        render_global_indices(snapshot)

        st.divider()

        render_supply_chain(snapshot)

    with tab_signals:
        render_predictions_panel()


if __name__ == "__main__":
    main()
