"""Macro panel: FII/DII flows, India VIX trend, USD/INR trend."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot


def render_macro_panel(snapshot: MarketSnapshot):
    """Render the macro indicators panel with charts."""
    st.subheader("Macro Indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**FII / DII Flows**")
        if snapshot.fii_dii_series:
            fii_df = pd.DataFrame(snapshot.fii_dii_series)
            if not fii_df.empty and "date" in fii_df.columns:
                # Take last 5 trading days
                fii_df = fii_df.tail(5)
                chart_df = fii_df.set_index("date")[["fii_net_buy", "dii_net_buy"]].copy()
                chart_df.columns = ["FII Net", "DII Net"]
                chart_df = chart_df.fillna(0)
                st.bar_chart(chart_df, height=250)

                # Summary metrics
                total_fii = chart_df["FII Net"].sum()
                total_dii = chart_df["DII Net"].sum()
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("FII (5D total)", f"{total_fii:,.0f}",
                              "Net Buyer" if total_fii > 0 else "Net Seller")
                with c2:
                    st.metric("DII (5D total)", f"{total_dii:,.0f}",
                              "Net Buyer" if total_dii > 0 else "Net Seller")
            else:
                st.caption("FII/DII data unavailable")
        else:
            st.caption("FII/DII data unavailable")

    with col2:
        st.markdown("**India VIX Trend**")
        if not snapshot.vix_series.empty:
            st.line_chart(snapshot.vix_series, height=250)
            # Context
            vix = snapshot.india_vix
            if vix < 13:
                st.caption("Low volatility — complacency zone")
            elif vix < 18:
                st.caption("Normal volatility range")
            elif vix < 25:
                st.caption("Elevated volatility — caution")
            else:
                st.caption("High volatility — fear/uncertainty")
        else:
            st.caption("VIX data unavailable")

    with col3:
        st.markdown("**USD/INR Trend**")
        if not snapshot.usdinr_series.empty:
            st.line_chart(snapshot.usdinr_series, height=250)
            # Context
            change = snapshot.usdinr_change
            if change > 0.3:
                st.caption("Rupee weakening sharply — negative for imports, FII sentiment")
            elif change > 0:
                st.caption("Rupee mildly weaker")
            elif change > -0.3:
                st.caption("Rupee mildly stronger")
            else:
                st.caption("Rupee strengthening — positive for FII flows")
        else:
            st.caption("USD/INR data unavailable")
