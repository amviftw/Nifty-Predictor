"""Macro panel: FII/DII flows, India VIX trend, USD/INR trend."""

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot


def render_macro_panel(snapshot: MarketSnapshot):
    """Render macro indicators with charts."""
    st.markdown("#### Macro Indicators")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**FII / DII Flows**")
        if snapshot.fii_dii_series:
            fii_df = pd.DataFrame(snapshot.fii_dii_series)
            if not fii_df.empty and "date" in fii_df.columns:
                fii_df = fii_df.tail(5)
                chart_df = fii_df.set_index("date")[["fii_net_buy", "dii_net_buy"]].copy()
                chart_df.columns = ["FII Net", "DII Net"]
                chart_df = chart_df.fillna(0)
                st.bar_chart(chart_df, height=240)

                total_fii = chart_df["FII Net"].sum()
                total_dii = chart_df["DII Net"].sum()
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("FII 5D", f"{total_fii:,.0f}",
                              "Buyer" if total_fii > 0 else "Seller")
                with m2:
                    st.metric("DII 5D", f"{total_dii:,.0f}",
                              "Buyer" if total_dii > 0 else "Seller")
            else:
                st.caption("FII/DII data unavailable")
        else:
            st.caption("FII/DII data unavailable")

    with c2:
        st.markdown("**India VIX**")
        if not snapshot.vix_series.empty:
            st.line_chart(snapshot.vix_series, height=240)
            vix = snapshot.india_vix
            if vix < 13:
                st.caption(":green[Low volatility — complacency]")
            elif vix < 18:
                st.caption(":blue[Normal range]")
            elif vix < 25:
                st.caption(":orange[Elevated — caution]")
            else:
                st.caption(":red[High volatility — fear]")
        else:
            st.caption("VIX data unavailable")

    with c3:
        st.markdown("**USD/INR**")
        if not snapshot.usdinr_series.empty:
            st.line_chart(snapshot.usdinr_series, height=240)
            change = snapshot.usdinr_change
            if change > 0.3:
                st.caption(":red[Rupee weakening sharply — FII risk]")
            elif change > 0:
                st.caption(":orange[Mildly weaker]")
            elif change > -0.3:
                st.caption(":blue[Mildly stronger]")
            else:
                st.caption(":green[Rupee strengthening — positive for flows]")
        else:
            st.caption("USD/INR data unavailable")
