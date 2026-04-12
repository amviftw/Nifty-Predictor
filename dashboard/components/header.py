"""Header component: market status, timestamp, F&O expiry."""

from datetime import date

import streamlit as st

from config.holidays import is_trading_day, prev_trading_day, days_to_expiry, next_fno_expiry
from dashboard.data_loader import MarketSnapshot


def render_header(snapshot: MarketSnapshot):
    """Render the dashboard header bar."""
    today = date.today()
    trading = is_trading_day(today)
    dte = days_to_expiry(today)
    expiry_date = next_fno_expiry(today)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if snapshot.is_market_open:
            st.markdown(":green[**MARKET OPEN**]")
        elif trading:
            st.markdown(":orange[**MARKET CLOSED** (after hours)]")
        else:
            st.markdown(":red[**MARKET CLOSED**]")

    with col2:
        if trading:
            st.caption(f"Data as of today ({today.strftime('%a, %b %d')})")
        else:
            ptd = prev_trading_day(today)
            st.caption(f"Data as of {ptd.strftime('%a, %b %d %Y')}")

    with col3:
        st.caption(f"F&O Expiry: {expiry_date.strftime('%b %d')} ({dte} trading days)")

    with col4:
        st.caption(f"Last refresh: {snapshot.timestamp.strftime('%I:%M %p IST')}")
