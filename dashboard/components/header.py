"""Header component: market status bar — Groww-style compact strip."""

from datetime import date

import streamlit as st

from config.holidays import is_trading_day, prev_trading_day, days_to_expiry, next_fno_expiry
from dashboard.data_loader import MarketSnapshot


def render_header(snapshot: MarketSnapshot):
    """Render a compact status bar at the top of the dashboard."""
    today = date.today()
    trading = is_trading_day(today)
    dte = days_to_expiry(today)
    expiry_date = next_fno_expiry(today)

    if snapshot.is_market_open:
        status_html = '<span style="color:#00d09c; font-weight:600;">● LIVE</span>'
        status_label = "Market open"
    elif trading:
        status_html = '<span style="color:#f5a623; font-weight:600;">● CLOSED</span>'
        status_label = "After hours"
    else:
        status_html = '<span style="color:#eb5757; font-weight:600;">● CLOSED</span>'
        status_label = "Non-trading day"

    if trading:
        data_date = today.strftime("%a, %b %d")
    else:
        data_date = prev_trading_day(today).strftime("%a, %b %d")

    bar = f"""
    <div style="
        display:flex; align-items:center; gap:24px;
        padding:10px 16px; margin-bottom:16px;
        background:#151922; border:1px solid #232834; border-radius:10px;
        font-size:0.82rem; color:#c9cfd9;
    ">
      <div style="display:flex; align-items:center; gap:8px;">
        {status_html}
        <span style="color:#7a8294;">{status_label}</span>
      </div>
      <div style="color:#3a4050;">│</div>
      <div>
        <span style="color:#7a8294; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">Data</span>
        <span style="margin-left:8px; color:#e8ecf1; font-weight:500;">{data_date}</span>
      </div>
      <div style="color:#3a4050;">│</div>
      <div>
        <span style="color:#7a8294; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">F&amp;O Expiry</span>
        <span style="margin-left:8px; color:#e8ecf1; font-weight:500;">{expiry_date.strftime('%b %d')}</span>
        <span style="color:#7a8294;"> · {dte}d</span>
      </div>
      <div style="margin-left:auto; color:#7a8294; font-size:0.78rem;">
        Updated {snapshot.timestamp.strftime('%I:%M %p IST')}
      </div>
    </div>
    """
    st.markdown(bar, unsafe_allow_html=True)
