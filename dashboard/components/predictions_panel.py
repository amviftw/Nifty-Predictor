"""ML predictions panel: shows today's BUY/SELL/HOLD signals if available."""

import os
import sys
from datetime import date

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.holidays import prev_trading_day, is_trading_day
from config.nifty50_tickers import NIFTY50_STOCKS


def render_predictions_panel():
    """Render ML predictions from the existing system, if available."""
    st.subheader("ML Predictions")

    try:
        from data.storage.db_manager import DBManager
        db = DBManager()

        # Try today, then previous trading day
        today = date.today()
        check_date = today if is_trading_day(today) else prev_trading_day(today)
        predictions = db.get_predictions(check_date.isoformat())

        if not predictions:
            # Try one more day back
            predictions = db.get_predictions(prev_trading_day(check_date).isoformat())

        if not predictions:
            st.info(
                "No ML predictions available. Run `python scripts/daily_predict.py` "
                "to generate predictions."
            )
            return

        pred_date = predictions[0].get("date", "")
        st.caption(f"Predictions for {pred_date}")

        # Separate by signal type
        buys = [p for p in predictions if p.get("signal") == "BUY"]
        sells = [p for p in predictions if p.get("signal") == "SELL"]
        holds = [p for p in predictions if p.get("signal") == "HOLD"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BUY signals", len(buys))
        with col2:
            st.metric("SELL signals", len(sells))
        with col3:
            st.metric("HOLD signals", len(holds))

        # Show BUY and SELL tables
        if buys:
            st.markdown("**:green[BUY Signals]**")
            _render_signal_table(buys)

        if sells:
            st.markdown("**:red[SELL Signals]**")
            _render_signal_table(sells)

    except Exception as e:
        st.info(f"ML predictions unavailable: {e}")


def _render_signal_table(signals: list[dict]):
    """Render a table of prediction signals."""
    records = []
    for s in signals:
        symbol = s.get("symbol", "")
        company = NIFTY50_STOCKS.get(symbol, (None, symbol, "Unknown"))[1]
        records.append({
            "Symbol": symbol,
            "Company": company,
            "Confidence": round((s.get("confidence") or 0) * 100, 1),
            "P(Up)": round((s.get("prob_up") or 0) * 100, 1),
            "P(Down)": round((s.get("prob_down") or 0) * 100, 1),
            "P(Flat)": round((s.get("prob_flat") or 0) * 100, 1),
        })

    df = pd.DataFrame(records)
    st.dataframe(
        df,
        column_config={
            "Confidence": st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100),
            "P(Up)": st.column_config.NumberColumn(format="%.1f%%"),
            "P(Down)": st.column_config.NumberColumn(format="%.1f%%"),
            "P(Flat)": st.column_config.NumberColumn(format="%.1f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )
