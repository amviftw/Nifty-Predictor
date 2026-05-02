"""Macro panel: FII/DII flows, India VIX trend, USD/INR trend.

The panel is framed as a *cascade of causality* — global yields and the dollar
move first, flows respond, volatility reprices, and finally sectors react.
The visuals follow the same left-to-right order so the reader's eye flows
with the story.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data_loader import MarketSnapshot


_MACRO_CSS = """
<style>
.cascade-rail {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    background: #0f131c;
    border: 1px solid #1f2633;
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.9rem;
}
.cascade-rail .step {
    background: #151922;
    border: 1px solid #232834;
    color: #c9cfd9;
    font-size: 0.78rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-weight: 500;
}
.cascade-rail .step b { color: #e8ecf1; font-weight: 600; }
.cascade-rail .arrow { color: #3b465d; font-size: 0.9rem; }
.cascade-rail .tag {
    background: rgba(0,208,156,0.12);
    color: #00d09c;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 700;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    margin-right: 0.4rem;
}
.flow-stat {
    background: #0f131c;
    border: 1px solid #1f2633;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.flow-stat .label {
    color: #7a8294;
    font-size: 0.64rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}
.flow-stat .value {
    color: #e8ecf1;
    font-size: 1.05rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    margin-top: 0.15rem;
}
.flow-stat .value.pos { color: #00d09c; }
.flow-stat .value.neg { color: #ef5350; }
.flow-stat .delta {
    color: #7a8294;
    font-size: 0.68rem;
    margin-top: 0.15rem;
}
</style>
"""


def render_macro_panel(snapshot: MarketSnapshot):
    """Render macro indicators with charts + causality framing."""
    st.markdown(_MACRO_CSS, unsafe_allow_html=True)

    st.markdown("#### Macro Indicators")

    # Causality rail — sets the story before we show numbers.
    st.markdown(
        "<div class='cascade-rail'>"
        "<span class='tag'>Read left → right</span>"
        "<span class='step'><b>Global yields & dollar</b> move first</span>"
        "<span class='arrow'>→</span>"
        "<span class='step'><b>FII / DII</b> flows respond</span>"
        "<span class='arrow'>→</span>"
        "<span class='step'><b>India VIX</b> reprices risk</span>"
        "<span class='arrow'>→</span>"
        "<span class='step'><b>Sectors</b> react"
        "</span></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        _render_fii_dii(snapshot)

    with c2:
        st.markdown("**India VIX**")
        if not snapshot.vix_series.empty:
            st.line_chart(snapshot.vix_series, height=200)
            _render_vix_stats(snapshot)
        else:
            st.caption("VIX data unavailable")

    with c3:
        st.markdown("**USD/INR**")
        if not snapshot.usdinr_series.empty:
            st.line_chart(snapshot.usdinr_series, height=200)
            _render_usdinr_stats(snapshot)
        else:
            st.caption("USD/INR data unavailable")


def _render_fii_dii(snapshot: MarketSnapshot):
    """FII/DII flows with WoW and MoM trajectory — the causality lens."""
    st.markdown("**FII / DII Flows**")

    if not snapshot.fii_dii_series:
        st.caption("FII/DII data unavailable")
        return

    fii_df = pd.DataFrame(snapshot.fii_dii_series)
    if fii_df.empty or "date" not in fii_df.columns:
        st.caption("FII/DII data unavailable")
        return

    fii_df = fii_df.sort_values("date").reset_index(drop=True)
    fii_df["fii_net_buy"] = fii_df["fii_net_buy"].fillna(0)
    fii_df["dii_net_buy"] = fii_df["dii_net_buy"].fillna(0)

    # Chart: last ~7 sessions for readability.
    chart_df = fii_df.tail(7).set_index("date")[["fii_net_buy", "dii_net_buy"]].copy()
    chart_df.columns = ["FII Net", "DII Net"]
    st.bar_chart(chart_df, height=200)

    # WoW (5d) and MoM (20d) aggregates for both FII and DII.
    fii_wow = float(fii_df["fii_net_buy"].tail(5).sum())
    dii_wow = float(fii_df["dii_net_buy"].tail(5).sum())
    fii_mom = float(fii_df["fii_net_buy"].tail(20).sum())
    dii_mom = float(fii_df["dii_net_buy"].tail(20).sum())

    st.markdown(
        "<div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;'>"
        f"{_flow_stat_html('FII · WoW', fii_wow)}"
        f"{_flow_stat_html('DII · WoW', dii_wow)}"
        f"{_flow_stat_html('FII · MoM', fii_mom)}"
        f"{_flow_stat_html('DII · MoM', dii_mom)}"
        "</div>",
        unsafe_allow_html=True,
    )

    # Narrative tag-line — tells the reader what the flows mean *now*.
    net_wow = fii_wow + dii_wow
    if fii_wow < 0 and dii_wow > 0:
        note = ":orange[DIIs absorbing FII selling — domestic floor holding]"
    elif fii_wow > 0 and dii_wow > 0:
        note = ":green[Both hands buying — risk-on setup]"
    elif fii_wow < 0 and dii_wow < 0:
        note = ":red[Both hands selling — broad risk-off]"
    elif net_wow > 0:
        note = ":green[Net inflow this week]"
    else:
        note = ":orange[Net outflow this week]"
    st.caption(note)


def _flow_stat_html(label: str, value: float) -> str:
    cls = "pos" if value > 0 else "neg" if value < 0 else ""
    sign = "+" if value > 0 else ""
    return (
        f"<div class='flow-stat'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value {cls}'>{sign}{value:,.0f}</div>"
        f"<div class='delta'>₹ cr</div>"
        f"</div>"
    )


def _render_vix_stats(snapshot: MarketSnapshot):
    vix = snapshot.india_vix
    series = snapshot.vix_series.iloc[:, 0] if not snapshot.vix_series.empty else None

    wow_txt, mom_txt = "", ""
    if series is not None and len(series) >= 2:
        from dashboard.data_loader import _week_pct_change
        wow_pct = _week_pct_change(series, dates=series.index)
        if wow_pct or len(series) >= 6:
            wow_txt = f"WoW {wow_pct:+.1f}%"
    if series is not None and len(series) >= 21:
        month_ago = float(series.iloc[-21])
        if month_ago:
            mom_txt = f"MoM {((vix - month_ago) / month_ago) * 100:+.1f}%"

    if vix < 13:
        regime = ":green[Low volatility — complacency]"
    elif vix < 18:
        regime = ":blue[Normal range]"
    elif vix < 25:
        regime = ":orange[Elevated — caution]"
    else:
        regime = ":red[High volatility — fear]"

    trail = " · ".join(x for x in [wow_txt, mom_txt] if x)
    st.caption(f"{regime}" + (f"  ·  {trail}" if trail else ""))


def _render_usdinr_stats(snapshot: MarketSnapshot):
    usd = snapshot.usdinr
    series = snapshot.usdinr_series.iloc[:, 0] if not snapshot.usdinr_series.empty else None

    wow_txt, mom_txt = "", ""
    if series is not None and len(series) >= 2:
        from dashboard.data_loader import _week_pct_change
        wow_pct = _week_pct_change(series, dates=series.index)
        if wow_pct or len(series) >= 6:
            wow_txt = f"WoW {wow_pct:+.1f}%"
    if series is not None and len(series) >= 21:
        month_ago = float(series.iloc[-21])
        if month_ago:
            mom_txt = f"MoM {((usd - month_ago) / month_ago) * 100:+.1f}%"

    change = snapshot.usdinr_change
    if change > 0.3:
        regime = ":red[Rupee weakening sharply — FII risk]"
    elif change > 0:
        regime = ":orange[Mildly weaker]"
    elif change > -0.3:
        regime = ":blue[Mildly stronger]"
    else:
        regime = ":green[Rupee strengthening — positive for flows]"

    trail = " · ".join(x for x in [wow_txt, mom_txt] if x)
    st.caption(f"{regime}" + (f"  ·  {trail}" if trail else ""))
