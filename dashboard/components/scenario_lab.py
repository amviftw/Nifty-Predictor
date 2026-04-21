"""
Scenario Lab — Bloomberg-terminal thinking, Varsity-style, for retail traders.

Three tabs:

1. Learn             — chapter cards teaching each macro driver.
2. Cause-Effect Graph — Driver -> Sector -> Stock Sankey, plus per-driver
                        top-impact rankings.
3. Scenario Builder  — user dials in a set of driver shocks, gets a ranked
                        list of trade ideas grounded in historical base rates.
4. Hypothesis Lab    — user writes a conditional rule and backtests it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from scenarios.drivers import DRIVERS, driver_ids, price_based_drivers
from scenarios.sensitivity import build_sensitivity_matrix, SensitivityMatrix
from scenarios.engine import run_scenario, TradeIdea, ScenarioResult
from scenarios.validator import validate_hypothesis, Condition
from config.nifty50_tickers import NIFTY50_STOCKS


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

_LAB_CSS = """
<style>
.driver-card {
    background: linear-gradient(145deg, #1a1d2a 0%, #14161e 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.6rem;
    height: 100%;
}
.driver-card h4 {
    margin: 0 0 0.3rem 0;
    color: #e0e3ec;
    font-size: 1.02rem;
}
.driver-card .cat {
    display: inline-block;
    background: rgba(63, 81, 181, 0.18);
    color: #8ea7ff;
    font-size: 0.68rem;
    padding: 0.12rem 0.55rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.driver-card .what {
    color: #a6adbb;
    font-size: 0.85rem;
    margin-bottom: 0.6rem;
    line-height: 1.4;
}
.driver-card .why {
    color: #d7dce4;
    font-size: 0.82rem;
    line-height: 1.45;
    border-left: 2px solid #3f51b5;
    padding-left: 0.7rem;
    margin-bottom: 0.7rem;
}
.driver-card .affected {
    font-size: 0.78rem;
    color: #97a0ae;
    margin-bottom: 0.3rem;
}
.driver-card .affected .plus { color: #26a69a; font-weight: 600; }
.driver-card .affected .minus { color: #ef5350; font-weight: 600; }
.driver-card .example {
    background: rgba(255, 193, 7, 0.07);
    border-left: 2px solid #ffa726;
    padding: 0.55rem 0.7rem;
    border-radius: 0 6px 6px 0;
    color: #e0c890;
    font-size: 0.78rem;
    margin-top: 0.5rem;
    line-height: 1.45;
}
.idea-card {
    background: #14161e;
    border: 1px solid rgba(255,255,255,0.06);
    border-left-width: 3px;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.55rem;
}
.idea-card.long  { border-left-color: #26a69a; }
.idea-card.short { border-left-color: #ef5350; }
.idea-card .ticker-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.3rem;
}
.idea-card .ticker {
    font-size: 1rem;
    font-weight: 700;
    color: #e0e3ec;
}
.idea-card .move {
    font-size: 0.95rem;
    font-weight: 700;
}
.idea-card .move.up  { color: #26a69a; }
.idea-card .move.down { color: #ef5350; }
.idea-card .meta {
    color: #8a92a0;
    font-size: 0.76rem;
    margin-bottom: 0.35rem;
}
.idea-card .hypothesis {
    color: #c8cdd6;
    font-size: 0.81rem;
    line-height: 1.45;
}
.idea-card .conf-HIGH   { background: rgba(38,166,154,0.18); color: #66d3c4; padding: 1px 7px; border-radius: 8px; font-size: 0.68rem; }
.idea-card .conf-MEDIUM { background: rgba(255,167,38,0.18); color: #ffcf87; padding: 1px 7px; border-radius: 8px; font-size: 0.68rem; }
.idea-card .conf-LOW    { background: rgba(120,144,156,0.18); color: #a6b0bf; padding: 1px 7px; border-radius: 8px; font-size: 0.68rem; }
.lab-banner {
    background: linear-gradient(90deg, rgba(26,35,126,0.25), rgba(63,81,181,0.10));
    border-left: 3px solid #3f51b5;
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
    color: #cbd2dd;
    font-size: 0.86rem;
    line-height: 1.5;
}
</style>
"""


# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_matrix(lookback_days: int) -> SensitivityMatrix:
    """Cache the sensitivity matrix for the app lifetime (per-lookback)."""
    return build_sensitivity_matrix(lookback_days=lookback_days)


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def render_scenario_lab():
    """Top-level render for the Scenario Lab mode."""
    st.markdown(_LAB_CSS, unsafe_allow_html=True)
    st.title("Scenario Lab")
    st.caption(
        "Bloomberg-terminal thinking, Varsity-style. "
        "Learn what moves your stocks, build scenarios, and validate your hypotheses "
        "against history before you risk capital."
    )

    # --- Matrix controls (shared) ---
    left, mid, right = st.columns([2, 1, 1])
    with left:
        st.markdown(
            "<div class='lab-banner'>"
            "🎓 <b>How to use this.</b> Start at <i>Learn</i> to meet the drivers. "
            "Move to <i>Cause-Effect Graph</i> to see who reacts to what, and by how much. "
            "Then open <i>Scenario Builder</i> to simulate a morning scenario and get trade ideas "
            "grounded in historical base rates. Use <i>Hypothesis Lab</i> to stress-test your own beliefs."
            "</div>",
            unsafe_allow_html=True,
        )
    with mid:
        lookback = st.selectbox(
            "History window",
            options=[250, 500, 1000, 1500],
            format_func=lambda d: f"{d} trading days (~{d//252}y)",
            index=1,
            key="lab_lookback",
            help="How far back to compute betas and hit-rates.",
        )
    with right:
        if st.button("↻ Recompute matrix", use_container_width=True, key="lab_refresh"):
            st.cache_resource.clear()
            st.rerun()

    with st.spinner(f"Loading {lookback}-day history for 50 stocks × {len(price_based_drivers())} drivers…"):
        matrix = _load_matrix(int(lookback))

    if matrix.betas.empty or matrix.betas.isna().all().all():
        st.error(
            "Could not fetch enough historical data from Yahoo Finance. "
            "Check your internet connection and try again."
        )
        return

    tabs = st.tabs(["📚 Learn", "🧬 Cause-Effect Graph", "🧪 Scenario Builder", "🔬 Hypothesis Lab"])

    with tabs[0]:
        _render_learn_tab(matrix)
    with tabs[1]:
        _render_graph_tab(matrix)
    with tabs[2]:
        _render_builder_tab(matrix)
    with tabs[3]:
        _render_hypothesis_tab()


# -----------------------------------------------------------------------------
# Tab 1: Learn
# -----------------------------------------------------------------------------

def _render_learn_tab(matrix: SensitivityMatrix):
    st.markdown(
        "### The Drivers — your vocabulary of the market\n"
        "Every scenario you build, and every trade you take, is a bet on *something moving*. "
        "These eight drivers explain most of the daily action in Nifty 50. Understand them, and you "
        "stop chasing prices — you start reading pressure."
    )

    cards = list(DRIVERS.values())
    cols_per_row = 2
    for i in range(0, len(cards), cols_per_row):
        row = st.columns(cols_per_row)
        for j, driver in enumerate(cards[i:i + cols_per_row]):
            with row[j]:
                helps = ", ".join(driver.who_positive) if driver.who_positive else "—"
                hurts = ", ".join(driver.who_negative) if driver.who_negative else "—"
                st.markdown(
                    f"""
                    <div class='driver-card'>
                        <span class='cat'>{driver.category}</span>
                        <h4>{driver.name}</h4>
                        <div class='what'><b>What it is.</b> {driver.what}</div>
                        <div class='why'><b>Why it matters.</b> {driver.why}</div>
                        <div class='affected'>
                            <span class='plus'>↑ helps:</span> {helps}<br/>
                            <span class='minus'>↓ hurts:</span> {hurts}
                        </div>
                        <div class='example'>📌 {driver.example}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Small historical signature: the top-3 stocks most sensitive to this driver.
                if driver.id in matrix.betas.index and not matrix.betas.loc[driver.id].dropna().empty:
                    top_pos = matrix.top_for_driver(driver.id, side="positive", n=3)
                    top_neg = matrix.top_for_driver(driver.id, side="negative", n=3)
                    with st.expander(f"Historical signature ({matrix.lookback_days}d)", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Moves most with the driver**")
                            st.dataframe(
                                top_pos.assign(
                                    beta=lambda d: d["beta"].round(2),
                                    up_hit_rate=lambda d: (d["up_hit_rate"] * 100).round(0),
                                )[["symbol", "sector", "beta", "up_hit_rate"]].rename(
                                    columns={"up_hit_rate": "hit %"}
                                ),
                                hide_index=True, use_container_width=True, height=150,
                            )
                        with c2:
                            st.markdown("**Moves against the driver**")
                            st.dataframe(
                                top_neg.assign(
                                    beta=lambda d: d["beta"].round(2),
                                    down_hit_rate=lambda d: (d["down_hit_rate"] * 100).round(0),
                                )[["symbol", "sector", "beta", "down_hit_rate"]].rename(
                                    columns={"down_hit_rate": "hit %"}
                                ),
                                hide_index=True, use_container_width=True, height=150,
                            )


# -----------------------------------------------------------------------------
# Tab 2: Cause-Effect Graph
# -----------------------------------------------------------------------------

def _render_graph_tab(matrix: SensitivityMatrix):
    st.markdown(
        "### Cause → Effect — who reacts to what\n"
        "Below is a historical map of Driver → Sector → Stock relationships. "
        "Edge thickness is the absolute sensitivity (beta). Green edges are positive, red are inverse."
    )

    driver_choices = [d_id for d_id in driver_ids() if d_id in matrix.betas.index]
    if not driver_choices:
        st.info("Sensitivity matrix is empty for price-based drivers.")
        return

    selected = st.multiselect(
        "Drivers to map",
        options=driver_choices,
        default=driver_choices[:4],
        format_func=lambda d: DRIVERS[d].name,
        key="graph_driver_select",
    )
    if not selected:
        st.info("Pick at least one driver to draw the graph.")
        return

    fig = _build_sankey(matrix, selected)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.divider()
    st.markdown("### Per-driver sector beta")
    sector_betas_cols = st.columns(min(len(selected), 3))
    for i, d_id in enumerate(selected):
        with sector_betas_cols[i % len(sector_betas_cols)]:
            s_beta = matrix.sector_beta(d_id)
            if s_beta.empty:
                continue
            colors = ["#26a69a" if v >= 0 else "#ef5350" for v in s_beta.values]
            bar = go.Figure(
                go.Bar(
                    x=s_beta.values,
                    y=s_beta.index,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.2f}" for v in s_beta.values],
                    textposition="outside",
                )
            )
            bar.update_layout(
                title=f"{DRIVERS[d_id].name}",
                template="plotly_dark",
                height=max(260, 30 * len(s_beta)),
                margin=dict(l=0, r=20, t=40, b=10),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                yaxis=dict(autorange="reversed"),
                xaxis_title="Avg beta",
            )
            st.plotly_chart(bar, use_container_width=True, config={"displaylogo": False})


def _build_sankey(matrix: SensitivityMatrix, driver_list: list[str]) -> go.Figure:
    """Three-level Sankey: Driver -> Sector -> Stock."""
    sectors_in_universe = sorted({info[2] for info in NIFTY50_STOCKS.values()})

    driver_labels = [DRIVERS[d].name for d in driver_list]
    sector_labels = sectors_in_universe

    # Gather a stock shortlist: per driver top 4 positive + top 4 negative.
    shortlisted_stocks: set[str] = set()
    for d in driver_list:
        pos = matrix.top_for_driver(d, "positive", n=4)["symbol"].tolist()
        neg = matrix.top_for_driver(d, "negative", n=4)["symbol"].tolist()
        shortlisted_stocks.update(pos + neg)

    stock_labels = sorted(shortlisted_stocks)

    labels = driver_labels + sector_labels + stock_labels
    driver_idx = {DRIVERS[d].name: i for i, d in enumerate(driver_list)}
    sector_idx = {s: len(driver_labels) + i for i, s in enumerate(sector_labels)}
    stock_idx = {s: len(driver_labels) + len(sector_labels) + i for i, s in enumerate(stock_labels)}

    # node colours by level
    node_colors = (
        ["#3f51b5"] * len(driver_labels)
        + ["#00838f"] * len(sector_labels)
        + ["#546e7a"] * len(stock_labels)
    )

    sources, targets, values, link_colors, link_labels = [], [], [], [], []

    # Driver -> Sector
    for d in driver_list:
        s_betas = matrix.sector_beta(d)
        for sector, beta in s_betas.items():
            if sector not in sector_idx or not np.isfinite(beta) or abs(beta) < 0.02:
                continue
            sources.append(driver_idx[DRIVERS[d].name])
            targets.append(sector_idx[sector])
            values.append(float(abs(beta) * 100))
            link_colors.append("rgba(38,166,154,0.35)" if beta >= 0 else "rgba(239,83,80,0.35)")
            link_labels.append(f"β = {beta:+.2f}")

    # Sector -> Stock (only for shortlisted stocks, edge value = cross-sector avg |beta|)
    for sym in stock_labels:
        sector = NIFTY50_STOCKS[sym][2]
        if sector not in sector_idx:
            continue
        # Sum absolute betas across the selected drivers as an "influence strength".
        betas = [matrix.betas.loc[d, sym] for d in driver_list if d in matrix.betas.index and sym in matrix.betas.columns]
        betas = [b for b in betas if np.isfinite(b)]
        if not betas:
            continue
        strength = float(np.mean([abs(b) for b in betas]))
        if strength < 0.02:
            continue
        sources.append(sector_idx[sector])
        targets.append(stock_idx[sym])
        values.append(strength * 100)
        avg_signed = float(np.mean(betas))
        link_colors.append("rgba(38,166,154,0.35)" if avg_signed >= 0 else "rgba(239,83,80,0.35)")
        link_labels.append(f"avg |β| = {strength:.2f}")

    fig = go.Figure(go.Sankey(
        node=dict(
            label=labels,
            color=node_colors,
            pad=14,
            thickness=16,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            label=link_labels,
        ),
        arrangement="snap",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=max(560, 18 * len(stock_labels) + 120),
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#0e1117",
        font=dict(color="#d7d7d7", size=12),
    )
    return fig


# -----------------------------------------------------------------------------
# Tab 3: Scenario Builder
# -----------------------------------------------------------------------------

def _render_builder_tab(matrix: SensitivityMatrix):
    st.markdown(
        "### Scenario Builder — *\"what if tomorrow morning…\"*\n"
        "Move the sliders to set the overnight / intraday shocks you expect, "
        "then hit **Simulate**. You will see the estimated impact on Nifty 50, "
        "each sector, and the top ranked trade ideas with historical base rates."
    )

    form_col, result_col = st.columns([1, 2])

    with form_col:
        st.markdown("#### Driver shocks")
        shocks: dict[str, float] = {}
        for d_id in driver_ids():
            driver = DRIVERS[d_id]
            shocks[d_id] = st.slider(
                f"{driver.name} ({driver.unit})",
                min_value=float(driver.shock_range[0]),
                max_value=float(driver.shock_range[1]),
                value=0.0,
                step=0.1 if driver.unit == "%" else 500.0 if d_id == "fii_flow" else 0.05,
                key=f"shock_{d_id}",
                help=driver.what,
            )

        run = st.button("▶ Simulate scenario", type="primary", use_container_width=True)
        preset_col1, preset_col2 = st.columns(2)
        with preset_col1:
            if st.button("🛢 Oil Spike", use_container_width=True):
                _apply_preset({"crude": 4.0, "vix": 8.0, "usdinr": 0.4})
                st.rerun()
            if st.button("💵 USD Strength", use_container_width=True):
                _apply_preset({"dxy": 1.0, "usdinr": 0.8, "us10y": 0.1})
                st.rerun()
        with preset_col2:
            if st.button("⚠ Risk-Off", use_container_width=True):
                _apply_preset({"sp500": -1.5, "vix": 20.0, "dxy": 0.6, "fii_flow": -3000.0})
                st.rerun()
            if st.button("🌱 Risk-On", use_container_width=True):
                _apply_preset({"sp500": 1.5, "vix": -15.0, "copper": 2.0, "fii_flow": 2000.0})
                st.rerun()

    with result_col:
        if not run and not any(v != 0 for v in shocks.values()):
            st.info("Move a slider or pick a preset on the left, then press **Simulate**.")
            return

        result = run_scenario(shocks, matrix)
        _render_scenario_result(result)


def _apply_preset(preset: dict[str, float]):
    for k, v in preset.items():
        st.session_state[f"shock_{k}"] = v


def _render_scenario_result(result: ScenarioResult):
    # Headline metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Est. Nifty 50 impact",
            f"{result.nifty_expected_pct:+.2f}%",
            help="Equal-weight mean of expected stock moves under your shocks."
        )
    with m2:
        top_sector = result.sector_expected_pct.idxmax() if not result.sector_expected_pct.empty else "—"
        top_val = result.sector_expected_pct.max() if not result.sector_expected_pct.empty else 0
        st.metric("Best sector", top_sector, f"{top_val:+.2f}%")
    with m3:
        worst_sector = result.sector_expected_pct.idxmin() if not result.sector_expected_pct.empty else "—"
        worst_val = result.sector_expected_pct.min() if not result.sector_expected_pct.empty else 0
        st.metric("Worst sector", worst_sector, f"{worst_val:+.2f}%")

    # Sector strip
    if not result.sector_expected_pct.empty:
        sec_df = result.sector_expected_pct.reset_index()
        sec_df.columns = ["sector", "expected_pct"]
        fig = px.bar(
            sec_df,
            x="expected_pct", y="sector",
            orientation="h",
            color="expected_pct",
            color_continuous_scale="RdYlGn",
            range_color=[-max(0.1, abs(sec_df["expected_pct"]).max()), max(0.1, abs(sec_df["expected_pct"]).max())],
            text=sec_df["expected_pct"].map(lambda x: f"{x:+.2f}%"),
        )
        fig.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            xaxis_title="Expected move (%)",
            yaxis=dict(autorange="reversed", title=""),
            coloraxis_showscale=False,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Trade ideas
    st.markdown("#### 🎯 Top trade ideas")
    long_col, short_col = st.columns(2)
    with long_col:
        st.markdown("**LONG ideas**")
        if not result.long_ideas:
            st.caption("No compelling long setups under this scenario.")
        for idea in result.long_ideas:
            _render_idea_card(idea)
    with short_col:
        st.markdown("**SHORT ideas**")
        if not result.short_ideas:
            st.caption("No compelling short setups under this scenario.")
        for idea in result.short_ideas:
            _render_idea_card(idea)

    # Full per-stock table behind an expander
    with st.expander("Full per-stock impact table"):
        table = result.per_stock.sort_values("expected_pct", ascending=False).copy()
        table["expected_pct"] = table["expected_pct"].round(2)
        st.dataframe(
            table.rename(columns={
                "symbol": "Symbol", "company": "Company",
                "sector": "Sector", "expected_pct": "Est. move %",
            }),
            hide_index=True, use_container_width=True, height=360,
        )


def _render_idea_card(idea: TradeIdea):
    klass = "long" if idea.direction == "LONG" else "short"
    move_klass = "up" if idea.expected_move_pct >= 0 else "down"
    base_rate_txt = (
        f"base rate <b>{idea.base_rate*100:.0f}%</b> (n={idea.n_analogs})"
        if not np.isnan(idea.base_rate)
        else f"base rate unavailable (n={idea.n_analogs})"
    )
    st.markdown(
        f"""
        <div class='idea-card {klass}'>
            <div class='ticker-row'>
                <span class='ticker'>{idea.direction} · {idea.symbol}</span>
                <span class='move {move_klass}'>{idea.expected_move_pct:+.2f}%</span>
            </div>
            <div class='meta'>
                {idea.company} · {idea.sector} ·
                <span class='conf-{idea.confidence}'>{idea.confidence}</span> · {base_rate_txt}
            </div>
            <div class='hypothesis'>{idea.hypothesis}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Tab 4: Hypothesis Lab
# -----------------------------------------------------------------------------

def _render_hypothesis_tab():
    st.markdown(
        "### Hypothesis Lab — test your own belief\n"
        "Describe a conditional rule you think is true. We'll find every day "
        "in recent history where your conditions fired and show you how often "
        "the trade actually worked."
    )

    st.markdown("#### Conditions (all must fire)")
    usable_drivers = price_based_drivers()
    driver_names = [DRIVERS[d].name for d in usable_drivers]
    driver_by_name = {DRIVERS[d].name: d for d in usable_drivers}

    # Fixed 3 condition rows for simplicity.
    conditions: list[Condition] = []
    for i in range(3):
        c1, c2, c3, c4 = st.columns([2.4, 0.8, 1, 1])
        with c1:
            drv_label = st.selectbox(
                f"Driver #{i+1}",
                ["— none —"] + driver_names,
                index=0 if i > 0 else 1,
                key=f"hyp_drv_{i}",
            )
        with c2:
            op = st.selectbox(
                "Op",
                [">", "<", ">=", "<="],
                index=1 if i == 0 else 0,
                key=f"hyp_op_{i}",
                label_visibility="visible",
            )
        with c3:
            thr = st.number_input(
                "Threshold %",
                value=-2.0 if i == 0 else (5.0 if i == 1 else 0.0),
                step=0.5,
                key=f"hyp_thr_{i}",
            )
        with c4:
            if drv_label != "— none —":
                st.caption(DRIVERS[driver_by_name[drv_label]].unit)

        if drv_label != "— none —":
            conditions.append(Condition(driver_id=driver_by_name[drv_label], op=op, threshold_pct=thr))

    st.markdown("#### Then trade…")
    c5, c6, c7, c8 = st.columns([2, 1, 1, 1])
    with c5:
        symbols = list(NIFTY50_STOCKS.keys())
        target = st.selectbox("Stock", symbols, index=symbols.index("HDFCBANK"), key="hyp_target")
    with c6:
        direction = st.selectbox("Direction", ["LONG", "SHORT"], key="hyp_dir")
    with c7:
        horizon = st.selectbox("Horizon", [1, 2, 3, 5], index=0, key="hyp_horizon")
    with c8:
        lookback = st.selectbox("Lookback", [500, 750, 1000, 1500], index=1, key="hyp_lookback")

    if st.button("🔬 Validate hypothesis", type="primary"):
        if not conditions:
            st.warning("Add at least one condition.")
            return

        with st.spinner("Searching history for analog days…"):
            result = validate_hypothesis(
                conditions=conditions,
                target_symbol=target,
                direction=direction,
                horizon_days=int(horizon),
                lookback_days=int(lookback),
            )

        st.caption(f"**Hypothesis:** {result.hypothesis}")

        if result.n_trigger_days == 0:
            st.warning(
                f"No days in the last {result.lookback_days} trading days matched all your conditions. "
                "Loosen the thresholds or shorten your condition list."
            )
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Hit rate",
            f"{result.hit_rate*100:.0f}%",
            help="Share of trigger days on which the trade direction made money over your horizon.",
        )
        m2.metric("Avg return", f"{result.avg_return_pct:+.2f}%")
        m3.metric("Median return", f"{result.median_return_pct:+.2f}%")
        m4.metric("Sample size", f"{result.n_trigger_days} days", f"of {result.n_total_days} total")

        # Verdict line
        if np.isnan(result.hit_rate):
            verdict = "Not enough data."
        elif result.hit_rate >= 0.6 and result.avg_return_pct > 0:
            verdict = "✅ Hypothesis holds up on history. Small sample — treat as a lean, not proof."
        elif result.hit_rate <= 0.4 and result.avg_return_pct < 0:
            verdict = "❌ History says the opposite is true. Consider flipping direction."
        else:
            verdict = "🟡 Mixed. Edge is not reliable — don't trade it blind."
        st.markdown(f"**Verdict.** {verdict}")

        # Return distribution
        if not result.returns.empty:
            fig = px.histogram(
                result.returns.rename("Return %"),
                nbins=25,
                title=f"Distribution of {direction} {target} returns on {result.n_trigger_days} trigger days",
            )
            fig.update_layout(
                template="plotly_dark",
                height=320,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
                xaxis_title="Return %",
                yaxis_title="Days",
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#8a92a0")
            fig.add_vline(x=result.avg_return_pct, line_color="#26a69a", annotation_text="avg")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
