#!/usr/bin/env python3
"""
MAIN ENTRY POINT: Daily prediction script.
Run at 9:05 AM IST (T-10 min before market open) to get trading signals.

Usage:
    python -m scripts.daily_predict
    python -m scripts.daily_predict --email          # Send results via email
    python -m scripts.daily_predict --skip-refresh   # Skip data fetch, use cached data
    python -m scripts.daily_predict --date 2026-02-28 # Predict for a specific date
"""

import sys
import time
import traceback
import argparse
from pathlib import Path
from datetime import date, datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import SETTINGS
from config.holidays import is_nse_holiday
from config.nifty50_tickers import get_symbols, get_sector, get_company_name
from data.storage.db_manager import DBManager
from data.pipeline import DailyPipeline
from features.feature_engineer import FeatureEngineer
from models.ensemble import EnsemblePredictor
from signals.generator import SignalGenerator, rank_and_select_signals
from signals.risk_manager import RiskManager
from output.console_reporter import print_daily_report
from output.file_reporter import write_signals_csv, write_signals_json
from models.evaluator import evaluate_signals_backtest


def main():
    parser = argparse.ArgumentParser(description="Daily stock prediction")
    parser.add_argument(
        "--skip-refresh", action="store_true",
        help="Skip data refresh, use cached data",
    )
    parser.add_argument(
        "--date", default=None,
        help="Predict for a specific date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-output-file", action="store_true",
        help="Skip writing output files",
    )
    parser.add_argument(
        "--email", action="store_true",
        help="Send results via email notification",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Configure logging
    logger.add(
        SETTINGS.LOGS_DIR / "daily_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
    )

    target_date = args.date or date.today().isoformat()
    target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()

    logger.info("=" * 60)
    logger.info(f"NIFTY 50 PREDICTOR - Daily Run for {target_date}")
    logger.info("=" * 60)

    try:
        _run_prediction(args, target_date, target_date_obj)
    except Exception as e:
        logger.error(f"Daily prediction failed: {e}")
        logger.error(traceback.format_exc())
        if args.email:
            from output.email_notifier import send_error_email
            send_error_email("daily_predict", traceback.format_exc())

    elapsed = time.time() - start_time
    logger.info(f"Daily prediction completed in {elapsed:.1f} seconds")


def _run_prediction(args, target_date: str, target_date_obj):
    """Core prediction logic, separated for error handling."""

    # ==================== STEP 1: PRE-FLIGHT CHECKS ====================
    logger.info("[Step 1/6] Pre-flight checks...")

    if is_nse_holiday(target_date_obj):
        logger.info(f"Market closed on {target_date} (holiday/weekend). Exiting.")
        print(f"\nMarket closed on {target_date}. No predictions generated.")
        return

    # Check model files exist
    model_files = ["lgb_model.joblib", "xgb_model.joblib", "rf_model.joblib",
                   "ensemble_weights.joblib", "feature_list.joblib"]
    missing = [f for f in model_files if not (SETTINGS.MODELS_DIR / f).exists()]
    if missing:
        logger.error(f"Missing model files: {missing}")
        print(f"\nERROR: Model files not found. Run 'python -m scripts.train_models' first.")
        return

    db = DBManager(SETTINGS.DB_PATH)

    # ==================== STEP 2: DATA REFRESH ====================
    if not args.skip_refresh:
        logger.info("[Step 2/6] Refreshing data...")
        pipeline = DailyPipeline(SETTINGS.DB_PATH)
        pipeline.run()
    else:
        logger.info("[Step 2/6] Skipping data refresh (--skip-refresh)")

    # ==================== STEP 3: FEATURE COMPUTATION ====================
    logger.info("[Step 3/6] Computing features...")
    engineer = FeatureEngineer(db)
    features_df = engineer.compute_features_for_date(target_date)

    if features_df.empty:
        logger.error("No features computed. Check data availability.")
        print("\nERROR: Could not compute features. Is data available?")
        return

    # Prepare feature matrix
    symbols = features_df["symbol"].tolist()
    meta_cols = ["symbol", "date"]

    # Load model to get expected features
    ensemble = EnsemblePredictor.load()
    expected_features = ensemble.feature_names

    # Align features with what the model expects
    X = pd.DataFrame(0.0, index=features_df.index, columns=expected_features)
    for col in expected_features:
        if col in features_df.columns:
            X[col] = features_df[col].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Feature matrix: {X.shape[0]} stocks x {X.shape[1]} features")

    # ==================== STEP 4: MODEL INFERENCE ====================
    logger.info("[Step 4/6] Running model inference...")
    probas = ensemble.predict_proba(X)
    logger.info(f"Predictions generated for {len(symbols)} stocks")

    # ==================== STEP 5: SIGNAL GENERATION ====================
    logger.info("[Step 5/6] Generating signals...")

    # Build sector mapping
    sectors = {sym: get_sector(sym) for sym in symbols}

    # Generate signals
    sig_gen = SignalGenerator()
    all_signals = sig_gen.generate_all_signals(symbols, probas, sectors)

    # Select top signals
    selected = rank_and_select_signals(all_signals)

    # Apply risk management
    risk_mgr = RiskManager()
    selected = risk_mgr.apply_risk_constraints(selected)

    # Mark remaining stocks as HOLD in the full list
    selected_symbols = {s["symbol"] for s in selected}
    hold_signals = [s for s in all_signals if s["symbol"] not in selected_symbols]

    # Enrich signals with key driver reasons from computed features
    features_by_symbol = features_df.set_index("symbol")
    for sig in selected:
        sym = sig["symbol"]
        if sym in features_by_symbol.index:
            sig["reasons"] = _build_signal_reasons(
                sig["signal"],
                features_by_symbol.loc[sym],
                sig.get("sector", ""),
            )

    # Combine: actionable first, then holds
    final_signals = selected + hold_signals

    # ==================== STEP 6: OUTPUT ====================
    logger.info("[Step 6/6] Generating output...")

    # Get macro context for display
    macro_records = db.get_macro(start_date=target_date, end_date=target_date)
    macro_context = macro_records[0] if macro_records else None

    # Get yesterday's performance
    yesterday_perf = None
    try:
        prev_preds = db.get_previous_predictions(n_days=3)
        if prev_preds:
            yesterday_perf = evaluate_signals_backtest(pd.DataFrame(prev_preds))
    except Exception as e:
        logger.warning(f"Could not compute yesterday's performance: {e}")

    # Get model info
    model_info = {
        "version": "v1.0",
        "retrained": _get_model_date(),
    }

    # Console output
    print_daily_report(
        final_signals,
        macro_context=macro_context,
        yesterday_perf=yesterday_perf,
        model_info=model_info,
    )

    # File output
    if not args.no_output_file:
        write_signals_csv(final_signals, target_date)
        write_signals_json(final_signals, target_date)

    # Email notification
    if args.email:
        from output.email_notifier import send_daily_signals_email
        send_daily_signals_email(final_signals, macro_context, yesterday_perf)

    # Save predictions to DB
    prediction_records = []
    for sig in final_signals:
        prediction_records.append({
            "date": target_date,
            "symbol": sig["symbol"],
            "prob_up": sig.get("prob_up", 0),
            "prob_down": sig.get("prob_down", 0),
            "prob_flat": sig.get("prob_flat", 0),
            "signal": sig["signal"],
            "confidence": sig.get("confidence", 0),
            "position_size": sig.get("position_size_pct", 0),
        })
    db.insert_predictions(prediction_records)

    print(f"\nCompleted successfully.")


_SECTOR_CONTEXT = {
    "IT": {
        "BUY": (
            "IT sector positioned to benefit from the global AI infrastructure buildout — "
            "Indian tech majors are winning large multi-year deals as enterprises modernise "
            "legacy stacks. Rupee depreciation historically amplifies dollar-revenue margins."
        ),
        "SELL": (
            "IT sector facing near-term pressure from client spending caution and project "
            "ramp-down risks. A strengthening rupee could compress USD-denominated margins "
            "in the current quarter."
        ),
    },
    "Banking": {
        "BUY": (
            "Banking sector entering a rate-easing cycle that typically expands NIMs on "
            "the floating-rate book. Credit growth remains robust, driven by retail, MSME, "
            "and infrastructure lending — a combination that has historically re-rated PSU "
            "and private banks alike."
        ),
        "SELL": (
            "Banking sector showing signs of NIM compression as deposit repricing catches up. "
            "Slowing credit growth and rising NPA concerns in unsecured lending segments "
            "could weigh on near-term earnings."
        ),
    },
    "Pharma": {
        "BUY": (
            "Pharma sector benefits from a steady pipeline of US FDA approvals and "
            "accelerating domestic formulations demand. India's API export story remains "
            "structurally intact as global supply chains diversify away from China — "
            "earnings visibility over the next 2–3 years is high."
        ),
        "SELL": (
            "Pharma facing pricing pressure in the US generics market and heightened FDA "
            "scrutiny. Domestic price controls and input cost inflation are compressing "
            "margins for several large-cap names."
        ),
    },
    "Auto": {
        "BUY": (
            "Auto sector riding a confluence of rural demand recovery (backed by good monsoon "
            "and farm income), a premiumisation wave in passenger vehicles, and early EV "
            "adoption tailwinds. The sector has compounded earnings at double-digit rates "
            "for the past three years."
        ),
        "SELL": (
            "Auto sector vulnerable to inventory build-up at the dealer level and softening "
            "urban demand. Rising input costs (steel, aluminium) and potential fuel price "
            "hikes could squeeze margins over the near term."
        ),
    },
    "FMCG": {
        "BUY": (
            "FMCG sector benefiting from rural volume recovery after two subdued years — "
            "companies with high rural exposure are now reporting sequential volume acceleration. "
            "Commodity cost tailwinds are adding a margin buffer that the market is yet to "
            "fully price in."
        ),
        "SELL": (
            "FMCG faces a difficult near-term setup — urban consumption is softening and "
            "rural recovery, while real, is slow. Premium valuations leave little room for "
            "disappointment if volume growth undershoots estimates."
        ),
    },
    "Metals": {
        "BUY": (
            "Metals sector supported by China's stimulus-driven infrastructure push and "
            "India's own capex supercycle — domestic steel and aluminium consumption has "
            "been growing at 8–10% annually over the last three years. Global supply "
            "discipline is keeping prices supported."
        ),
        "SELL": (
            "Metals under pressure from China demand uncertainty and rising domestic "
            "overcapacity risks. Falling commodity prices globally are squeezing realisation "
            "per tonne, directly impacting EBITDA per unit."
        ),
    },
    "Infrastructure": {
        "BUY": (
            "Infrastructure sector is the direct beneficiary of India's record capital "
            "expenditure push — the government's ₹11L Cr infra budget is flowing into "
            "roads, ports, and logistics. Order book visibility for L&T-type companies "
            "extends 3–4 years out."
        ),
        "SELL": (
            "Infrastructure execution pace has slowed due to election cycles and state-level "
            "spending cuts. Order inflows are decelerating, and stretched working capital "
            "cycles are a growing risk for the sector."
        ),
    },
    "Oil & Gas": {
        "BUY": (
            "Oil & Gas sector positioned for margin expansion — refinery spreads remain "
            "healthy and upstream E&P companies benefit from elevated crude prices. "
            "Government's push to reduce import dependence is structurally positive "
            "for domestic producers."
        ),
        "SELL": (
            "Oil & Gas sector sensitive to crude price volatility. A global demand slowdown "
            "narrative is building, and refining margins have narrowed from their peak. "
            "Downstream marketing margins remain under political pricing pressure."
        ),
    },
    "Power": {
        "BUY": (
            "Power sector at an inflection point — India's peak power demand is growing "
            "at 6–8% annually, and the renewable capacity addition pipeline is the largest "
            "in the country's history. NTPC and Power Grid are compounding cash flows "
            "in a highly predictable regulatory environment."
        ),
        "SELL": (
            "Power sector facing regulatory risk on tariff revisions and state discom "
            "payment delays. Overcapacity in thermal generation is capping merchant "
            "power tariffs, weighing on merchant-heavy operators."
        ),
    },
    "Telecom": {
        "BUY": (
            "Telecom sector in a sustained ARPU upgrade cycle — the industry consolidated "
            "to three players years ago, and each tariff hike has stuck without meaningful "
            "churn. 5G monetisation is beginning to show up in enterprise revenue lines, "
            "with the consumer segment still ahead."
        ),
        "SELL": (
            "Telecom sector faces near-term pressure from elevated capex for 5G rollout "
            "compressing free cash flows. Competitive intensity could re-emerge if pricing "
            "discipline breaks down."
        ),
    },
    "Insurance": {
        "BUY": (
            "Insurance sector benefits from India's structurally underpenetrated protection "
            "market. Rising health awareness post-pandemic has driven consistent premium "
            "growth at 15–20% annually. VNB margins continue to expand as product mix "
            "shifts toward higher-margin protection products."
        ),
        "SELL": (
            "Insurance sector valuations remain stretched on embedded value multiples. "
            "Rising claims ratios in health insurance and regulatory changes around "
            "surrender charges could compress near-term profitability."
        ),
    },
    "Financial Services": {
        "BUY": (
            "NBFCs and diversified financials benefit from the broad credit growth cycle "
            "and AUM expansion in asset management. The secular shift of household savings "
            "from physical to financial assets has a long runway — SIP inflows are at "
            "record highs."
        ),
        "SELL": (
            "Financial services sector faces asset quality concerns in consumer finance "
            "and MFI segments. Regulatory tightening on unsecured loans and rising "
            "borrowing costs are squeezing spreads for several NBFCs."
        ),
    },
    "Consumer Goods": {
        "BUY": (
            "Consumer discretionary names benefiting from India's rising aspirational "
            "middle class. Premium jewellery and lifestyle brands have compounded revenue "
            "at 20%+ over the last three years, with little sign of demand saturation "
            "in Tier 1 and 2 cities."
        ),
        "SELL": (
            "Consumer discretionary facing pressure from slowing urban discretionary spend "
            "and high base effects. Elevated gold prices are creating near-term headwinds "
            "for jewellery demand."
        ),
    },
    "Cement": {
        "BUY": (
            "Cement sector is a direct play on India's housing and infrastructure boom — "
            "per-capita cement consumption remains well below developed-market levels, "
            "implying a decade-long structural tailwind. Recent capacity additions are "
            "being absorbed by healthy demand."
        ),
        "SELL": (
            "Cement sector under pressure from regional overcapacity and softening "
            "realisations. Energy cost volatility (pet coke and coal) continues to "
            "create margin uncertainty for large producers."
        ),
    },
    "Chemicals": {
        "BUY": (
            "Specialty chemicals sector benefits from the China+1 sourcing shift, "
            "with global multinationals diversifying their supply chains to Indian "
            "manufacturers. The sector has delivered 25%+ earnings CAGR over three years "
            "for leading players."
        ),
        "SELL": (
            "Chemicals sector facing volume and pricing pressure as Chinese producers "
            "resume aggressive export activity, undercutting Indian players on price. "
            "Working capital cycles have elongated, increasing cash flow risk."
        ),
    },
    "Conglomerate": {
        "BUY": (
            "Diversified conglomerates offer broad exposure to India's multi-sector growth "
            "story — from new energy and retail to telecom and financial services. "
            "Reliance-type holding structures historically re-rate when multiple business "
            "lines hit an inflection simultaneously."
        ),
        "SELL": (
            "Conglomerate discount may widen as investors prefer pure-play exposure. "
            "Capital-intensive new businesses are still in investment mode, creating "
            "near-term drag on consolidated returns."
        ),
    },
    "Mining": {
        "BUY": (
            "Coal India remains insulated from global commodity volatility given its "
            "captive domestic demand base — India's thermal power sector cannot reduce "
            "coal dependency meaningfully in the near term. Consistent dividend yield "
            "provides downside support."
        ),
        "SELL": (
            "Mining sector faces ESG headwinds and a gradual policy push toward "
            "renewable substitution. Volume growth is plateauing as the low-hanging "
            "production gains from recent quarters begin to normalise."
        ),
    },
}


def _build_signal_reasons(signal: str, feat, sector: str = "") -> list[str]:
    """
    Build a human-readable list of reasons for a BUY/SELL signal,
    layering sector macro context, business trajectory, and technical timing.
    """
    reasons = []

    def fv(key, default=0.0):
        val = feat.get(key, default)
        try:
            v = float(val)
            return default if (v != v) else v  # NaN check
        except (TypeError, ValueError):
            return default

    sector = sector or str(feat.get("sector", "") or "")

    # ── Layer 1: Sector macro narrative ─────────────────────────
    sector_ctx = _SECTOR_CONTEXT.get(sector, {})
    if sector_ctx.get(signal):
        reasons.append(sector_ctx[signal])

    # ── Layer 2: Business trajectory ────────────────────────────
    eg = fv("earnings_growth")
    rg = fv("revenue_growth")
    pe_vs_sector = fv("pe_vs_sector_median")
    pm = fv("profit_margin")

    if signal == "BUY":
        if eg > 0.20 and rg > 0.10:
            reasons.append(
                f"Business fundamentals are strong — earnings up {eg * 100:.0f}% "
                f"and revenue up {rg * 100:.0f}% year-on-year, showing the growth "
                f"story is backed by real numbers."
            )
        elif eg > 0.12:
            reasons.append(
                f"Earnings have grown {eg * 100:.0f}% year-on-year, suggesting the "
                f"company is in a sustained profitability expansion phase."
            )
        if pe_vs_sector < -0.20:
            reasons.append(
                f"Trades at a meaningful discount to its sector peers on P/E, "
                f"suggesting the market has not yet priced in the improving fundamentals."
            )
    elif signal == "SELL":
        if eg < -0.10:
            reasons.append(
                f"Earnings have contracted {abs(eg) * 100:.0f}% year-on-year — "
                f"the fundamental story is deteriorating, not just the price action."
            )
        if pe_vs_sector > 0.30:
            reasons.append(
                f"Trades at a significant premium to sector peers on P/E with earnings "
                f"momentum slowing — a re-rating lower looks increasingly likely."
            )

    # ── Layer 3: Technical timing narrative ─────────────────────
    rsi = fv("rsi_14")
    macd_hist = fv("macd_histogram")
    vol_ratio = fv("volume_ratio")
    ret5 = fv("ret_5d") * 100
    ret20 = fv("ret_20d") * 100
    close_vs_sma20 = fv("close_vs_sma20") * 100
    bb_pctb = fv("bb_pctb")

    tech_parts = []

    if signal == "BUY":
        # RSI — primary timing signal
        if rsi > 0 and rsi < 35:
            tech_parts.append(
                f"RSI at {rsi:.0f} signals the stock has been oversold — "
                f"sellers are exhausted and historically price has mean-reverted "
                f"strongly from these levels"
            )
        elif rsi > 0 and 50 < rsi < 68:
            tech_parts.append(
                f"RSI at {rsi:.0f} sits in bullish territory without being overbought, "
                f"leaving room for further upside"
            )

        # MACD confirmation
        if macd_hist > 0:
            tech_parts.append("MACD histogram has crossed into positive territory, confirming short-term momentum has turned")

        # Price trend / recent returns
        if ret5 > 3 and vol_ratio > 1.3:
            tech_parts.append(
                f"up {ret5:.1f}% over the past 5 days on {vol_ratio:.1f}x average "
                f"volume — institutional accumulation is visible in the tape"
            )
        elif ret20 < -8 and rsi < 45:
            tech_parts.append(
                f"stock has corrected {abs(ret20):.1f}% over the past month, "
                f"creating a meaningful discount to its recent range"
            )

        # Candlestick reversal (only meaningful alongside RSI/price context)
        if fv("hammer") > 0 or fv("morning_star") > 0:
            tech_parts.append("a bullish reversal candlestick pattern on the daily chart adds conviction to the entry")
        elif fv("engulfing") > 0:
            tech_parts.append("a bullish engulfing candle suggests buyers overpowered sellers decisively at this level")

        # BB only as a supporting note when RSI also confirms oversold
        if bb_pctb < 0.15 and rsi < 40:
            tech_parts.append("price is at the extreme lower end of its volatility range, compounding the oversold signal")

    elif signal == "SELL":
        # RSI — primary timing signal
        if rsi > 70:
            tech_parts.append(
                f"RSI at {rsi:.0f} is deep in overbought territory — "
                f"the rally has stretched too far too fast and a pullback is historically overdue"
            )
        elif rsi > 0 and rsi < 50:
            tech_parts.append(
                f"RSI at {rsi:.0f} has slipped below the 50 midline, signalling "
                f"the trend has shifted from buyers to sellers"
            )

        # MACD confirmation
        if macd_hist < 0:
            tech_parts.append("MACD histogram has crossed into negative territory, confirming momentum has rolled over")

        # Price trend
        if ret5 < -3 and vol_ratio > 1.3:
            tech_parts.append(
                f"down {abs(ret5):.1f}% over the past 5 days on elevated volume "
                f"— distribution by larger players is visible"
            )
        elif close_vs_sma20 > 8:
            tech_parts.append(
                f"price is trading {close_vs_sma20:.1f}% above its 20-day average "
                f"— extended moves of this magnitude typically see reversion"
            )

        # BB only as a supporting note when RSI also confirms overbought
        if bb_pctb > 0.85 and rsi > 65:
            tech_parts.append("price is at the extreme upper end of its volatility band, compounding the overbought signal")

    if tech_parts:
        reasons.append("Technically: " + "; ".join(tech_parts) + ".")

    # ── Layer 4: Market tailwinds / headwinds ───────────────────
    fii = fv("fii_net_buy")
    sp500 = fv("sp500_overnight_ret")
    vix = fv("india_vix")

    market_parts = []
    if signal == "BUY":
        if fii > 0:
            market_parts.append("FII flows are net positive, providing a broad market tailwind")
        if sp500 > 0.005:
            market_parts.append(f"US markets closed up {sp500 * 100:.1f}% overnight, setting a supportive global tone")
        if vix > 0 and vix < 15:
            market_parts.append("India VIX is subdued, suggesting low fear in the market — a constructive backdrop for longs")
    elif signal == "SELL":
        if fii < 0:
            market_parts.append("FII flows are net negative, adding selling pressure across the broader market")
        if sp500 < -0.005:
            market_parts.append(f"US markets dropped {abs(sp500) * 100:.1f}% overnight, creating a risk-off opening")
        if vix > 0 and vix > 20:
            market_parts.append(f"India VIX has spiked to {vix:.1f}, indicating elevated fear — headwind for risk assets")

    if market_parts:
        reasons.append("Market backdrop: " + "; ".join(market_parts) + ".")

    # ── Layer 5: News sentiment (if meaningful) ─────────────────
    news_count = fv("news_count_24h")
    vader = fv("vader_compound_mean")
    if news_count >= 3:
        if signal == "BUY" and vader > 0.15:
            reasons.append(
                f"News flow is turning positive — {int(news_count)} recent articles "
                f"are skewing bullish, which often precedes institutional re-rating."
            )
        elif signal == "SELL" and vader < -0.15:
            reasons.append(
                f"News flow is deteriorating — {int(news_count)} recent articles "
                f"carry negative sentiment, which can accelerate selling pressure."
            )

    return reasons if reasons else [
        "Model's ensemble of LightGBM, XGBoost, and Random Forest is in strong agreement "
        "on the direction, even without a single dominant technical or fundamental trigger."
    ]


def _get_model_date() -> str:
    """Get the last modified date of the model files."""
    model_path = SETTINGS.MODELS_DIR / "lgb_model.joblib"
    if model_path.exists():
        mtime = model_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    return "Unknown"


if __name__ == "__main__":
    main()
