#!/usr/bin/env python3
"""
Backtest and evaluate historical signal accuracy.
Compares past predictions against actual outcomes.

Usage:
    python -m scripts.evaluate_signals
    python -m scripts.evaluate_signals --days 30
"""

import sys
import argparse
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from config.settings import SETTINGS
from data.storage.db_manager import DBManager
from models.evaluator import evaluate_signals_backtest

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate signal performance")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days to look back for evaluation",
    )
    parser.add_argument(
        "--email", action="store_true",
        help="Send evaluation report via email",
    )
    args = parser.parse_args()

    db = DBManager(SETTINGS.DB_PATH)

    logger.info("=" * 60)
    logger.info("SIGNAL PERFORMANCE EVALUATION")
    logger.info("=" * 60)

    # Step 1: Update outcomes for past predictions
    logger.info("Updating actual outcomes for past predictions...")
    _update_outcomes(db)

    # Step 2: Get predictions with outcomes
    prev_preds = db.get_previous_predictions(n_days=args.days)

    if not prev_preds:
        print("\nNo historical predictions found. Run daily_predict.py first.")
        return

    df = pd.DataFrame(prev_preds)
    df_with_outcomes = df.dropna(subset=["actual_ret"])

    if df_with_outcomes.empty:
        print("\nNo outcomes available yet. Predictions need at least 1 day to verify.")
        return

    # Step 3: Overall performance
    perf = evaluate_signals_backtest(df_with_outcomes)

    console.print()
    console.rule("[bold cyan]Signal Performance Report")
    console.print()

    # Summary table
    summary_table = Table(title="Overall Performance", box=box.ROUNDED)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Total Predictions", str(perf.get("total_signals", 0)))
    summary_table.add_row("Actionable Signals", str(perf.get("actionable_signals", 0)))

    if "overall_signal_accuracy" in perf:
        summary_table.add_row(
            "Signal Accuracy",
            f"{perf['overall_signal_accuracy'] * 100:.1f}%",
        )

    if "buy_count" in perf:
        summary_table.add_row("BUY Signals", str(perf["buy_count"]))
        summary_table.add_row("BUY Avg Return", f"{perf['buy_avg_return'] * 100:.2f}%")
        summary_table.add_row("BUY Win Rate", f"{perf['buy_win_rate'] * 100:.0f}%")
        summary_table.add_row("BUY Correct (>1%)", f"{perf['buy_correct_rate'] * 100:.0f}%")

    if "sell_count" in perf:
        summary_table.add_row("SELL Signals", str(perf["sell_count"]))
        summary_table.add_row("SELL Avg Return", f"{perf['sell_avg_return'] * 100:.2f}%")
        summary_table.add_row("SELL Win Rate", f"{perf['sell_win_rate'] * 100:.0f}%")
        summary_table.add_row("SELL Correct (<-1%)", f"{perf['sell_correct_rate'] * 100:.0f}%")

    console.print(summary_table)
    console.print()

    # Step 4: Daily breakdown
    daily_table = Table(title="Daily Breakdown", box=box.SIMPLE)
    daily_table.add_column("Date", style="bold")
    daily_table.add_column("BUY", justify="center")
    daily_table.add_column("SELL", justify="center")
    daily_table.add_column("Correct", justify="center")
    daily_table.add_column("Accuracy", justify="right")

    for d, group in df_with_outcomes.groupby("date"):
        actionable = group[group["signal"] != "HOLD"]
        if actionable.empty:
            continue

        buys = (actionable["signal"] == "BUY").sum()
        sells = (actionable["signal"] == "SELL").sum()
        correct = (
            ((actionable["signal"] == "BUY") & (actionable["actual_class"] == 2))
            | ((actionable["signal"] == "SELL") & (actionable["actual_class"] == 0))
        ).sum()
        total = len(actionable)
        acc = correct / total if total > 0 else 0

        daily_table.add_row(
            str(d), str(buys), str(sells), f"{correct}/{total}", f"{acc * 100:.0f}%"
        )

    console.print(daily_table)
    console.print()

    # Step 5: Top performing and worst performing signals
    _print_best_worst(df_with_outcomes)

    console.rule(style="cyan")

    # Email notification
    if args.email:
        try:
            from output.email_notifier import send_evaluation_report_email
            send_evaluation_report_email(perf, days=args.days)
        except Exception as e:
            logger.error(f"Failed to send evaluation email: {e}")


def _update_outcomes(db: DBManager):
    """Update actual outcomes for past predictions."""
    with db.connect() as conn:
        # Find predictions without outcomes
        rows = conn.execute(
            """SELECT DISTINCT p.date, p.symbol
               FROM predictions p
               LEFT JOIN outcomes o ON p.date = o.date AND p.symbol = o.symbol
               WHERE o.actual_ret IS NULL"""
        ).fetchall()

    if not rows:
        return

    for row in rows:
        pred_date = row["date"]
        symbol = row["symbol"]

        # Get the next trading day's close
        ohlcv = db.get_ohlcv(symbol, start_date=pred_date)
        if len(ohlcv) < 2:
            continue

        # ohlcv[0] is the prediction date, ohlcv[1] is the next day
        close_today = ohlcv[0]["adj_close"]
        close_next = ohlcv[1]["adj_close"]

        if close_today and close_next and close_today > 0:
            actual_ret = (close_next - close_today) / close_today
            if actual_ret > SETTINGS.TARGET_THRESHOLD:
                actual_class = 2  # UP
            elif actual_ret < -SETTINGS.TARGET_THRESHOLD:
                actual_class = 0  # DOWN
            else:
                actual_class = 1  # FLAT

            db.insert_outcomes([{
                "date": pred_date,
                "symbol": symbol,
                "actual_ret": actual_ret,
                "actual_class": actual_class,
            }])


def _print_best_worst(df: pd.DataFrame):
    """Print best and worst signal performances."""
    actionable = df[df["signal"] != "HOLD"].copy()
    if actionable.empty:
        return

    # Best BUY signals
    buys = actionable[actionable["signal"] == "BUY"].copy()
    if not buys.empty:
        buys = buys.sort_values("actual_ret", ascending=False)
        table = Table(title="Top BUY Signals", box=box.SIMPLE)
        table.add_column("Date")
        table.add_column("Stock", style="bold")
        table.add_column("Confidence", justify="right")
        table.add_column("Actual Return", justify="right")

        for _, row in buys.head(5).iterrows():
            ret_color = "green" if row["actual_ret"] > 0 else "red"
            table.add_row(
                str(row["date"]),
                str(row["symbol"]),
                f"{row['confidence'] * 100:.1f}%",
                f"[{ret_color}]{row['actual_ret'] * 100:+.2f}%[/{ret_color}]",
            )
        console.print(table)
        console.print()

    # Worst signals
    worst = actionable.copy()
    worst["signal_correct"] = (
        ((worst["signal"] == "BUY") & (worst["actual_ret"] > 0))
        | ((worst["signal"] == "SELL") & (worst["actual_ret"] < 0))
    )
    wrong = worst[~worst["signal_correct"]].sort_values("confidence", ascending=False)

    if not wrong.empty:
        table = Table(title="Worst Mispredictions (high confidence, wrong direction)", box=box.SIMPLE)
        table.add_column("Date")
        table.add_column("Stock", style="bold")
        table.add_column("Signal")
        table.add_column("Confidence", justify="right")
        table.add_column("Actual Return", justify="right")

        for _, row in wrong.head(5).iterrows():
            table.add_row(
                str(row["date"]),
                str(row["symbol"]),
                str(row["signal"]),
                f"{row['confidence'] * 100:.1f}%",
                f"[red]{row['actual_ret'] * 100:+.2f}%[/red]",
            )
        console.print(table)


if __name__ == "__main__":
    main()
