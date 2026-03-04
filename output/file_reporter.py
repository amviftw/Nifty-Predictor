"""
File-based output for daily predictions.
Writes signals to CSV and JSON files for record-keeping.
"""

import csv
import json
from datetime import date
from pathlib import Path

from loguru import logger

from config.settings import SETTINGS


def write_signals_csv(signals: list[dict], target_date: str = None):
    """
    Write signals to a dated CSV file.
    File: storage/signals/YYYY-MM-DD_signals.csv
    """
    if target_date is None:
        target_date = date.today().isoformat()

    output_dir = SETTINGS.SIGNALS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{target_date}_signals.csv"

    fieldnames = [
        "date", "symbol", "signal", "confidence", "strength",
        "prob_up", "prob_flat", "prob_down", "position_size_pct", "sector",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sig in signals:
            row = {"date": target_date, **sig}
            writer.writerow(row)

    logger.info(f"Signals written to {filepath}")
    return filepath


def write_signals_json(signals: list[dict], target_date: str = None):
    """
    Write signals to a dated JSON file.
    File: storage/signals/YYYY-MM-DD_signals.json
    """
    if target_date is None:
        target_date = date.today().isoformat()

    output_dir = SETTINGS.SIGNALS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{target_date}_signals.json"

    output = {
        "date": target_date,
        "generated_at": date.today().isoformat(),
        "total_stocks": len(signals),
        "buy_signals": sum(1 for s in signals if s["signal"] == "BUY"),
        "sell_signals": sum(1 for s in signals if s["signal"] == "SELL"),
        "signals": signals,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Signals written to {filepath}")
    return filepath
