"""
Model evaluation metrics and reporting.
Provides classification metrics, confusion matrices, and backtesting analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from loguru import logger


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, probas: np.ndarray = None
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True class labels (0=DOWN, 1=FLAT, 2=UP)
        y_pred: Predicted class labels
        probas: Optional probability matrix (n_samples, 3)

    Returns: dict with all metrics
    """
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Per-class metrics
    class_names = ["DOWN", "FLAT", "UP"]
    for cls_idx, cls_name in enumerate(class_names):
        cls_mask = y_true == cls_idx
        pred_mask = y_pred == cls_idx

        if cls_mask.sum() > 0:
            metrics[f"{cls_name}_precision"] = precision_score(
                y_true == cls_idx, y_pred == cls_idx, zero_division=0
            )
            metrics[f"{cls_name}_recall"] = recall_score(
                y_true == cls_idx, y_pred == cls_idx, zero_division=0
            )
            metrics[f"{cls_name}_count"] = int(cls_mask.sum())

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics["confusion_matrix"] = cm.tolist()

    # Directional accuracy (for BUY/SELL signals only, ignoring HOLD)
    directional_mask = (y_pred != 1) & (y_true != 1)  # Both non-FLAT
    if directional_mask.sum() > 0:
        metrics["directional_accuracy"] = accuracy_score(
            y_true[directional_mask], y_pred[directional_mask]
        )
    else:
        metrics["directional_accuracy"] = 0.0

    return metrics


def print_evaluation_report(metrics: dict, title: str = "Model Evaluation"):
    """Print a formatted evaluation report."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {title}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  F1-macro:            {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-weighted:         {metrics['f1_weighted']:.4f}")
    logger.info(f"  Accuracy:            {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (macro):   {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro):      {metrics['recall_macro']:.4f}")
    logger.info(f"  Directional Acc:     {metrics['directional_accuracy']:.4f}")
    logger.info("")

    # Per-class
    for cls in ["DOWN", "FLAT", "UP"]:
        p = metrics.get(f"{cls}_precision", 0)
        r = metrics.get(f"{cls}_recall", 0)
        n = metrics.get(f"{cls}_count", 0)
        logger.info(f"  {cls:5s}  precision={p:.3f}  recall={r:.3f}  support={n}")

    # Confusion matrix
    cm = metrics.get("confusion_matrix", [])
    if cm:
        logger.info("")
        logger.info("  Confusion Matrix (rows=actual, cols=predicted):")
        logger.info("              DOWN    FLAT      UP")
        labels = ["DOWN", "FLAT", "  UP"]
        for i, row in enumerate(cm):
            logger.info(f"  {labels[i]:5s}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")

    logger.info(f"{'=' * 60}")


def evaluate_signals_backtest(
    predictions_df: pd.DataFrame,
) -> dict:
    """
    Evaluate historical signal performance.

    predictions_df must have columns:
        signal, confidence, actual_ret, actual_class

    Returns dict with signal-level performance metrics.
    """
    if predictions_df.empty:
        return {}

    # Filter to predictions with actual outcomes
    df = predictions_df.dropna(subset=["actual_ret"]).copy()
    if df.empty:
        return {"message": "No outcomes available for backtesting"}

    results = {
        "total_signals": len(df),
    }

    # BUY signal performance
    buy_mask = df["signal"] == "BUY"
    if buy_mask.sum() > 0:
        buy_df = df[buy_mask]
        results["buy_count"] = int(buy_mask.sum())
        results["buy_avg_return"] = float(buy_df["actual_ret"].mean())
        results["buy_win_rate"] = float((buy_df["actual_ret"] > 0).mean())
        results["buy_correct_rate"] = float(
            (buy_df["actual_class"] == 2).mean()
        )

    # SELL signal performance
    sell_mask = df["signal"] == "SELL"
    if sell_mask.sum() > 0:
        sell_df = df[sell_mask]
        results["sell_count"] = int(sell_mask.sum())
        results["sell_avg_return"] = float(sell_df["actual_ret"].mean())
        results["sell_win_rate"] = float((sell_df["actual_ret"] < 0).mean())
        results["sell_correct_rate"] = float(
            (sell_df["actual_class"] == 0).mean()
        )

    # Overall signal accuracy
    actionable = df[df["signal"] != "HOLD"]
    if len(actionable) > 0:
        correct = (
            ((actionable["signal"] == "BUY") & (actionable["actual_class"] == 2))
            | ((actionable["signal"] == "SELL") & (actionable["actual_class"] == 0))
        )
        results["overall_signal_accuracy"] = float(correct.mean())
        results["actionable_signals"] = int(len(actionable))

    return results
