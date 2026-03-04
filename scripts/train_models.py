#!/usr/bin/env python3
"""
Model training script.
Run weekly (or on demand) to retrain the ensemble model.

Usage:
    python -m scripts.train_models
    python -m scripts.train_models --start-date 2023-06-01 --end-date 2026-02-28
"""

import sys
import argparse
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score

from config.settings import SETTINGS
from data.storage.db_manager import DBManager
from features.feature_engineer import FeatureEngineer, get_all_feature_names
from models.target import compute_targets_for_training, get_class_distribution
from models.feature_selector import select_features
from models.trainer import WalkForwardTrainer
from models.ensemble import EnsemblePredictor
from models.evaluator import evaluate_predictions, print_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Train prediction models")
    parser.add_argument("--start-date", default=SETTINGS.BACKFILL_START)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--email", action="store_true", help="Send training report via email")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NIFTY 50 PREDICTOR - MODEL TRAINING")
    logger.info("=" * 60)

    db = DBManager(SETTINGS.DB_PATH)

    # Step 1: Compute features
    logger.info("Step 1: Computing features for training period...")
    engineer = FeatureEngineer(db)
    features_df = engineer.compute_training_features(args.start_date, args.end_date)

    if features_df.empty:
        logger.error("No features computed. Run backfill_data.py first.")
        return

    logger.info(f"Features shape: {features_df.shape}")

    # Step 2: Compute targets
    logger.info("Step 2: Computing target labels...")
    targets = compute_targets_for_training(features_df, db, SETTINGS.TARGET_THRESHOLD)

    # Remove rows with NaN targets
    valid_mask = targets.notna()
    features_df = features_df[valid_mask].reset_index(drop=True)
    targets = targets[valid_mask].reset_index(drop=True).astype(int)

    logger.info(f"Training samples: {len(features_df)}")
    dist = get_class_distribution(targets)
    logger.info(f"Class distribution: {dist}")

    # Step 3: Prepare feature matrix
    meta_cols = ["symbol", "date"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    X = features_df[feature_cols].astype(float)
    y = targets
    dates = features_df["date"]

    # Handle infinities and NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Step 4: Feature selection
    logger.info("Step 3: Feature selection...")
    selected_features = select_features(X, y, top_k=60, correlation_threshold=0.95)
    X = X[selected_features]
    logger.info(f"Selected {len(selected_features)} features")

    # Step 5: Walk-forward validation
    logger.info("Step 4: Walk-forward validation...")
    trainer = WalkForwardTrainer(
        min_train_days=SETTINGS.MIN_TRAIN_DAYS,
        val_window_days=SETTINGS.VAL_WINDOW_DAYS,
        step_days=SETTINGS.VAL_WINDOW_DAYS,
        purge_days=1,
    )

    metric_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)

    # Validate each model type
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    logger.info("\nValidating LightGBM...")
    lgb_results = trainer.validate(
        X, y, dates,
        model_factory=lambda: lgb.LGBMClassifier(**{
            k: v for k, v in SETTINGS.LGB_PARAMS.items()
            if k not in ("objective", "metric", "num_class")
        }, objective="multiclass", num_class=3),
        metric_fn=metric_fn,
    )

    logger.info("\nValidating XGBoost...")
    xgb_results = trainer.validate(
        X, y, dates,
        model_factory=lambda: xgb.XGBClassifier(**{
            k: v for k, v in SETTINGS.XGB_PARAMS.items()
            if k not in ("objective", "eval_metric", "num_class")
        }, objective="multi:softprob", num_class=3, eval_metric="mlogloss"),
        metric_fn=metric_fn,
    )

    logger.info("\nValidating Random Forest...")
    rf_results = trainer.validate(
        X, y, dates,
        model_factory=lambda: RandomForestClassifier(**SETTINGS.RF_PARAMS),
        metric_fn=metric_fn,
    )

    # Print summary
    lgb_mean = np.mean([r["score"] for r in lgb_results]) if lgb_results else 0
    xgb_mean = np.mean([r["score"] for r in xgb_results]) if xgb_results else 0
    rf_mean = np.mean([r["score"] for r in rf_results]) if rf_results else 0

    logger.info(f"\nWalk-Forward F1-Macro Scores:")
    logger.info(f"  LightGBM:      {lgb_mean:.4f}")
    logger.info(f"  XGBoost:       {xgb_mean:.4f}")
    logger.info(f"  Random Forest: {rf_mean:.4f}")

    # Step 6: Train final production models
    logger.info("\nStep 5: Training final production models on all data...")
    ensemble = EnsemblePredictor()
    ensemble.train(X, y)

    # Compute dynamic weights from validation scores
    total_score = lgb_mean + xgb_mean + rf_mean
    if total_score > 0:
        ensemble.weights = (
            lgb_mean / total_score,
            xgb_mean / total_score,
            rf_mean / total_score,
        )
    logger.info(
        f"Ensemble weights: LGB={ensemble.weights[0]:.3f}, "
        f"XGB={ensemble.weights[1]:.3f}, RF={ensemble.weights[2]:.3f}"
    )

    # Evaluate on last validation fold (in-sample, for reference)
    y_pred = ensemble.predict(X)
    probas = ensemble.predict_proba(X)
    metrics = evaluate_predictions(y.values, y_pred, probas)
    print_evaluation_report(metrics, "In-Sample Evaluation (reference only)")

    # Step 7: Save models
    logger.info("\nStep 6: Saving models...")
    ensemble.feature_names = selected_features
    ensemble.save()

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Models saved to: {SETTINGS.MODELS_DIR}")
    logger.info(f"  Features: {len(selected_features)}")
    logger.info(f"  Training samples: {len(X)}")
    logger.info("=" * 60)

    # Email notification
    if args.email:
        try:
            from output.email_notifier import send_training_report_email
            send_training_report_email(
                lgb_score=lgb_mean,
                xgb_score=xgb_mean,
                rf_score=rf_mean,
                weights=ensemble.weights,
                n_features=len(selected_features),
                n_samples=len(X),
            )
        except Exception as e:
            logger.error(f"Failed to send training email: {e}")


if __name__ == "__main__":
    main()
