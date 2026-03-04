"""
Hyperparameter tuning using Optuna with walk-forward validation.
Run monthly to find optimal model parameters.
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from loguru import logger

from models.trainer import WalkForwardTrainer
from config.settings import SETTINGS


# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    n_trials: int = 50,
    timeout: int = 3600,
) -> dict:
    """
    Tune LightGBM hyperparameters using Optuna.
    Objective: maximize F1-macro on walk-forward validation.
    """
    trainer = WalkForwardTrainer(
        min_train_days=SETTINGS.MIN_TRAIN_DAYS,
        val_window_days=SETTINGS.VAL_WINDOW_DAYS,
        step_days=SETTINGS.VAL_WINDOW_DAYS,
        purge_days=1,
    )
    splits = trainer.generate_splits(dates)

    if not splits:
        logger.error("Not enough data for tuning")
        return SETTINGS.LGB_PARAMS

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "verbose": -1,
            "class_weight": "balanced",
        }

        scores = []
        for train_mask, val_mask in splits:
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            if len(y_train.unique()) < 3:
                continue

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        return np.mean(scores) if scores else 0.0

    logger.info(f"Starting LightGBM tuning ({n_trials} trials, {timeout}s timeout)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_params
    best.update({
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
        "class_weight": "balanced",
    })

    logger.info(f"Best LightGBM F1-macro: {study.best_value:.4f}")
    logger.info(f"Best params: {best}")
    return best


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    n_trials: int = 50,
    timeout: int = 3600,
) -> dict:
    """Tune XGBoost hyperparameters using Optuna."""
    trainer = WalkForwardTrainer(
        min_train_days=SETTINGS.MIN_TRAIN_DAYS,
        val_window_days=SETTINGS.VAL_WINDOW_DAYS,
        step_days=SETTINGS.VAL_WINDOW_DAYS,
        purge_days=1,
    )
    splits = trainer.generate_splits(dates)

    if not splits:
        return SETTINGS.XGB_PARAMS

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "tree_method": "hist",
            "verbosity": 0,
        }

        scores = []
        for train_mask, val_mask in splits:
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            if len(y_train.unique()) < 3:
                continue

            # Sample weights for class imbalance
            class_counts = y_train.value_counts()
            total = len(y_train)
            weights = y_train.map(lambda c: total / (3 * class_counts[c]))

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=weights)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        return np.mean(scores) if scores else 0.0

    logger.info(f"Starting XGBoost tuning ({n_trials} trials, {timeout}s timeout)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_params
    best.update({
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "verbosity": 0,
    })

    logger.info(f"Best XGBoost F1-macro: {study.best_value:.4f}")
    return best


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    n_trials: int = 30,
    timeout: int = 1800,
) -> dict:
    """Tune Random Forest hyperparameters using Optuna."""
    trainer = WalkForwardTrainer(
        min_train_days=SETTINGS.MIN_TRAIN_DAYS,
        val_window_days=SETTINGS.VAL_WINDOW_DAYS,
        step_days=SETTINGS.VAL_WINDOW_DAYS,
        purge_days=1,
    )
    splits = trainer.generate_splits(dates)

    if not splits:
        return SETTINGS.RF_PARAMS

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        }

        scores = []
        for train_mask, val_mask in splits:
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            if len(y_train.unique()) < 3:
                continue

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        return np.mean(scores) if scores else 0.0

    logger.info(f"Starting RF tuning ({n_trials} trials, {timeout}s timeout)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_params
    best.update({"class_weight": "balanced", "n_jobs": -1, "random_state": 42})

    logger.info(f"Best RF F1-macro: {study.best_value:.4f}")
    return best
