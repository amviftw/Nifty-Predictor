"""
Multi-model ensemble predictor.
Combines LightGBM, XGBoost, and Random Forest via weighted soft voting.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config.settings import SETTINGS


class EnsemblePredictor:
    """Weighted soft-voting ensemble of LightGBM, XGBoost, and Random Forest."""

    def __init__(
        self,
        lgb_model=None,
        xgb_model=None,
        rf_model=None,
        weights: tuple = (0.40, 0.35, 0.25),
    ):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.rf_model = rf_model
        self.weights = weights
        self.feature_names = None

    def train(
        self, X: pd.DataFrame, y: pd.Series,
        lgb_params: dict = None,
        xgb_params: dict = None,
        rf_params: dict = None,
    ):
        """Train all three models on the provided data."""
        if lgb_params is None:
            lgb_params = SETTINGS.LGB_PARAMS.copy()
        if xgb_params is None:
            xgb_params = SETTINGS.XGB_PARAMS.copy()
        if rf_params is None:
            rf_params = SETTINGS.RF_PARAMS.copy()

        self.feature_names = list(X.columns)

        # Train LightGBM
        logger.info("Training LightGBM...")
        lgb_clf_params = {k: v for k, v in lgb_params.items()
                          if k not in ("objective", "metric", "num_class")}
        self.lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            **lgb_clf_params,
        )
        self.lgb_model.fit(X, y)
        logger.info("LightGBM trained")

        # Train XGBoost
        logger.info("Training XGBoost...")
        # Compute sample weights for class imbalance
        class_counts = y.value_counts()
        total = len(y)
        sample_weights = y.map(lambda c: total / (3 * class_counts[c]))

        xgb_clf_params = {k: v for k, v in xgb_params.items()
                          if k not in ("objective", "eval_metric", "num_class")}
        self.xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            **xgb_clf_params,
        )
        self.xgb_model.fit(X, y, sample_weight=sample_weights)
        logger.info("XGBoost trained")

        # Train Random Forest
        logger.info("Training Random Forest...")
        self.rf_model = RandomForestClassifier(**rf_params)
        self.rf_model.fit(X, y)
        logger.info("Random Forest trained")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute weighted average of class probabilities.

        Returns: ndarray of shape (n_samples, 3) for [prob_down, prob_flat, prob_up]
        """
        probas = np.zeros((X.shape[0], 3))

        models = [self.lgb_model, self.xgb_model, self.rf_model]
        for model, weight in zip(models, self.weights):
            if model is not None:
                probas += weight * model.predict_proba(X)

        # Normalize (in case weights don't sum to 1 due to missing models)
        row_sums = probas.sum(axis=1, keepdims=True)
        probas = probas / (row_sums + 1e-10)

        return probas

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (argmax of probabilities)."""
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)

    def update_weights(
        self, X_val: pd.DataFrame, y_val: pd.Series
    ):
        """
        Update ensemble weights based on validation performance.
        Each model's weight is proportional to its F1-macro score.
        """
        scores = []
        models = [self.lgb_model, self.xgb_model, self.rf_model]
        names = ["LightGBM", "XGBoost", "RandomForest"]

        for model, name in zip(models, names):
            if model is not None:
                pred = model.predict(X_val)
                score = f1_score(y_val, pred, average="macro")
                scores.append(score)
                logger.info(f"{name} validation F1-macro: {score:.4f}")
            else:
                scores.append(0.0)

        total = sum(scores)
        if total > 0:
            self.weights = tuple(s / total for s in scores)
        logger.info(
            f"Updated weights: LGB={self.weights[0]:.3f}, "
            f"XGB={self.weights[1]:.3f}, RF={self.weights[2]:.3f}"
        )

    def save(self, models_dir: Path = None):
        """Save all models and metadata to disk."""
        if models_dir is None:
            models_dir = SETTINGS.MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.lgb_model, models_dir / "lgb_model.joblib")
        joblib.dump(self.xgb_model, models_dir / "xgb_model.joblib")
        joblib.dump(self.rf_model, models_dir / "rf_model.joblib")
        joblib.dump(self.weights, models_dir / "ensemble_weights.joblib")
        joblib.dump(self.feature_names, models_dir / "feature_list.joblib")

        logger.info(f"Models saved to {models_dir}")

    @classmethod
    def load(cls, models_dir: Path = None) -> "EnsemblePredictor":
        """Load a saved ensemble from disk."""
        if models_dir is None:
            models_dir = SETTINGS.MODELS_DIR

        lgb_model = joblib.load(models_dir / "lgb_model.joblib")
        xgb_model = joblib.load(models_dir / "xgb_model.joblib")
        rf_model = joblib.load(models_dir / "rf_model.joblib")
        weights = joblib.load(models_dir / "ensemble_weights.joblib")
        feature_names = joblib.load(models_dir / "feature_list.joblib")

        ensemble = cls(lgb_model, xgb_model, rf_model, weights)
        ensemble.feature_names = feature_names

        logger.info(
            f"Loaded ensemble from {models_dir} "
            f"(weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, RF={weights[2]:.3f})"
        )
        return ensemble
