"""
Feature selection pipeline.
Three-stage approach: variance filter -> model importance -> correlation filter.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    variance_threshold: float = 0.01,
    top_k: int = 60,
    correlation_threshold: float = 0.95,
) -> list[str]:
    """
    Three-stage feature selection pipeline.

    Stage 1: Remove near-constant features (variance < threshold)
    Stage 2: Select top-k features by LightGBM importance
    Stage 3: Remove one of each pair with correlation > threshold

    Returns: list of selected feature names
    """
    original_count = X.shape[1]
    logger.info(f"Feature selection: starting with {original_count} features")

    # Stage 1: Variance threshold
    variances = X.var()
    low_var = variances[variances < variance_threshold].index.tolist()
    if low_var:
        X = X.drop(columns=low_var)
        logger.info(f"Stage 1: Removed {len(low_var)} low-variance features")

    # Stage 2: Model-based importance
    if X.shape[1] > top_k:
        selected = _select_by_importance(X, y, top_k)
        X = X[selected]
        logger.info(f"Stage 2: Selected top {len(selected)} features by importance")

    # Stage 3: Remove highly correlated features
    before_corr = X.shape[1]
    selected = _remove_correlated(X, correlation_threshold)
    logger.info(
        f"Stage 3: Removed {before_corr - len(selected)} correlated features"
    )

    logger.info(
        f"Feature selection complete: {original_count} -> {len(selected)} features"
    )
    return selected


def _select_by_importance(
    X: pd.DataFrame, y: pd.Series, top_k: int
) -> list[str]:
    """Select top-k features using LightGBM feature importance (gain)."""
    model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        verbose=-1,
        class_weight="balanced",
        importance_type="gain",
    )
    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return importances.head(top_k).index.tolist()


def _remove_correlated(
    X: pd.DataFrame, threshold: float
) -> list[str]:
    """Remove one of each pair of features with correlation > threshold."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        if correlated:
            to_drop.add(col)

    return [c for c in X.columns if c not in to_drop]
