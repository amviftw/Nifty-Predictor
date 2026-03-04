"""
Walk-forward validation training loop.
Ensures no lookahead bias in model training and evaluation.
"""

import numpy as np
import pandas as pd
from loguru import logger


class WalkForwardTrainer:
    """
    Walk-forward (expanding window) cross-validator for time series.

    Timeline:
        Fold 1: Train on months 1-12,  Validate on month 13
        Fold 2: Train on months 1-13,  Validate on month 14
        ...
        Fold N: Train on months 1-(N+11), Validate on month (N+12)
    """

    def __init__(
        self,
        min_train_days: int = 252,
        val_window_days: int = 21,
        step_days: int = 21,
        purge_days: int = 1,
    ):
        """
        Args:
            min_train_days: Minimum number of trading days for initial training window
            val_window_days: Number of trading days in each validation window
            step_days: Number of trading days to step forward between folds
            purge_days: Gap between train and val to prevent leakage
        """
        self.min_train_days = min_train_days
        self.val_window_days = val_window_days
        self.step_days = step_days
        self.purge_days = purge_days

    def generate_splits(
        self, dates: pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_indices, val_indices) tuples for walk-forward CV.

        Args:
            dates: Series of date strings, aligned with feature matrix index

        Returns:
            List of (train_mask, val_mask) boolean arrays
        """
        unique_dates = sorted(dates.unique())
        n = len(unique_dates)

        if n < self.min_train_days + self.val_window_days:
            logger.warning(
                f"Not enough data for walk-forward: {n} dates, "
                f"need {self.min_train_days + self.val_window_days}"
            )
            return []

        splits = []
        val_start_idx = self.min_train_days

        while val_start_idx + self.val_window_days <= n:
            train_end_idx = val_start_idx - self.purge_days
            val_end_idx = val_start_idx + self.val_window_days

            train_dates = set(unique_dates[:train_end_idx])
            val_dates = set(unique_dates[val_start_idx:val_end_idx])

            train_mask = dates.isin(train_dates).values
            val_mask = dates.isin(val_dates).values

            if train_mask.sum() > 0 and val_mask.sum() > 0:
                splits.append((train_mask, val_mask))

            val_start_idx += self.step_days

        logger.info(f"Generated {len(splits)} walk-forward folds")
        return splits

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        model_factory,
        metric_fn,
    ) -> list[dict]:
        """
        Run walk-forward validation.

        Args:
            X: Feature matrix
            y: Target labels
            dates: Date series aligned with X
            model_factory: Callable that returns a fresh model instance
            metric_fn: Callable(y_true, y_pred) -> float

        Returns:
            List of dicts with fold results
        """
        splits = self.generate_splits(dates)
        results = []

        for fold_idx, (train_mask, val_mask) in enumerate(splits):
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            # Skip if any class is missing from training
            if len(y_train.unique()) < 3:
                logger.warning(f"Fold {fold_idx}: skipping, not all classes present")
                continue

            model = model_factory()

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = metric_fn(y_val, y_pred)

                results.append({
                    "fold": fold_idx,
                    "train_size": int(train_mask.sum()),
                    "val_size": int(val_mask.sum()),
                    "score": score,
                })

                if fold_idx % 3 == 0:
                    logger.info(
                        f"Fold {fold_idx}: train={train_mask.sum()}, "
                        f"val={val_mask.sum()}, score={score:.4f}"
                    )
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                continue

        if results:
            scores = [r["score"] for r in results]
            logger.info(
                f"Walk-forward results: mean={np.mean(scores):.4f}, "
                f"std={np.std(scores):.4f}, n_folds={len(results)}"
            )

        return results
