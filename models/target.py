"""
Target variable construction for the prediction model.
Computes 3-class labels: UP (>1%), FLAT (-1% to +1%), DOWN (<-1%).
"""

import numpy as np
import pandas as pd


def compute_target(
    ohlcv_df: pd.DataFrame, threshold: float = 0.01
) -> pd.Series:
    """
    Compute next-day return and classify into 3 classes.

    Uses adj_close for return calculation.
    Target is shifted: row t has the label for day t+1.

    Classes:
        0 = DOWN:  next_day_return < -threshold
        1 = FLAT:  -threshold <= next_day_return <= +threshold
        2 = UP:    next_day_return > +threshold

    Returns: Series of integer labels (0, 1, 2) with NaN for last row.
    """
    close = ohlcv_df["adj_close"].astype(float)
    next_day_ret = close.pct_change().shift(-1)

    target = pd.Series(1, index=ohlcv_df.index, dtype=float)  # default FLAT
    target[next_day_ret > threshold] = 2    # UP
    target[next_day_ret < -threshold] = 0   # DOWN
    target[next_day_ret.isna()] = np.nan    # unknown (last row)

    return target


def compute_returns(ohlcv_df: pd.DataFrame) -> pd.Series:
    """Compute next-day returns (for evaluation)."""
    close = ohlcv_df["adj_close"].astype(float)
    return close.pct_change().shift(-1)


def compute_targets_for_training(
    features_df: pd.DataFrame,
    db,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Compute target labels for the training dataset.

    features_df must have 'symbol' and 'date' columns.
    Looks up next-day returns from the database.

    Returns Series aligned with features_df index.
    """
    targets = pd.Series(np.nan, index=features_df.index)

    for symbol in features_df["symbol"].unique():
        mask = features_df["symbol"] == symbol
        symbol_dates = features_df.loc[mask, "date"].tolist()

        # Get OHLCV for this stock
        ohlcv = db.get_ohlcv(symbol)
        if not ohlcv:
            continue

        ohlcv_df = pd.DataFrame(ohlcv)
        ohlcv_df = ohlcv_df.sort_values("date")

        # Build a date -> next_day_return mapping
        closes = ohlcv_df.set_index("date")["adj_close"].astype(float)
        returns = closes.pct_change().shift(-1)

        for idx in features_df.index[mask]:
            d = features_df.loc[idx, "date"]
            if d in returns.index:
                ret = returns[d]
                if pd.notna(ret):
                    if ret > threshold:
                        targets[idx] = 2
                    elif ret < -threshold:
                        targets[idx] = 0
                    else:
                        targets[idx] = 1

    return targets


def get_class_distribution(targets: pd.Series) -> dict:
    """Get class distribution statistics."""
    valid = targets.dropna().astype(int)
    total = len(valid)
    if total == 0:
        return {"total": 0, "down": 0, "flat": 0, "up": 0}

    return {
        "total": total,
        "down": int((valid == 0).sum()),
        "down_pct": f"{(valid == 0).mean() * 100:.1f}%",
        "flat": int((valid == 1).sum()),
        "flat_pct": f"{(valid == 1).mean() * 100:.1f}%",
        "up": int((valid == 2).sum()),
        "up_pct": f"{(valid == 2).mean() * 100:.1f}%",
    }
