"""
Technical analysis feature computation.
Computes ~60 technical indicators from OHLCV data using pandas-ta.
"""

import numpy as np
import pandas as pd

try:
    import pandas_ta as pta
    _PTA_AVAILABLE = True
except ImportError:
    _PTA_AVAILABLE = False

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features from OHLCV data.

    Input: DataFrame with columns [date, open, high, low, close, adj_close, volume]
           Must be sorted by date ascending.
    Output: DataFrame with ~60 new feature columns appended.
    """
    df = df.copy()

    # Use adj_close for all calculations to handle splits/dividends
    close = df["adj_close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    # ========== TREND INDICATORS (8 features) ==========
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["close_vs_sma20"] = (close - df["sma_20"]) / df["sma_20"]
    df["close_vs_sma50"] = (close - df["sma_50"]) / df["sma_50"]

    # ========== MOMENTUM INDICATORS (12 features) ==========
    # RSI
    df["rsi_14"] = _compute_rsi(close, 14)
    df["rsi_7"] = _compute_rsi(close, 7)

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # Stochastic Oscillator
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Williams %R
    df["williams_r"] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)

    # Rate of Change
    df["roc_10"] = close.pct_change(10) * 100
    df["roc_20"] = close.pct_change(20) * 100

    # CCI
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (tp - tp_sma) / (0.015 * tp_mad + 1e-10)

    # Money Flow Index
    df["mfi_14"] = _compute_mfi(high, low, close, volume, 14)

    # ========== VOLATILITY INDICATORS (8 features) ==========
    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_middle"] = bb_mid
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
    df["bb_pctb"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / (close + 1e-10)
    df["natr_14"] = (df["atr_14"] / close) * 100

    # ========== VOLUME INDICATORS (8 features) ==========
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-10)

    # On-Balance Volume
    obv = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    df["obv"] = obv
    df["obv_sma_10"] = obv.rolling(10).mean()

    # Accumulation/Distribution Line
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    df["ad_line"] = (clv * volume).cumsum()

    # Chaikin Money Flow
    mf_volume = clv * volume
    df["cmf_20"] = mf_volume.rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)

    # VWAP approximation deviation
    vwap = (tp * volume).cumsum() / (volume.cumsum() + 1e-10)
    df["vwap_deviation"] = (close - vwap) / (vwap + 1e-10)

    # Force Index
    df["force_index"] = (close.diff() * volume).ewm(span=13, adjust=False).mean()

    # ========== RETURN-BASED FEATURES (10 features) ==========
    df["ret_1d"] = np.log(close / close.shift(1))
    df["ret_2d"] = np.log(close / close.shift(2))
    df["ret_5d"] = np.log(close / close.shift(5))
    df["ret_10d"] = np.log(close / close.shift(10))
    df["ret_20d"] = np.log(close / close.shift(20))

    df["volatility_5d"] = df["ret_1d"].rolling(5).std()
    df["volatility_10d"] = df["ret_1d"].rolling(10).std()
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()

    # Max drawdown over 10 days
    rolling_max = close.rolling(10).max()
    df["max_drawdown_10d"] = (close - rolling_max) / (rolling_max + 1e-10)

    # Overnight gap
    df["gap_pct"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-10)

    # ========== CANDLESTICK PATTERNS (6 features) ==========
    body = close - open_
    body_abs = body.abs()
    candle_range = high - low + 1e-10
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

    # Doji: body < 10% of range
    df["doji"] = (body_abs / candle_range < 0.1).astype(float)

    # Hammer: lower shadow > 2x body, small upper shadow
    df["hammer"] = (
        (lower_shadow > 2 * body_abs) & (upper_shadow < body_abs * 0.5)
    ).astype(float)

    # Engulfing: current body engulfs previous body
    prev_body = body.shift(1)
    bullish_engulfing = (body > 0) & (prev_body < 0) & (body_abs > prev_body.abs())
    bearish_engulfing = (body < 0) & (prev_body > 0) & (body_abs > prev_body.abs())
    df["engulfing"] = bullish_engulfing.astype(float) - bearish_engulfing.astype(float)

    # Morning Star (simplified): down candle, small body, up candle
    df["morning_star"] = (
        (body.shift(2) < 0)
        & (body_abs.shift(1) < body_abs.shift(2) * 0.3)
        & (body > 0)
        & (close > (open_.shift(2) + close.shift(2)) / 2)
    ).astype(float)

    # Harami: current body is within previous body
    bullish_harami = (
        (prev_body < 0) & (body > 0)
        & (open_ > close.shift(1)) & (close < open_.shift(1))
    )
    bearish_harami = (
        (prev_body > 0) & (body < 0)
        & (open_ < close.shift(1)) & (close > open_.shift(1))
    )
    df["harami"] = bullish_harami.astype(float) - bearish_harami.astype(float)

    # Three White Soldiers / Three Black Crows
    three_up = (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0)
    three_down = (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0)
    df["three_soldiers"] = three_up.astype(float) - three_down.astype(float)

    # ========== SUPPORT/RESISTANCE FEATURES (4 features) ==========
    high_52w = high.rolling(252).max()
    low_52w = low.rolling(252).min()
    df["dist_to_52w_high"] = (close - high_52w) / (high_52w + 1e-10)
    df["dist_to_52w_low"] = (close - low_52w) / (low_52w + 1e-10)

    # Pivot Points
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    s1 = 2 * pivot - high.shift(1)
    r1 = 2 * pivot - low.shift(1)
    df["pivot_support_dist"] = (close - s1) / (close + 1e-10)
    df["pivot_resist_dist"] = (close - r1) / (close + 1e-10)

    # Drop raw indicator columns we don't need as features
    cols_to_drop = [
        "sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50",
        "bb_upper", "bb_middle", "bb_lower", "volume_sma_20",
        "obv_sma_10", "obv",
    ]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)

    return df


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_mfi(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, period: int
) -> pd.Series:
    """Compute Money Flow Index."""
    tp = (high + low + close) / 3
    raw_money_flow = tp * volume

    pos_flow = pd.Series(0.0, index=close.index)
    neg_flow = pd.Series(0.0, index=close.index)

    tp_diff = tp.diff()
    pos_flow[tp_diff > 0] = raw_money_flow[tp_diff > 0]
    neg_flow[tp_diff < 0] = raw_money_flow[tp_diff < 0]

    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()

    mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))
    return mfi


def get_technical_feature_names() -> list[str]:
    """Return list of technical feature column names."""
    return [
        # Trend
        "close_vs_sma20", "close_vs_sma50",
        # Momentum
        "rsi_14", "rsi_7", "macd_line", "macd_signal", "macd_histogram",
        "stoch_k", "stoch_d", "williams_r", "roc_10", "roc_20", "cci_20", "mfi_14",
        # Volatility
        "bb_width", "bb_pctb", "atr_14", "atr_pct", "natr_14",
        # Volume
        "volume_ratio", "ad_line", "cmf_20", "vwap_deviation", "force_index",
        # Returns
        "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "max_drawdown_10d", "gap_pct",
        # Candlesticks
        "doji", "hammer", "engulfing", "morning_star", "harami", "three_soldiers",
        # Support/Resistance
        "dist_to_52w_high", "dist_to_52w_low", "pivot_support_dist", "pivot_resist_dist",
    ]
