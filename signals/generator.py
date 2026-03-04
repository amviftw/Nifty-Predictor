"""
Signal generation module.
Converts ensemble model probability outputs into actionable BUY/SELL/HOLD signals.
"""

import numpy as np
from loguru import logger

from config.settings import SETTINGS


class SignalGenerator:
    """Convert model probabilities into trading signals."""

    def __init__(
        self,
        confidence_threshold: float = None,
        strong_confidence: float = None,
        min_edge: float = None,
    ):
        self.confidence_threshold = confidence_threshold or SETTINGS.CONFIDENCE_THRESHOLD
        self.strong_confidence = strong_confidence or SETTINGS.STRONG_CONFIDENCE
        self.min_edge = min_edge or SETTINGS.MIN_EDGE

    def generate_signal(self, proba: np.ndarray) -> dict:
        """
        Generate a trading signal from class probabilities.

        Args:
            proba: array of [prob_down, prob_flat, prob_up]

        Returns:
            dict with signal, confidence, strength, and probabilities
        """
        prob_down, prob_flat, prob_up = proba

        # Find dominant class
        sorted_probs = sorted(enumerate(proba), key=lambda x: -x[1])
        dominant_class, dominant_prob = sorted_probs[0]
        _, second_prob = sorted_probs[1]
        edge = dominant_prob - second_prob

        # Default: HOLD
        signal = "HOLD"
        confidence = prob_flat
        strength = "WEAK"

        # BUY signal
        if (
            dominant_class == 2
            and prob_up >= self.confidence_threshold
            and edge >= self.min_edge
        ):
            signal = "BUY"
            confidence = prob_up
            strength = "STRONG" if prob_up >= self.strong_confidence else "MODERATE"

        # SELL signal
        elif (
            dominant_class == 0
            and prob_down >= self.confidence_threshold
            and edge >= self.min_edge
        ):
            signal = "SELL"
            confidence = prob_down
            strength = "STRONG" if prob_down >= self.strong_confidence else "MODERATE"

        return {
            "signal": signal,
            "confidence": round(float(confidence), 4),
            "strength": strength,
            "prob_up": round(float(prob_up), 4),
            "prob_flat": round(float(prob_flat), 4),
            "prob_down": round(float(prob_down), 4),
        }

    def generate_all_signals(
        self,
        symbols: list[str],
        probas: np.ndarray,
        sectors: dict[str, str] = None,
    ) -> list[dict]:
        """
        Generate signals for all stocks.

        Args:
            symbols: list of stock symbols
            probas: ndarray of shape (n_stocks, 3)
            sectors: optional dict mapping symbol -> sector

        Returns:
            list of signal dicts (one per stock)
        """
        signals = []
        for i, symbol in enumerate(symbols):
            sig = self.generate_signal(probas[i])
            sig["symbol"] = symbol
            if sectors:
                sig["sector"] = sectors.get(symbol, "Unknown")
            signals.append(sig)

        # Sort by confidence descending
        signals.sort(key=lambda s: s["confidence"], reverse=True)

        buy_count = sum(1 for s in signals if s["signal"] == "BUY")
        sell_count = sum(1 for s in signals if s["signal"] == "SELL")
        logger.info(
            f"Generated signals: {buy_count} BUY, {sell_count} SELL, "
            f"{len(signals) - buy_count - sell_count} HOLD"
        )

        return signals


def rank_and_select_signals(
    signals: list[dict], max_positions: int = None
) -> list[dict]:
    """
    Rank signals by confidence and select top positions.
    Applies sector diversification (max 3 per sector).

    Args:
        signals: list of signal dicts from generate_all_signals
        max_positions: max number of actionable signals

    Returns:
        list of selected signals
    """
    if max_positions is None:
        max_positions = SETTINGS.MAX_POSITIONS

    actionable = [s for s in signals if s["signal"] != "HOLD"]
    actionable.sort(key=lambda s: s["confidence"], reverse=True)

    selected = []
    sector_count = {}

    for s in actionable:
        sector = s.get("sector", "Unknown")
        if sector_count.get(sector, 0) < 3 and len(selected) < max_positions:
            selected.append(s)
            sector_count[sector] = sector_count.get(sector, 0) + 1

    return selected
