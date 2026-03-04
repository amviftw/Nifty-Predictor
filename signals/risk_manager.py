"""
Risk management module.
Applies position sizing and portfolio-level exposure constraints.
"""

from loguru import logger
from config.settings import SETTINGS


class RiskManager:
    """Apply position sizing and risk constraints to signals."""

    def __init__(
        self,
        max_total_positions: int = None,
        max_per_stock_pct: float = None,
        max_sector_pct: float = None,
        max_long_pct: float = None,
        max_short_pct: float = None,
        max_total_exposure_pct: float = None,
    ):
        self.max_total_positions = max_total_positions or SETTINGS.MAX_POSITIONS
        self.max_per_stock_pct = max_per_stock_pct or SETTINGS.MAX_PER_STOCK_PCT
        self.max_sector_pct = max_sector_pct or SETTINGS.MAX_SECTOR_PCT
        self.max_long_pct = max_long_pct or SETTINGS.MAX_LONG_PCT
        self.max_short_pct = max_short_pct or SETTINGS.MAX_SHORT_PCT
        self.max_total_exposure_pct = (
            max_total_exposure_pct or SETTINGS.MAX_TOTAL_EXPOSURE_PCT
        )

    def apply_risk_constraints(self, signals: list[dict]) -> list[dict]:
        """
        Apply position sizing and risk constraints to a list of signals.

        Modifies each signal dict in-place, adding 'position_size_pct'.
        Returns the modified list.
        """
        portfolio = {
            "total_long_pct": 0.0,
            "total_short_pct": 0.0,
            "num_positions": 0,
            "sector_exposure": {},
        }

        for signal in signals:
            signal = self._size_position(signal, portfolio)

            # Update portfolio tracking
            size = signal.get("position_size_pct", 0)
            if size > 0:
                sector = signal.get("sector", "Unknown")
                portfolio["num_positions"] += 1
                portfolio["sector_exposure"][sector] = (
                    portfolio["sector_exposure"].get(sector, 0) + size
                )
                if signal["signal"] == "BUY":
                    portfolio["total_long_pct"] += size
                elif signal["signal"] == "SELL":
                    portfolio["total_short_pct"] += size

        total_exposure = portfolio["total_long_pct"] + portfolio["total_short_pct"]
        logger.info(
            f"Portfolio: {portfolio['num_positions']} positions, "
            f"long={portfolio['total_long_pct']:.1%}, "
            f"short={portfolio['total_short_pct']:.1%}, "
            f"total={total_exposure:.1%}"
        )

        return signals

    def _size_position(self, signal: dict, portfolio: dict) -> dict:
        """Determine position size for a single signal."""
        if signal["signal"] == "HOLD" or signal["strength"] == "WEAK":
            signal["position_size_pct"] = 0.0
            return signal

        # Base size from strength
        if signal["strength"] == "STRONG":
            base_size = self.max_per_stock_pct  # 10%
        else:  # MODERATE
            base_size = self.max_per_stock_pct * 0.6  # 6%

        # Scale by confidence
        conf = signal["confidence"]
        confidence_scale = (conf - SETTINGS.CONFIDENCE_THRESHOLD) / (
            1.0 - SETTINGS.CONFIDENCE_THRESHOLD + 1e-10
        )
        position_size = base_size * max(0.5, min(1.0, confidence_scale))

        # Check constraints
        sector = signal.get("sector", "Unknown")

        # Max positions
        if portfolio["num_positions"] >= self.max_total_positions:
            signal["position_size_pct"] = 0.0
            signal["reject_reason"] = "max_positions_reached"
            return signal

        # Sector limit
        sector_used = portfolio["sector_exposure"].get(sector, 0)
        if sector_used + position_size > self.max_sector_pct:
            position_size = max(0, self.max_sector_pct - sector_used)

        # Directional limits
        if signal["signal"] == "BUY":
            remaining_long = self.max_long_pct - portfolio["total_long_pct"]
            position_size = min(position_size, max(0, remaining_long))
        elif signal["signal"] == "SELL":
            remaining_short = self.max_short_pct - portfolio["total_short_pct"]
            position_size = min(position_size, max(0, remaining_short))

        # Total exposure limit
        total_used = portfolio["total_long_pct"] + portfolio["total_short_pct"]
        remaining_total = self.max_total_exposure_pct - total_used
        position_size = min(position_size, max(0, remaining_total))

        signal["position_size_pct"] = round(position_size, 4)
        return signal
