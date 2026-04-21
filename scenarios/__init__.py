"""
Scenario Lab — cause-effect reasoning and scenario-based trade ideation
for the Nifty 50 universe.

Public API:
    from scenarios.drivers import DRIVERS, get_driver
    from scenarios.sensitivity import build_sensitivity_matrix
    from scenarios.engine import run_scenario
    from scenarios.validator import validate_hypothesis
"""

from scenarios.drivers import DRIVERS, get_driver, Driver
from scenarios.sensitivity import build_sensitivity_matrix, SensitivityMatrix
from scenarios.engine import run_scenario, TradeIdea, ScenarioResult
from scenarios.validator import validate_hypothesis, HypothesisResult

__all__ = [
    "DRIVERS",
    "get_driver",
    "Driver",
    "build_sensitivity_matrix",
    "SensitivityMatrix",
    "run_scenario",
    "TradeIdea",
    "ScenarioResult",
    "validate_hypothesis",
    "HypothesisResult",
]
