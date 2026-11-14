"""Always-unbiased, uncertainty-adaptive control-variate ESCHER."""

from .estimator import (
    ControlVariateEstimate,
    control_variate_advantage,
    residual_adaptive_sampling_policy,
    variance_optimal_beta,
)
from .solver import UnbiasedControlVariateEscher
from .grouped_wide_solver import GroupedWideUnbiasedControlVariateEscher
from .hulhe_structured_solver import StructuredHULHEGroupedWideUCVEscher

__all__ = [
    "ControlVariateEstimate",
    "GroupedWideUnbiasedControlVariateEscher",
    "StructuredHULHEGroupedWideUCVEscher",
    "UnbiasedControlVariateEscher",
    "control_variate_advantage",
    "residual_adaptive_sampling_policy",
    "variance_optimal_beta",
]
