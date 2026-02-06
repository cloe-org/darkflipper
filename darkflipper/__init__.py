"""Public API for darkflipper."""

from .fom import covariance_matrix, polygon
from .fob import chi2_bias, parameter_bias
from .pte import (
    critical_value,
    effect_size,
    mahalanobis_distance,
    pvalue,
    test_statistic,
)

__all__ = [
    "polygon",
    "covariance_matrix",
    "parameter_bias",
    "chi2_bias",
    "pvalue",
    "mahalanobis_distance",
    "test_statistic",
    "critical_value",
    "effect_size",
]
