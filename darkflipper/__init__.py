"""Public API for darkflipper."""

from .fom import covariance_matrix, polygon
from .fob import chi2_bias, parameter_bias
from .pte import (
    effect_size,
    mahalanobis_distance,
    pvalue,
    theoretical_prediction,
)

__all__ = [
    "polygon",
    "covariance_matrix",
    "parameter_bias",
    "chi2_bias",
    "pvalue",
    "mahalanobis_distance",
    "effect_size",
    "theoretical_prediction",
]
