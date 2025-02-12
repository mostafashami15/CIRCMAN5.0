"""Type definitions for manufacturing optimization module.

This module defines TypedDict classes for type checking optimization-related
data structures including predictions, metrics, and configuration.
"""

from typing import TypedDict, Dict, List, Tuple, Optional
import numpy.typing as npt
import numpy as np


class PredictionDict(TypedDict):
    """Type definition for prediction outputs."""

    predicted_output: float
    predicted_quality: float


class MetricsDict(TypedDict):
    """Type definition for model performance metrics."""

    mse: float
    r2: float


class OptimizationResults(TypedDict):
    """Type definition for optimization results."""

    original_params: Dict[str, float]
    optimized_params: Dict[str, float]
    improvement: float


class ModelConfig(TypedDict):
    """Type definition for model configuration."""

    feature_columns: List[str]
    target_column: str
    test_size: float
    random_state: int
