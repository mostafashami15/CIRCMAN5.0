# src/circman5/manufacturing/optimization/types.py

from typing import (
    Mapping,
    TypedDict,
    Dict,
    List,
    Tuple,
    Optional,
    TypeVar,
    Union,
    TYPE_CHECKING,
)
import numpy.typing as npt
import numpy as np

if TYPE_CHECKING:
    from .model import ManufacturingModel

# Type variable for ManufacturingModel
ManufacturingModelType = TypeVar("ManufacturingModelType", bound="ManufacturingModel")


class PredictionDict(TypedDict):
    """Type definition for prediction outputs."""

    predicted_output: float
    predicted_quality: float
    confidence_score: float


class MetricsDict(TypedDict):
    """Type definition for model performance metrics."""

    mse: float  # Mean squared error
    rmse: float  # Root mean squared error
    mae: float  # Mean absolute error
    r2: float  # R-squared score
    cv_r2_mean: float  # Cross-validation R2 mean
    cv_r2_std: float  # Cross-validation R2 standard deviation
    feature_importance: Mapping[str, Union[float, str]]  # Feature importance scores


class OptimizationResults(TypedDict):
    """Type definition for optimization results."""

    original_params: Dict[str, float]
    optimized_params: Dict[str, float]
    improvement: Dict[
        str, float
    ]  # Changed from float to Dict for per-parameter improvements
    optimization_success: bool
    optimization_message: str
    iterations: int
    objective_value: float


class ModelConfig(TypedDict):
    """Type definition for model configuration."""

    feature_columns: List[str]
    target_column: str
    test_size: float
    random_state: int
    cv_folds: int
    model_params: Dict[str, Union[int, float, str]]
