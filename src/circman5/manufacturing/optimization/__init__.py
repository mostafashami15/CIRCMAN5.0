"""Manufacturing process optimization module.

This package provides functionality for optimizing manufacturing processes using ML.

Components:
- types: Type definitions for optimization data structures
- model: Core ML model implementation
- optimizer: Process optimization logic
"""

from .types import PredictionDict, MetricsDict, OptimizationResults
from .model import ManufacturingModel
from .optimizer import ProcessOptimizer

__all__ = [
    "PredictionDict",
    "MetricsDict",
    "OptimizationResults",
    "ManufacturingModel",
    "ProcessOptimizer",
]
