"""
Core manufacturing optimization implementation.
"""

import numpy as np
import pandas as pd
import json
from typing import Tuple

from .optimization_base import OptimizerBase
from .optimization_types import (
    PredictionDict,
    MetricsDict,
    OptimizationResults,
    ModelConfig,
)


class ManufacturingOptimizer(OptimizerBase):
    """Implements AI-driven optimization for PV manufacturing processes."""

    def prepare_manufacturing_data(
        self, production_data: pd.DataFrame, quality_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and combine manufacturing data for training.

        Args:
            production_data: DataFrame containing production metrics
            quality_data: DataFrame containing quality metrics

        Returns:
            Tuple containing features array and targets array
        """
        self.logger.info("Preparing manufacturing data for optimization")

        try:
            # Combine datasets
            features = pd.merge(
                production_data,
                quality_data,
                on="batch_id",
                suffixes=("_prod", "_qual"),
            )

            # Save feature data configuration
            feature_data = {
                "features": self.config["feature_columns"],
                "target": self.config["target_column"],
            }
            with open(self.results_dir / "feature_data.json", "w") as f:
                json.dump(feature_data, f, indent=2)

            # Extract features and target
            X = features[self.config["feature_columns"]].to_numpy()
            y = features[self.config["target_column"]].to_numpy().reshape(-1, 1)

            # Scale data
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y)

            self.logger.info(f"Prepared {len(X)} samples for training")
            return X_scaled, y_scaled

        except Exception as e:
            self.logger.error(f"Error preparing manufacturing data: {str(e)}")
            raise
