"""Manufacturing optimization model implementation.

This module implements the core ML model for optimizing manufacturing processes,
including training, prediction, and evaluation functionality.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple

from .types import PredictionDict, MetricsDict
from ...utils.logging_config import setup_logger
from ...config.project_paths import project_paths
from ...config.constants import MODEL_CONFIG, OPTIMIZATION_TARGETS


class ManufacturingModel:
    def __init__(self):
        self.logger = setup_logger("manufacturing_model")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_trained = False
        self.config = MODEL_CONFIG

        # Setup paths
        self.run_dir = project_paths.get_run_directory()
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Configuration
        self.config = {
            "feature_columns": [
                "input_amount",
                "energy_used",
                "cycle_time",
                "efficiency",
                "defect_rate",
                "thickness_uniformity",
            ],
            "target_column": "output_amount",
            "test_size": 0.2,
            "random_state": 42,
        }

    def train_optimization_model(
        self, production_data: pd.DataFrame, quality_data: pd.DataFrame
    ) -> MetricsDict:
        """Train the AI optimization model with current manufacturing data."""
        if production_data.empty or quality_data.empty:
            raise ValueError("No data available for training optimization model")

        try:
            # Prepare data
            features = pd.merge(
                production_data,
                quality_data,
                on="batch_id",
                suffixes=("_prod", "_qual"),
            )

            X = features[self.config["feature_columns"]].to_numpy()
            y = features[self.config["target_column"]].to_numpy().reshape(-1, 1)

            # Scale data
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y_scaled,
                test_size=self.config["test_size"],
                random_state=self.config["random_state"],
            )

            # Train model
            self.model.fit(X_train, y_train.ravel())

            # Evaluate performance
            y_pred = self.model.predict(X_test)

            metrics: MetricsDict = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }

            self.is_trained = True
            self.logger.info(f"Model trained successfully. Metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def predict_batch_outcomes(
        self, process_params: Dict[str, float]
    ) -> PredictionDict:
        """Predict manufacturing outcomes for given parameters."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Prepare input
            param_array = np.array([list(process_params.values())])
            scaled_params = self.feature_scaler.transform(param_array)

            # Make predictions
            prediction_scaled = self.model.predict(scaled_params)

            # Scale prediction back
            prediction_actual = float(
                self.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[
                    0
                ][0]
            )

            predictions: PredictionDict = {
                "predicted_output": prediction_actual,
                "predicted_quality": prediction_actual * 0.9,  # Example quality metric
            }

            self.logger.info(f"Generated predictions: {predictions}")
            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
