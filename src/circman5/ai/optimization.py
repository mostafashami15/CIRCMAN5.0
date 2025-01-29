"""
AI-driven optimization engine for PV manufacturing processes.
Implements predictive modeling and process optimization using scikit-learn.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional, Union, TypedDict
import numpy.typing as npt


class PredictionDict(TypedDict):
    predicted_output: float
    predicted_quality: float


class MetricsDict(TypedDict):
    mse: float
    r2: float


class ManufacturingOptimizer:
    """Implements AI-driven optimization for PV manufacturing processes."""

    def __init__(self):
        """Initialize optimization models and scalers."""
        self.efficiency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.quality_model = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_trained = False

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
        # Combine relevant features
        features = pd.merge(
            production_data, quality_data, on="batch_id", suffixes=("_prod", "_qual")
        )

        # Select feature columns
        feature_columns = [
            "input_amount",
            "energy_used",
            "cycle_time",
            "efficiency",
            "defect_rate",
            "thickness_uniformity",
        ]
        target_column = "output_amount"

        # Convert to numpy arrays explicitly
        X = features[feature_columns].to_numpy()
        y = features[target_column].to_numpy()
        y = y[:, np.newaxis]  # Reshape using numpy's alternative

        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)

        return X_scaled, y_scaled

    def train_optimization_models(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> MetricsDict:
        """
        Train optimization models on prepared manufacturing data.

        Args:
            X: Scaled feature array
            y: Scaled target array
            test_size: Proportion of data to use for testing

        Returns:
            Dict containing model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train models
        self.efficiency_model.fit(X_train, y_train.ravel())
        self.quality_model.fit(X_train, y_train.ravel())

        # Evaluate performance
        y_pred = self.efficiency_model.predict(X_test)

        metrics: MetricsDict = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

        self.is_trained = True
        return metrics

    def optimize_process_parameters(
        self,
        current_params: Dict[str, float],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        Optimize manufacturing process parameters.

        Args:
            current_params: Current manufacturing parameters
            constraints: Optional parameter constraints as (min, max) tuples

        Returns:
            Dict containing optimized parameters
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before optimization")

        # Convert parameters to feature array
        param_array = np.array([list(current_params.values())])
        scaled_params = self.feature_scaler.transform(param_array)

        # Generate parameter variations
        n_variations = 1000
        param_variations = np.random.normal(
            loc=scaled_params, scale=0.1, size=(n_variations, scaled_params.shape[1])
        )

        # Apply constraints if provided
        if constraints:
            for i, (param_name, (min_val, max_val)) in enumerate(constraints.items()):
                param_variations[:, i] = np.clip(
                    param_variations[:, i], min_val, max_val
                )

        # Predict outcomes
        predicted_outputs = self.efficiency_model.predict(param_variations)

        # Select best parameters
        best_idx = np.argmax(predicted_outputs)
        optimized_params = self.feature_scaler.inverse_transform(
            param_variations[best_idx].reshape(1, -1)
        )[0]

        # Convert numpy values to Python floats
        return {k: float(v) for k, v in zip(current_params.keys(), optimized_params)}

    def predict_manufacturing_outcomes(
        self, process_params: Dict[str, float]
    ) -> PredictionDict:
        """
        Predict manufacturing outcomes for given parameters.

        Args:
            process_params: Manufacturing process parameters

        Returns:
            Dict containing predicted outcomes
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")

        # Prepare input
        param_array = np.array([list(process_params.values())])
        scaled_params = self.feature_scaler.transform(param_array)

        # Make predictions
        efficiency_pred = self.efficiency_model.predict(scaled_params)
        quality_pred = self.quality_model.predict(scaled_params)

        # Scale predictions back
        efficiency_actual = self.target_scaler.inverse_transform(
            efficiency_pred.reshape(-1, 1)
        )[0][0]
        quality_actual = self.target_scaler.inverse_transform(
            quality_pred.reshape(-1, 1)
        )[0][0]

        predictions: PredictionDict = {
            "predicted_output": float(efficiency_actual),
            "predicted_quality": float(quality_actual),
        }
        return predictions
