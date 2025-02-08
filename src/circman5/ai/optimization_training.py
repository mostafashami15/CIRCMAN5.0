"""
Training and optimization implementation for manufacturing optimization.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Optional

from .optimization_core import ManufacturingOptimizer
from .optimization_types import MetricsDict, OptimizationResults


class ManufacturingOptimizerTraining(ManufacturingOptimizer):
    def train_optimization_models(self, X: np.ndarray, y: np.ndarray) -> MetricsDict:
        """
        Train optimization models on prepared manufacturing data.

        Args:
            X: Scaled feature array
            y: Scaled target array

        Returns:
            Dict containing model performance metrics
        """
        self.logger.info("Training optimization models")

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config["test_size"],
                random_state=self.config["random_state"],
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

            # Save metrics
            with open(self.results_dir / "optimization_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Create and save performance visualization
            self._plot_performance(y_test, y_pred)

            self.is_trained = True
            self.logger.info(f"Models trained successfully. Metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    def optimize_process_parameters(
        self,
        current_params: Dict[str, float],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        Optimize manufacturing process parameters with constraints.

        Args:
            current_params: Current manufacturing parameters
            constraints: Optional parameter constraints as (min, max) tuples

        Returns:
            Dict containing optimized parameters
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before optimization")

        try:
            # Convert parameters to feature array
            param_array = np.array([list(current_params.values())])
            scaled_params = self.feature_scaler.transform(param_array)

            # Generate parameter variations
            n_variations = 1000
            param_variations = np.random.normal(
                loc=scaled_params,
                scale=0.1,
                size=(n_variations, scaled_params.shape[1]),
            )

            # Apply constraints before scaling back
            if constraints:
                # Scale constraints to match the scaled parameter space
                scaled_constraints = {}
                for i, (param_name, (min_val, max_val)) in enumerate(
                    constraints.items()
                ):
                    min_scaled = self.feature_scaler.transform([[min_val]])[0][0]
                    max_scaled = self.feature_scaler.transform([[max_val]])[0][0]
                    param_variations[:, i] = np.clip(
                        param_variations[:, i], min_scaled, max_scaled
                    )

            # Predict outcomes for all variations
            predicted_outputs = self.efficiency_model.predict(param_variations)
            best_idx = np.argmax(predicted_outputs)

            # Transform back to original scale
            optimized_params_array = self.feature_scaler.inverse_transform(
                param_variations[best_idx].reshape(1, -1)
            )[0]

            # Create dictionary and apply constraints one final time
            optimized_params = {}
            for param_name, value in zip(current_params.keys(), optimized_params_array):
                if constraints and param_name in constraints:
                    min_val, max_val = constraints[param_name]
                    value = np.clip(value, min_val, max_val)
                optimized_params[param_name] = float(value)

            # Save optimization results
            results = {
                "original_params": current_params,
                "optimized_params": optimized_params,
                "improvement": float(
                    predicted_outputs[best_idx] - predicted_outputs[0]
                ),
            }

            with open(self.results_dir / "optimization_results.json", "w") as f:
                json.dump(results, f, indent=2)

            self.logger.info("Parameter optimization completed successfully")
            return optimized_params

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            raise

    def _plot_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Create and save model performance visualization."""
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot(
                [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
            )
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Model Performance: Predicted vs Actual")

            # Save plot
            plt.savefig(
                self.results_dir / "model_performance.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            self.logger.error(f"Error creating performance plot: {str(e)}")
            raise
