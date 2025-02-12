"""Process optimization implementation.

This module implements the optimization logic for manufacturing processes,
using the trained model to suggest optimal parameters.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from .types import OptimizationResults
from ...utils.logging_config import setup_logger
from ...config.project_paths import project_paths
from .model import ManufacturingModel


class ProcessOptimizer:
    def __init__(self, model=None):
        self.logger = setup_logger("process_optimizer")
        self.model = model or ManufacturingModel()

        # Setup paths
        self.run_dir = project_paths.get_run_directory()
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

    def optimize_process_parameters(
        self,
        current_params: Dict[str, float],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """Optimize manufacturing process parameters."""
        try:
            if not self.model.is_trained:
                raise ValueError("Model must be trained before optimization")

            # Convert parameters to feature array
            param_array = np.array([list(current_params.values())])
            scaled_params = self.model.feature_scaler.transform(param_array)

            # Generate parameter variations
            n_variations = 1000
            param_variations = np.random.normal(
                loc=scaled_params,
                scale=0.1,
                size=(n_variations, scaled_params.shape[1]),
            )

            # Predict outcomes for all variations
            predicted_outputs = self.model.model.predict(param_variations)
            best_idx = np.argmax(predicted_outputs)

            # Transform back to original scale
            optimized_params_array = self.model.feature_scaler.inverse_transform(
                param_variations[best_idx].reshape(1, -1)
            )[0]

            # Create dictionary and apply constraints
            optimized_params = {}
            for i, (param_name, value) in enumerate(
                zip(current_params.keys(), optimized_params_array)
            ):
                if constraints and param_name in constraints:
                    min_val, max_val = constraints[param_name]
                    value = np.clip(value, min_val, max_val)
                optimized_params[param_name] = float(value)

            self.logger.info("Parameter optimization completed successfully")
            return optimized_params

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            raise

    def analyze_optimization_potential(
        self, production_data: pd.DataFrame, quality_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze potential optimizations based on historical data."""
        try:
            # Train model if not already trained
            if not self.model.is_trained:
                self.model.train_optimization_model(production_data, quality_data)

            # Get average current parameters
            current_params = {
                "input_amount": production_data["input_amount"].mean(),
                "energy_used": production_data["energy_used"].mean(),
                "cycle_time": production_data["cycle_time"].mean(),
                "efficiency": quality_data["efficiency"].mean(),
                "defect_rate": quality_data["defect_rate"].mean(),
                "thickness_uniformity": quality_data["thickness_uniformity"].mean(),
            }

            # Get optimized parameters
            optimized_params = self.optimize_process_parameters(current_params)

            # Calculate potential improvements
            improvements = {
                param: (
                    (optimized_params[param] - current_params[param])
                    / current_params[param]
                    * 100
                )
                for param in current_params
            }

            self.logger.info(
                "Optimization potential analysis:\n"
                + "\n".join(f"{k}: {v:.1f}%" for k, v in improvements.items())
            )

            return improvements

        except Exception as e:
            self.logger.error(f"Error in optimization potential analysis: {str(e)}")
            raise
