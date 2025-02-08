"""
Prediction functionality for manufacturing optimization.
"""

import numpy as np
import json
from typing import Dict

from .optimization_training import ManufacturingOptimizerTraining
from .optimization_types import PredictionDict


class ManufacturingOptimizer(ManufacturingOptimizerTraining):
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

        try:
            # Prepare input
            param_array = np.array([list(process_params.values())])
            scaled_params = self.feature_scaler.transform(param_array)

            # Make predictions
            efficiency_pred = self.efficiency_model.predict(scaled_params)
            quality_pred = self.quality_model.predict(scaled_params)

            # Scale predictions back
            efficiency_actual = float(
                self.target_scaler.inverse_transform(efficiency_pred.reshape(-1, 1))[0][
                    0
                ]
            )

            quality_actual = float(
                self.target_scaler.inverse_transform(quality_pred.reshape(-1, 1))[0][0]
            )

            predictions: PredictionDict = {
                "predicted_output": efficiency_actual,
                "predicted_quality": quality_actual,
            }

            # Save predictions
            with open(self.results_dir / "predictions.json", "w") as f:
                json.dump(predictions, f, indent=2)

            self.logger.info(f"Generated predictions: {predictions}")
            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save_model_state(self) -> None:
        """Save current model state and configuration."""
        try:
            # Save configuration
            with open(self.results_dir / "model_config.json", "w") as f:
                json.dump(self.config, f, indent=2)

            self.logger.info("Model state saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving model state: {str(e)}")
            raise
