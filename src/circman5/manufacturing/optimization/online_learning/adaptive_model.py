# src/circman5/manufacturing/optimization/online_learning/adaptive_model.py

import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, cast
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from ....adapters.services.constants_service import ConstantsService
from ....utils.results_manager import results_manager
from ....utils.logging_config import setup_logger
from ..advanced_models.deep_learning import DeepLearningModel
from ..advanced_models.ensemble import EnsembleModel


class AdaptiveModel:
    """Adaptive model that updates in response to new data."""

    def __init__(self, base_model_type: str = "ensemble"):
        """
        Initialize adaptive model.

        Args:
            base_model_type: Type of base model to use (ensemble or deep_learning)
        """
        self.logger = setup_logger("adaptive_model")
        self.constants = ConstantsService()
        self.config = self.constants.get_optimization_config()
        self.online_config = self.config.get("ONLINE_LEARNING", {})

        # Initialize configuration parameters
        self.window_size = self.online_config.get("window_size", 100)
        self.learning_rate = self.online_config.get("learning_rate", 0.01)
        self.update_frequency = self.online_config.get("update_frequency", 10)
        self.forgetting_factor = self.online_config.get("forgetting_factor", 0.95)
        self.max_model_age = self.online_config.get("max_model_age", 24)  # in hours
        self.persistence_interval = self.online_config.get(
            "model_persistence_interval", 60
        )  # in minutes

        # Initialize model
        self.base_model_type = base_model_type
        if base_model_type == "deep_learning":
            self.model = DeepLearningModel()
        else:  # ensemble
            self.model = EnsembleModel()

        # Initialize data buffers
        self.data_buffer_X = []
        self.data_buffer_y = []
        self.buffer_weights = []

        # Initialize tracking variables
        self.updates_counter = 0
        self.last_update_time = datetime.now()
        self.last_persistence_time = datetime.now()
        self.model_created_time = datetime.now()
        self.is_initialized = False

        self.logger.info(
            f"Initialized adaptive model with {base_model_type} base model"
        )

    def add_data_point(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        weight: float = 1.0,
    ) -> bool:
        """
        Add a new data point to the buffer and potentially update the model.

        Args:
            X: Feature vector (single sample)
            y: Target value
            weight: Importance weight for this sample

        Returns:
            bool: True if model was updated, False otherwise
        """
        try:
            # Convert pandas to numpy if needed
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            if isinstance(y, pd.DataFrame):
                y = y.to_numpy()

            # Reshape if needed
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            if len(y.shape) == 1:
                y = y.reshape(1, -1)

            # Add to buffer
            self.data_buffer_X.append(X)
            self.data_buffer_y.append(y)
            self.buffer_weights.append(weight)

            # Apply forgetting factor to existing weights
            self.buffer_weights = [
                w * self.forgetting_factor for w in self.buffer_weights
            ]

            # Trim buffer if it exceeds window size
            if len(self.data_buffer_X) > self.window_size:
                self.data_buffer_X = self.data_buffer_X[-self.window_size :]
                self.data_buffer_y = self.data_buffer_y[-self.window_size :]
                self.buffer_weights = self.buffer_weights[-self.window_size :]

            # Check if update is needed
            self.updates_counter += 1
            current_time = datetime.now()

            update_needed = (
                self.updates_counter >= self.update_frequency
                or (current_time - self.last_update_time).total_seconds()
                >= 3600  # At least hourly
            )

            model_updated = False
            if update_needed and len(self.data_buffer_X) >= self.update_frequency:
                model_updated = self._update_model()
                self.last_update_time = current_time
                self.updates_counter = 0

            # Check if persistence is needed
            persistence_needed = (
                current_time - self.last_persistence_time
            ).total_seconds() >= (self.persistence_interval * 60)

            if persistence_needed:
                self._persist_model()
                self.last_persistence_time = current_time

            return model_updated

        except Exception as e:
            self.logger.error(f"Error adding data point: {str(e)}")
            return False

    def _update_model(self) -> bool:
        """
        Update the model with current buffer data.

        Returns:
            bool: True if update was successful
        """
        try:
            self.logger.info(
                f"Updating adaptive model with {len(self.data_buffer_X)} samples"
            )

            # Convert buffer to numpy arrays
            X = np.vstack(self.data_buffer_X)
            y = np.vstack(self.data_buffer_y)
            weights = np.array(self.buffer_weights)

            # Check if model needs to be recreated based on age
            current_time = datetime.now()
            model_age_hours = (
                current_time - self.model_created_time
            ).total_seconds() / 3600

            if model_age_hours > self.max_model_age or not self.is_initialized:
                self.logger.info(f"Recreating model (age: {model_age_hours:.1f} hours)")
                if self.base_model_type == "deep_learning":
                    self.model = DeepLearningModel()
                else:  # ensemble
                    self.model = EnsembleModel()
                self.model_created_time = current_time
                self.is_initialized = True

            # Train the model
            # In actual implementation with weight support:
            # self.model.train(X, y, sample_weight=weights)

            # For placeholder:
            self.model.train(X, y)

            self.logger.info(f"Model updated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            return False

    def _persist_model(self) -> None:
        """Save the current model state to disk."""
        try:
            self.logger.info("Persisting adaptive model")

            if not self.is_initialized:
                self.logger.warning("Model not initialized, skipping persistence")
                return

            # Save model using results_manager
            model_dir = results_manager.get_path("digital_twin") / "models" / "adaptive"
            model_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save model
            model_path = model_dir / f"adaptive_model_{timestamp}.pkl"
            self.model.save_model(model_path)

            # Save metadata
            metadata = {
                "base_model_type": self.base_model_type,
                "window_size": self.window_size,
                "learning_rate": self.learning_rate,
                "update_frequency": self.update_frequency,
                "buffer_size": len(self.data_buffer_X),
                "model_age_hours": (
                    datetime.now() - self.model_created_time
                ).total_seconds()
                / 3600,
                "timestamp": timestamp,
            }

            metadata_path = model_dir / f"metadata_{timestamp}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Model persisted with timestamp {timestamp}")

        except Exception as e:
            self.logger.error(f"Error persisting model: {str(e)}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate predictions using the adaptive model.

        Args:
            X: Input features

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_initialized:
            raise ValueError("Model not initialized")

        return self.model.predict(X)

    def evaluate(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Evaluate adaptive model performance.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.is_initialized:
            raise ValueError("Model not initialized")

        metrics = self.model.evaluate(X, y)

        # Add adaptive model specific metrics
        metrics["buffer_size"] = len(self.data_buffer_X)
        metrics["model_age_hours"] = (
            datetime.now() - self.model_created_time
        ).total_seconds() / 3600
        metrics["updates_count"] = self.updates_counter

        return metrics
