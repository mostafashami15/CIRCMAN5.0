# src/circman5/manufacturing/optimization/online_learning/real_time_trainer.py

import logging
import threading
import time
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from ....adapters.services.constants_service import ConstantsService
from ....utils.results_manager import results_manager
from ....utils.logging_config import setup_logger
from .adaptive_model import AdaptiveModel


class RealTimeModelTrainer:
    """
    Real-time model trainer that integrates with Digital Twin for
    continuous learning from streaming data.
    """

    def __init__(self, data_source_callback: Optional[Callable] = None):
        """
        Initialize real-time trainer.

        Args:
            data_source_callback: Optional callback function to retrieve new data
        """
        self.logger = setup_logger("real_time_trainer")
        self.constants = ConstantsService()
        self.config = self.constants.get_optimization_config()
        self.online_config = self.config.get("ONLINE_LEARNING", {})

        # Store the data source callback
        self.data_source_callback = data_source_callback

        # Initialize the adaptive model
        self.adaptive_model = AdaptiveModel()

        # Initialize the training thread
        self.training_thread = None
        self.stop_training = threading.Event()

        # Initialize tracking variables
        self.processed_samples = 0
        self.start_time = None
        self.metrics_history = []

        self.logger.info("Initialized real-time model trainer")

    def start(self, interval_seconds: int = 10) -> None:
        """
        Start the real-time training loop.

        Args:
            interval_seconds: Seconds between training iterations
        """
        if self.training_thread is not None and self.training_thread.is_alive():
            self.logger.warning("Training already in progress")
            return

        self.logger.info(
            f"Starting real-time training with {interval_seconds}s interval"
        )
        self.stop_training.clear()
        self.start_time = datetime.now()

        # Create and start training thread
        self.training_thread = threading.Thread(
            target=self._training_loop, args=(interval_seconds,), daemon=True
        )
        self.training_thread.start()

    def stop(self) -> None:
        """Stop the training loop."""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.logger.warning("No training in progress")
            return

        self.logger.info("Stopping real-time training")
        self.stop_training.set()
        self.training_thread.join(timeout=30)

        if self.training_thread.is_alive():
            self.logger.warning("Training thread did not stop gracefully")
        else:
            self.logger.info("Training thread stopped")

        # Save final metrics
        self._save_metrics()

    def _training_loop(self, interval_seconds: int) -> None:
        """
        Main training loop that runs in a separate thread.

        Args:
            interval_seconds: Seconds between training iterations
        """
        while not self.stop_training.is_set():
            try:
                # Get new data if callback is provided
                if self.data_source_callback is not None:
                    data = self.data_source_callback()
                    if data is not None:
                        X, y = data
                        self._process_data(X, y)

                # Record metrics periodically
                if self.processed_samples > 0 and self.processed_samples % 100 == 0:
                    self._record_metrics()

                # Save metrics periodically
                if self.processed_samples > 0 and self.processed_samples % 1000 == 0:
                    self._save_metrics()

            except Exception as e:
                self.logger.error(f"Error in training loop: {str(e)}")

            # Wait for next iteration
            time.sleep(interval_seconds)

    def _process_data(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Process incoming data batches.

        Args:
            X: Input features
            y: Target values
        """
        try:
            # Convert pandas to numpy if needed
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            if isinstance(y, pd.DataFrame):
                y = y.to_numpy()

            # Add each data point to adaptive model
            batch_size = len(X) if hasattr(X, "__len__") else 1

            if batch_size > 1:
                for i in range(batch_size):
                    X_i = (
                        X[i].reshape(1, -1)
                        if len(X.shape) > 1
                        else np.array([X[i]]).reshape(1, -1)
                    )
                    y_i = (
                        y[i].reshape(1, -1)
                        if len(y.shape) > 1
                        else np.array([y[i]]).reshape(1, -1)
                    )
                    self.adaptive_model.add_data_point(X_i, y_i)
            else:
                X_reshaped = (
                    X.reshape(1, -1)
                    if len(X.shape) > 1
                    else np.array([X]).reshape(1, -1)
                )
                y_reshaped = (
                    y.reshape(1, -1)
                    if len(y.shape) > 1
                    else np.array([y]).reshape(1, -1)
                )
                self.adaptive_model.add_data_point(X_reshaped, y_reshaped)

            self.processed_samples += batch_size

        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")

    def _record_metrics(self) -> None:
        """Record current performance metrics."""
        try:
            # Calculate metrics
            if self.start_time is not None:
                elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
            else:
                elapsed_seconds = 0
            samples_per_second = (
                self.processed_samples / elapsed_seconds if elapsed_seconds > 0 else 0
            )

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "processed_samples": self.processed_samples,
                "uptime_seconds": elapsed_seconds,
                "samples_per_second": samples_per_second,
                "model_updates": self.adaptive_model.updates_counter,
                "buffer_size": len(self.adaptive_model.data_buffer_X),
                "model_age_hours": (
                    datetime.now() - self.adaptive_model.model_created_time
                ).total_seconds()
                / 3600,
            }

            self.metrics_history.append(metrics)
            self.logger.info(
                f"Training metrics: processed={self.processed_samples}, "
                f"rate={samples_per_second:.1f} samples/sec, "
                f"buffer_size={len(self.adaptive_model.data_buffer_X)}"
            )

        except Exception as e:
            self.logger.error(f"Error recording metrics: {str(e)}")

    def _save_metrics(self) -> None:
        """Save metrics history to disk."""
        try:
            if not self.metrics_history:
                return

            metrics_dir = results_manager.get_path("metrics") / "real_time_training"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = metrics_dir / f"training_metrics_{timestamp}.json"

            with open(metrics_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)

            self.logger.info(f"Saved metrics to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
