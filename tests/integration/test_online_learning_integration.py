# tests/integration/test_online_learning_integration.py

import pytest
import numpy as np
import pandas as pd
import time
from circman5.manufacturing.optimization.online_learning.adaptive_model import (
    AdaptiveModel,
)
from circman5.manufacturing.optimization.online_learning.real_time_trainer import (
    RealTimeModelTrainer,
)
from circman5.manufacturing.optimization.advanced_models.deep_learning import (
    DeepLearningModel,
)
from circman5.manufacturing.optimization.advanced_models.ensemble import EnsembleModel
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture
def time_series_data():
    """Generate time series data for testing online learning."""
    data_gen = ManufacturingDataGenerator()
    try:
        # Check if the enhanced generator method exists
        if hasattr(data_gen, "generate_time_series_data"):
            return data_gen.generate_time_series_data(days=5, interval_minutes=30)
    except Exception:
        pass

    # Fallback to regular data generation
    production_data = data_gen.generate_production_data()
    production_data["timestamp"] = pd.date_range(
        start="2025-01-01", periods=100, freq="30min"
    )
    return production_data


def test_adaptive_model_incremental_learning(time_series_data):
    """Test adaptive model with incremental data stream."""
    # Create model
    adaptive_model = AdaptiveModel(base_model_type="ensemble")

    # Prepare features and targets
    features = time_series_data[["input_amount", "energy_used", "cycle_time"]]
    targets = time_series_data["output_amount"]

    # Initial baseline metrics
    initial_metrics = None

    # Simulate data stream - feed data in batches
    batch_size = 10
    for i in range(0, len(features), batch_size):
        end_idx = min(i + batch_size, len(features))
        batch_X = features.iloc[i:end_idx]
        batch_y = targets.iloc[i:end_idx]

        # Feed data points one by one
        for j in range(len(batch_X)):
            adaptive_model.add_data_point(
                batch_X.iloc[j].values, pd.DataFrame([batch_y.iloc[j]]), weight=1.0
            )

        # If model initialized, evaluate performance
        if adaptive_model.is_initialized:
            # Use all data until now for evaluation
            eval_X = features.iloc[:end_idx]
            eval_y = targets.iloc[:end_idx]

            curr_metrics = adaptive_model.evaluate(eval_X, eval_y.to_frame())

            # If we have initial metrics, compare
            if initial_metrics is not None:
                # Typically expect some improvement over time
                print(f"Initial RMSE: {initial_metrics.get('mse', float('inf'))**0.5}")
                print(f"Current RMSE: {curr_metrics.get('mse', float('inf'))**0.5}")
            else:
                initial_metrics = curr_metrics

    # Verify final model state
    assert adaptive_model.is_initialized
    assert len(adaptive_model.data_buffer_X) > 0
    assert len(adaptive_model.data_buffer_y) > 0


def test_real_time_training_with_data_callback(time_series_data):
    """Test real-time trainer with data callback function."""
    # Prepare features and targets
    features = time_series_data[["input_amount", "energy_used", "cycle_time"]]
    targets = time_series_data["output_amount"]

    # Create data index for streaming simulation
    data_index = 0

    # Data callback function to simulate real-time data stream
    def data_callback():
        nonlocal data_index
        if data_index >= len(features):
            return None

        X = features.iloc[data_index : data_index + 1]
        y = targets.iloc[data_index : data_index + 1]
        data_index += 1

        return X, pd.DataFrame(y)

    # Create trainer with callback
    trainer = RealTimeModelTrainer(data_source_callback=data_callback)

    # Start training for a short period
    trainer.start(interval_seconds=1)

    # Let it run for a few seconds to process data
    time.sleep(3)

    # Stop training
    trainer.stop()

    # Verify data was processed
    assert trainer.processed_samples > 0
    assert len(trainer.adaptive_model.data_buffer_X) > 0
    assert len(trainer.adaptive_model.data_buffer_y) > 0


@pytest.mark.skip(reason="Long running test")
def test_model_comparison_online_vs_batch(time_series_data):
    """Compare online learning vs batch learning performance."""
    # Prepare features and targets
    features = time_series_data[["input_amount", "energy_used", "cycle_time"]]
    targets = time_series_data["output_amount"]

    # Train size - use 80% for training
    train_size = int(0.8 * len(features))
    train_X, test_X = features.iloc[:train_size], features.iloc[train_size:]
    train_y, test_y = targets.iloc[:train_size], targets.iloc[train_size:]

    # 1. Train batch model
    batch_model = EnsembleModel()
    batch_model.train(train_X, train_y.to_frame())

    # 2. Train adaptive model
    adaptive_model = AdaptiveModel()
    for i in range(train_size):
        adaptive_model.add_data_point(
            train_X.iloc[i].values, pd.DataFrame([train_y.iloc[i]]), weight=1.0
        )

    # Evaluate on test set
    batch_metrics = batch_model.evaluate(test_X, test_y.to_frame())
    adaptive_metrics = adaptive_model.evaluate(test_X, test_y.to_frame())

    # Compare performance
    print(f"Batch model R²: {batch_metrics.get('r2', 'N/A')}")
    print(f"Adaptive model R²: {adaptive_metrics.get('r2', 'N/A')}")

    # Verify both models produce valid predictions
    batch_preds = batch_model.predict(test_X)
    adaptive_preds = adaptive_model.predict(test_X)

    assert len(batch_preds) == len(test_X)
    assert len(adaptive_preds) == len(test_X)
