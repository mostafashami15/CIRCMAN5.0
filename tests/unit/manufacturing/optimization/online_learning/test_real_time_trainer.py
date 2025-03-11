# tests/unit/manufacturing/optimization/online_learning/test_real_time_trainer.py

import pytest
import numpy as np
import pandas as pd
import time
from circman5.manufacturing.optimization.online_learning.real_time_trainer import (
    RealTimeModelTrainer,
)
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    data_gen = ManufacturingDataGenerator()
    production_data = data_gen.generate_production_data()
    features = production_data[["input_amount", "energy_used", "cycle_time"]].iloc[:50]
    targets = production_data["output_amount"].iloc[:50]
    return features, targets


def test_trainer_initialization():
    """Test that the real-time trainer initializes correctly."""
    trainer = RealTimeModelTrainer()
    assert trainer is not None
    assert trainer.data_source_callback is None
    assert trainer.adaptive_model is not None
    assert trainer.training_thread is None
    assert trainer.processed_samples == 0

    # Test with data source callback
    def dummy_callback():
        return None

    trainer2 = RealTimeModelTrainer(data_source_callback=dummy_callback)
    assert trainer2.data_source_callback is dummy_callback


def test_process_data(sample_data):
    """Test processing data through the trainer."""
    X, y = sample_data
    trainer = RealTimeModelTrainer()

    # Process a batch of data
    trainer._process_data(X.iloc[:5], y.iloc[:5])

    # Verify processing
    assert trainer.processed_samples == 5
    assert len(trainer.adaptive_model.data_buffer_X) == 5
    assert len(trainer.adaptive_model.data_buffer_y) == 5


def test_metrics_recording(sample_data):
    """Test recording of metrics."""
    X, y = sample_data
    trainer = RealTimeModelTrainer()

    # Set start time for testing
    trainer.start_time = pd.Timestamp.now()

    # Process data
    trainer._process_data(X.iloc[:5], y.iloc[:5])

    # Record metrics
    trainer._record_metrics()

    # Verify metrics
    assert len(trainer.metrics_history) == 1
    metrics = trainer.metrics_history[0]
    assert "timestamp" in metrics
    assert "processed_samples" in metrics
    assert metrics["processed_samples"] == 5
    assert "uptime_seconds" in metrics
    assert "samples_per_second" in metrics
    assert "model_updates" in metrics
    assert "buffer_size" in metrics
    assert metrics["buffer_size"] == 5


@pytest.mark.skip(reason="Involves threading and might cause test interference")
def test_training_loop_start_stop():
    """Test starting and stopping the training loop."""
    trainer = RealTimeModelTrainer()

    # Start training
    trainer.start(interval_seconds=1)  # Short interval for testing
    assert trainer.training_thread is not None
    assert trainer.training_thread.is_alive()

    # Short sleep to allow thread to run
    time.sleep(0.3)

    # Stop training
    trainer.stop()
    assert not trainer.stop_training.is_set()

    # Verify thread stopped
    time.sleep(0.2)  # Give thread time to stop
    assert not trainer.training_thread.is_alive()
