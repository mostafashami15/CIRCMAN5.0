# tests/unit/manufacturing/reporting/test_optimization_visualizer.py

import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np

from circman5.manufacturing.reporting.optimization_visualizer import (
    OptimizationVisualizer,
)
from circman5.manufacturing.optimization.types import OptimizationResults, MetricsDict
from circman5.utils.results_manager import results_manager


@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """Setup and cleanup for tests."""
    plt.close("all")  # Close any existing figures
    yield
    plt.close("all")  # Cleanup after test


@pytest.fixture
def visualizer():
    return OptimizationVisualizer()


@pytest.fixture
def sample_optimization_results() -> OptimizationResults:
    return OptimizationResults(
        original_params={
            "input_amount": 100.0,
            "energy_used": 50.0,
            "cycle_time": 30.0,
            "efficiency": 0.8,
        },
        optimized_params={
            "input_amount": 95.0,
            "energy_used": 45.0,
            "cycle_time": 28.0,
            "efficiency": 0.85,
        },
        improvement={
            "input_amount": -5.0,
            "energy_used": -10.0,
            "cycle_time": -6.67,
            "efficiency": 6.25,
        },
        optimization_success=True,
        optimization_message="Optimization successful",
        iterations=100,
        objective_value=0.95,
    )


@pytest.fixture
def sample_metrics() -> MetricsDict:
    # Include all required fields for MetricsDict
    return {
        "mse": 0.02,
        "rmse": 0.14,
        "mae": 0.12,
        "r2": 0.95,
        "cv_r2_mean": 0.94,
        "cv_r2_std": 0.02,
        "cv_mse_mean": 0.03,
        "mean_uncertainty": 0.05,
        "feature_importance": {
            "input_amount": 0.35,
            "energy_used": 0.25,
            "cycle_time": 0.20,
            "efficiency": 0.20,
        },
    }


def test_plot_optimization_impact(visualizer, sample_optimization_results):
    output_path = visualizer.plot_optimization_impact(sample_optimization_results)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    plt.close("all")


def test_plot_feature_importance(visualizer, sample_metrics):
    output_path = visualizer.plot_feature_importance(sample_metrics)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    plt.close("all")


def test_plot_convergence_history(visualizer, sample_optimization_results):
    output_path = visualizer.plot_convergence_history(sample_optimization_results)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    plt.close("all")


def test_plot_parameter_comparison(visualizer, sample_optimization_results):
    output_path = visualizer.plot_parameter_comparison(sample_optimization_results)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    plt.close("all")


def test_create_optimization_dashboard(
    visualizer, sample_optimization_results, sample_metrics
):
    output_path = visualizer.create_optimization_dashboard(
        sample_optimization_results, sample_metrics
    )
    assert output_path.exists()
    assert output_path.suffix == ".png"
    plt.close("all")


def test_load_optimization_results(visualizer, sample_optimization_results, tmp_path):
    # Create temporary results file
    results_file = tmp_path / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(sample_optimization_results, f)

    # Load and verify results
    loaded_results = visualizer.load_optimization_results(results_file)
    assert loaded_results == sample_optimization_results
