# tests/integration/test_data_saving.py
"""Test data generation and saving locations."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from circman5.manufacturing.reporting.visualizations import ManufacturingVisualizer
from circman5.manufacturing.lifecycle import LCAAnalyzer, LifeCycleImpact
from circman5.monitoring import ManufacturingMonitor
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.results_manager import results_manager  # Updated import


@pytest.fixture(scope="module")
def test_generator():
    """Create a test data generator instance."""
    return ManufacturingDataGenerator(start_date="2024-01-01", days=5)


@pytest.fixture(scope="module")
def generated_test_data(test_generator):
    """Generate test data for all components."""
    return {
        "production": test_generator.generate_production_data(),
        "energy": test_generator.generate_energy_data(),
        "quality": test_generator.generate_quality_data(),
        "material": test_generator.generate_material_flow_data(),
        "lca": test_generator.generate_complete_lca_dataset(),
    }


@pytest.fixture(scope="module")
def test_run_dir():
    """Create and maintain a single run directory for the entire test module using ResultsManager."""
    run_dir = results_manager.get_run_dir()  # Updated to use ResultsManager

    # Ensure required subdirectories exist (they are normally created by ResultsManager,
    # but this guarantees they are present for the tests)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)

    return run_dir


def test_all_components_saving(test_run_dir):
    """Test component saving functionality."""
    # Create test data with correct columns
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    production_data = pd.DataFrame(
        {
            "timestamp": dates,
            "production_rate": np.random.normal(100, 10, 10),
            "energy_efficiency": np.random.normal(0.8, 0.1, 10),
        }
    )

    # Test visualizer
    viz = ManufacturingVisualizer()
    viz.plot_efficiency_trends(
        production_data,
        save_path=str(test_run_dir / "visualizations" / "efficiency_trends.png"),
    )
    assert (test_run_dir / "visualizations" / "efficiency_trends.png").exists()

    # Test monitor
    monitor = ManufacturingMonitor()
    monitor.metrics_history["efficiency"] = production_data
    monitor.save_metrics(
        "efficiency", save_path=test_run_dir / "reports" / "efficiency_metrics.csv"
    )
    assert (test_run_dir / "reports" / "efficiency_metrics.csv").exists()

    # Test LCA
    lca = LCAAnalyzer()
    impact = lca.perform_full_lca(
        material_inputs={"silicon": 100.0, "glass": 200.0},
        energy_consumption=1000.0,
        lifetime_years=25.0,
        annual_energy_generation=2000.0,
        grid_carbon_intensity=0.5,
        recycling_rates={"silicon": 0.8, "glass": 0.9},
        transport_distance=100.0,
    )
    # Pass the output directory rather than a full file path with the filename.
    output_dir = test_run_dir / "reports"
    lca.save_results(impact, "TEST_001", output_dir)
    expected_file = output_dir / "lca_impact_TEST_001.xlsx"
    assert expected_file.exists()


def test_data_generator_saving(test_generator, test_run_dir):
    """Test data generator's save functionality."""
    synthetic_dir = results_manager.get_path(
        "SYNTHETIC_DATA"
    )  # Updated to use ResultsManager
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    # Define expected files outside try block
    expected_files = [
        "test_energy_data.csv",
        "test_material_data.csv",
        "test_process_data.csv",
        "test_production_data.csv",
    ]

    try:
        # Save generated data
        test_generator.save_generated_data()

        # Verify files were created in correct location
        for filename in expected_files:
            file_path = synthetic_dir / filename
            assert (
                file_path.exists()
            ), f"File not found in synthetic data directory: {filename}"
            df = pd.read_csv(file_path)
            assert not df.empty, f"Empty data in {filename}"

    finally:
        # Clean up test files
        for filename in expected_files:
            file_path = synthetic_dir / filename
            if file_path.exists():
                file_path.unlink()


def test_generated_data_integration(generated_test_data, test_run_dir):
    """Test integration of generated data with system components."""
    # Test with visualizer
    viz = ManufacturingVisualizer()
    efficiency_data = pd.DataFrame(
        {
            "timestamp": generated_test_data["production"]["timestamp"],
            "production_rate": generated_test_data["production"]["output_amount"],
            "energy_efficiency": generated_test_data["production"]["yield_rate"] / 100,
        }
    )
    viz.plot_efficiency_trends(
        efficiency_data,
        save_path=str(test_run_dir / "visualizations" / "efficiency_trends.png"),
    )
    assert (test_run_dir / "visualizations" / "efficiency_trends.png").exists()

    # Test with monitor
    monitor = ManufacturingMonitor()
    monitor.metrics_history["efficiency"] = generated_test_data["production"]
    monitor.save_metrics(
        "efficiency", save_path=test_run_dir / "reports" / "efficiency_metrics.csv"
    )
    assert (test_run_dir / "reports" / "efficiency_metrics.csv").exists()

    # Test with LCA analyzer
    lca = LCAAnalyzer()
    lca_data = generated_test_data["lca"]
    material_inputs = {
        "silicon": lca_data["material_flow"]["quantity_used"].sum(),
        "glass": lca_data["material_flow"]["quantity_used"].sum(),
    }

    impact = lca.perform_full_lca(
        material_inputs=material_inputs,
        energy_consumption=lca_data["energy_consumption"]["energy_consumption"].sum(),
        lifetime_years=25.0,
        annual_energy_generation=2000.0,
        grid_carbon_intensity=0.5,
        recycling_rates={"silicon": 0.8, "glass": 0.9},
        transport_distance=100.0,
    )
    # Again, pass the output directory instead of a full file path.
    output_dir = test_run_dir / "reports"
    lca.save_results(impact, "TEST_002", output_dir)
    expected_file = output_dir / "lca_impact_TEST_002.xlsx"
    assert expected_file.exists()


def test_data_validation(generated_test_data):
    """Test validation of generated data."""
    production_data = generated_test_data["production"]

    # Test production data constraints
    assert (production_data["input_amount"] > 0).all()
    assert (production_data["output_amount"] > 0).all()
    assert (production_data["output_amount"] <= production_data["input_amount"]).all()

    # Test quality data constraints
    quality_data = generated_test_data["quality"]
    assert (quality_data["efficiency"] >= 0).all() and (
        quality_data["efficiency"] <= 100
    ).all()
    assert (quality_data["defect_rate"] >= 0).all() and (
        quality_data["defect_rate"] <= 100
    ).all()

    # Test energy data constraints
    energy_data = generated_test_data["energy"]
    assert (energy_data["energy_consumption"] > 0).all()
    assert energy_data["energy_source"].isin(["grid", "solar", "wind"]).all()
