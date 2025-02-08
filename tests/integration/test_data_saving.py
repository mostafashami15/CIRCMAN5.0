"""Test data generation and saving locations."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from circman5.visualization.manufacturing_visualizer import ManufacturingVisualizer
from circman5.analysis.lca.core import LCAAnalyzer, LifeCycleImpact
from circman5.monitoring import ManufacturingMonitor
from circman5.config.project_paths import project_paths
from circman5.test_data_generator import ManufacturingDataGenerator


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
    """Create and maintain a single run directory for the entire test module."""
    run_dir = project_paths.get_run_directory()

    # Create required subdirectories
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
    output_path = test_run_dir / "reports" / "lca_impact_TEST_001.xlsx"
    lca.save_results(impact, "TEST_001", output_path)
    assert output_path.exists()


def test_data_generator_saving(test_generator, test_run_dir):
    """Test data generator's save functionality."""
    synthetic_dir = Path(project_paths.get_path("SYNTHETIC_DATA"))

    # Save generated data
    test_generator.save_generated_data()

    expected_files = [
        "test_energy_data.csv",
        "test_material_data.csv",
        "test_process_data.csv",
    ]

    for filename in expected_files:
        assert (synthetic_dir / filename).exists()
        df = pd.read_csv(synthetic_dir / filename)
        assert not df.empty


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

    output_path = test_run_dir / "reports" / "lca_impact_TEST_002.xlsx"
    lca.save_results(impact, "TEST_002", output_path)
    assert output_path.exists()


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
