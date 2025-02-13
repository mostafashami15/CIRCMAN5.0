# tests/unit/manufacturing/lifecycle/conftest.py

"""Shared fixtures for LCA testing."""

import shutil
import pytest
import pandas as pd
from pathlib import Path
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.manufacturing.lifecycle import LCAAnalyzer
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.utils.cleanup import cleanup_old_runs
from circman5.utils.result_paths import get_run_directory


# Directory Structure Fixtures
@pytest.fixture(scope="session")
def test_run_dir(tmp_path_factory):
    """Create and get test run directory."""
    run_dir = get_run_directory()
    yield run_dir
    # Optional: Clean up old runs
    cleanup_old_runs(run_dir.parent, keep_last=5)


@pytest.fixture
def visualizations_dir(test_run_dir):
    """Get visualizations directory for current test run."""
    viz_dir = test_run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    return viz_dir


@pytest.fixture
def reports_dir(test_run_dir):
    """Get reports directory for current test run."""
    reports_dir = test_run_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


@pytest.fixture
def input_data_dir(test_run_dir):
    """Get input data directory for current test run."""
    data_dir = test_run_dir / "input_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def lca_results_dir(test_run_dir):
    """Get LCA results directory for current test run."""
    lca_dir = test_run_dir / "lca_results"
    lca_dir.mkdir(exist_ok=True)
    return lca_dir


# Data Generation Fixtures
@pytest.fixture
def test_data_generator():
    """Create a data generator for testing."""
    return ManufacturingDataGenerator(
        start_date="2024-01-01", days=5  # Small dataset for faster testing
    )


# Core Component Fixtures
@pytest.fixture
def lca_analyzer():
    """Create LCAAnalyzer instance for testing."""
    return LCAAnalyzer()


@pytest.fixture
def manufacturing_analyzer(complete_test_data):
    """Create SoliTekManufacturingAnalysis instance with test data."""
    analyzer = SoliTekManufacturingAnalysis()

    try:
        # Load test data
        analyzer.production_data = complete_test_data["production_data"]
        analyzer.quality_data = complete_test_data["quality_data"]
        analyzer.energy_data = complete_test_data["energy_consumption"]
        analyzer.material_flow = complete_test_data["material_flow"]

        # Set LCA data
        analyzer.lca_data = {
            "material_flow": complete_test_data["material_flow"],
            "energy_consumption": complete_test_data["energy_consumption"],
            "process_data": complete_test_data["process_data"],
        }

        return analyzer
    except KeyError as e:
        raise KeyError(f"Missing required data component: {e}") from e


# Sample Data Fixtures
@pytest.fixture
def sample_lca_data(test_data_generator):
    """Generate sample LCA data with meaningful values."""
    # Generate base data
    material_data = test_data_generator.generate_lca_material_data()
    energy_data = test_data_generator.generate_lca_energy_data()
    process_data = test_data_generator.generate_lca_process_data()

    # Ensure energy data has efficiency rate
    energy_data["efficiency_rate"] = 0.85

    # Ensure material quantities are significant enough for meaningful use phase
    material_data["quantity_used"] = material_data["quantity_used"] * 100

    # Ensure a nonzero "solar_glass" entry
    if "solar_glass" not in material_data["material_type"].unique():
        extra = pd.DataFrame(
            {
                "material_type": ["solar_glass"],
                "quantity_used": [100.0],
                "waste_generated": [0.0],
                "recycled_amount": [0.0],
            }
        )
        material_data = pd.concat([material_data, extra], ignore_index=True)

    return {
        "material_flow": material_data,
        "energy_consumption": energy_data,
        "process_data": process_data,
    }


@pytest.fixture
def sample_energy_data(test_data_generator):
    """Generate sample energy data with all required columns."""
    data = test_data_generator.generate_lca_energy_data()
    data["efficiency_rate"] = 0.85
    return data


@pytest.fixture
def sample_material_data(test_data_generator):
    """Generate sample material data."""
    return test_data_generator.generate_lca_material_data()


@pytest.fixture
def sample_process_data(test_data_generator):
    """Generate sample process data."""
    return test_data_generator.generate_lca_process_data()


@pytest.fixture
def mock_impact_factors():
    """Mock impact factors for testing."""
    return {
        "manufacturing": {"silicon": 10.0, "glass": 5.0},
        "recycling": {"silicon": -8.0, "glass": -4.0},
        "energy": {"grid": 0.5, "solar": 0.0},
    }


@pytest.fixture
def complete_test_data(test_data_generator):
    """Generate complete test dataset with all required columns."""
    production_data = test_data_generator.generate_production_data()
    quality_data = test_data_generator.generate_quality_data()
    energy_data = test_data_generator.generate_lca_energy_data()
    energy_data["efficiency_rate"] = 0.85
    material_data = test_data_generator.generate_lca_material_data()
    material_data["quantity_used"] = material_data["quantity_used"].fillna(100.0)
    process_data = test_data_generator.generate_lca_process_data()

    return {
        "production_data": production_data,
        "quality_data": quality_data,
        "material_flow": material_data,
        "energy_consumption": energy_data,
        "process_data": process_data,
    }


@pytest.fixture
def copy_synthetic_input_files(input_data_dir):
    """Copy synthetic test data files to the test run input directory."""
    project_root = Path(__file__).resolve().parents[4]
    synthetic_dir = project_root / "data" / "synthetic"

    files_to_copy = {
        "material_data.csv": "test_material_data.csv",
        "energy_data.csv": "test_energy_data.csv",
        "process_data.csv": "test_process_data.csv",
    }

    for dest_name, src_name in files_to_copy.items():
        src_path = synthetic_dir / src_name
        dest_path = input_data_dir / dest_name
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        shutil.copy(src_path, dest_path)

    return input_data_dir
