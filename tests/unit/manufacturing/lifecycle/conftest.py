# tests/unit/manufacturing/lifecycle/conftest.py

"""Shared fixtures for LCA testing."""

import shutil
import pytest
import pandas as pd
from pathlib import Path
from typing import Dict
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.manufacturing.lifecycle import LCAAnalyzer
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.utils.results_manager import results_manager
from circman5.config.project_paths import project_paths


def debug_directory_contents(directory: Path, pattern: str = "*") -> None:
    """Debug helper to print directory contents."""
    print(f"\nDebug: Contents of {directory}")
    print(f"Directory exists: {directory.exists()}")
    if directory.exists():
        files = list(directory.glob(pattern))
        print(f"Files matching '{pattern}': {[f.name for f in files]}")
        print("-" * 50)


# Directory Structure Fixtures
@pytest.fixture(scope="session")
def test_run_dir():
    """Create and get test run directory."""
    run_dir = results_manager.get_run_dir()
    debug_directory_contents(run_dir)
    yield run_dir
    # Optional: Clean up old runs after tests
    results_manager.cleanup_old_runs(keep_last=5)


@pytest.fixture
def visualizations_dir():
    """Get visualizations directory."""
    return results_manager.get_path("visualizations")


@pytest.fixture
def reports_dir():
    """Get reports directory."""
    return results_manager.get_path("reports")


@pytest.fixture
def input_data_dir():
    """Get input data directory."""
    return results_manager.get_path("input_data")


@pytest.fixture
def lca_results_dir():
    """Get LCA results directory."""
    return results_manager.get_path("lca_results")


# Data Generation Fixtures
@pytest.fixture
def test_data_generator():
    """Create a data generator for testing."""
    return ManufacturingDataGenerator(
        start_date="2024-01-01", days=5  # Small dataset for faster testing
    )


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
    try:
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
    except Exception as e:
        pytest.fail(f"Failed to generate complete test data: {str(e)}")


@pytest.fixture
def setup_synthetic_data(test_data_generator) -> Dict[str, Path]:
    """Setup synthetic data files in the correct location."""
    synthetic_dir = results_manager.get_path("SYNTHETIC_DATA")
    Path(synthetic_dir).mkdir(parents=True, exist_ok=True)

    # Generate and save test files
    test_files = {
        "test_material_data.csv": test_data_generator.generate_material_flow_data(),
        "test_energy_data.csv": test_data_generator.generate_energy_data(),
        "test_process_data.csv": test_data_generator.generate_lca_process_data(),
        "test_production_data.csv": test_data_generator.generate_production_data(),
        "test_quality_data.csv": test_data_generator.generate_quality_data(),
    }

    file_paths = {}
    for filename, data in test_files.items():
        file_path = Path(synthetic_dir) / filename
        data.to_csv(file_path, index=False)
        file_paths[filename] = file_path

    debug_directory_contents(Path(synthetic_dir), "*.csv")
    return file_paths


@pytest.fixture
def copy_synthetic_input_files(input_data_dir, setup_synthetic_data):
    """Copy synthetic test data files to the test run input directory."""
    debug_directory_contents(input_data_dir)

    # Define mapping between source and destination filenames
    files_to_copy = {
        "material_data.csv": "test_material_data.csv",
        "energy_data.csv": "test_energy_data.csv",
        "process_data.csv": "test_process_data.csv",
    }

    # Copy files using paths from setup_synthetic_data
    for dest_name, src_name in files_to_copy.items():
        if src_name not in setup_synthetic_data:
            raise FileNotFoundError(f"Source file not found: {src_name}")

        src_path = setup_synthetic_data[src_name]
        dest_path = input_data_dir / dest_name

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")

    debug_directory_contents(input_data_dir, "*.csv")
    return input_data_dir
