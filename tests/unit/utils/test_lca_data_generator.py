# tests/unit/utils/test_lca_data_generator.py

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.results_manager import ResultsManager


@pytest.fixture
def results_manager():
    """Create ResultsManager instance with cleanup."""
    manager = ResultsManager()
    yield manager
    # Cleanup test files
    for path_key in ["SYNTHETIC_DATA", "PROCESSED_DATA"]:
        data_dir = manager.get_path(path_key)
        for file in data_dir.glob("test_*.csv"):
            file.unlink(missing_ok=True)


@pytest.fixture
def data_generator():
    """Create a data generator instance for testing."""
    return ManufacturingDataGenerator(
        start_date="2024-01-01",
        days=5,  # Using a smaller number of days for faster testing
    )


def test_lca_material_data_generation(data_generator, results_manager):
    """Test material data generation for LCA calculations."""
    material_data = data_generator.generate_material_flow_data()

    # Check data structure
    assert isinstance(material_data, pd.DataFrame)
    expected_columns = {
        "timestamp",
        "material_type",
        "quantity_used",
        "waste_generated",
        "recycled_amount",
        "batch_id",
    }
    assert set(material_data.columns) == expected_columns

    # Verify data constraints
    assert not material_data.empty, "Material data should not be empty"
    assert material_data["quantity_used"].min() > 0, "Quantity used should be positive"

    # Check waste and recycling relationships
    waste_ratio = material_data["waste_generated"] / material_data["quantity_used"]
    assert all(waste_ratio <= 1.0), "Waste cannot exceed quantity used"

    recycling_ratio = (
        material_data["recycled_amount"] / material_data["waste_generated"]
    )
    acceptable_rows = recycling_ratio <= 1.0
    assert all(acceptable_rows), "Recycled amount cannot exceed waste generated"


def test_lca_energy_data_generation(data_generator):
    """Test energy data generation for LCA calculations."""
    energy_data = data_generator.generate_energy_data()

    # Check structure
    assert isinstance(energy_data, pd.DataFrame)
    expected_columns = {
        "timestamp",
        "production_line",
        "energy_source",
        "energy_consumption",
        "efficiency_rate",
    }
    assert set(energy_data.columns) == expected_columns

    # Verify energy sources
    energy_sources = energy_data["energy_source"].unique()
    assert all(source in ["grid", "solar", "wind"] for source in energy_sources)

    # Check value ranges
    assert energy_data["energy_consumption"].min() >= 0
    assert all(0 <= energy_data["efficiency_rate"])
    assert all(energy_data["efficiency_rate"] <= 1.0)


def test_lca_process_data_generation(data_generator):
    """Test process data generation for LCA calculations."""
    process_data = data_generator.generate_lca_process_data()

    # Check structure
    assert isinstance(process_data, pd.DataFrame)
    expected_columns = {
        "timestamp",
        "production_line",
        "process_step",
        "process_time",
        "batch_id",
    }
    assert set(process_data.columns) == expected_columns

    # Verify process times
    assert process_data["process_time"].min() >= 45
    assert process_data["process_time"].max() <= 75


def test_complete_lca_dataset_generation(data_generator):
    """Test generation of complete LCA dataset."""
    complete_dataset = data_generator.generate_complete_lca_dataset()

    # Check components
    expected_components = {
        "material_flow",
        "energy_consumption",
        "process_data",
        "production_data",
    }
    assert set(complete_dataset.keys()) == expected_components

    # Verify each component
    for component in complete_dataset.values():
        assert isinstance(component, pd.DataFrame)
        assert not component.empty


def test_data_consistency_across_components(data_generator):
    """Test data consistency between different LCA components."""
    # Generate datasets
    production_data = data_generator.generate_production_data()
    energy_data = data_generator.generate_energy_data()
    material_data = data_generator.generate_material_flow_data()

    # Check timestamps
    assert "timestamp" in production_data.columns
    assert "timestamp" in energy_data.columns
    assert "timestamp" in material_data.columns

    # Convert timestamps to datetime if needed
    prod_dates = pd.to_datetime(production_data["timestamp"])
    energy_dates = pd.to_datetime(energy_data["timestamp"])
    material_dates = pd.to_datetime(material_data["timestamp"])

    # Check date ranges
    assert prod_dates.min() <= energy_dates.min()
    assert prod_dates.max() >= energy_dates.max()
    assert prod_dates.min() <= material_dates.min()
    assert prod_dates.max() >= material_dates.max()


def test_data_saving_and_loading(data_generator, results_manager):
    """Test saving and loading of LCA data."""
    # Generate and save data
    data_generator.save_generated_data()

    # Verify files exist in synthetic data directory
    synthetic_dir = results_manager.get_path("SYNTHETIC_DATA")
    expected_files = [
        "test_energy_data.csv",
        "test_material_data.csv",
        "test_process_data.csv",
        "test_production_data.csv",
    ]

    for filename in expected_files:
        file_path = synthetic_dir / filename
        assert file_path.exists(), f"File {filename} not found"

        # Verify data integrity
        df = pd.read_csv(file_path)
        assert not df.empty, f"File {filename} is empty"
        assert all(
            col in df.columns for col in ["timestamp"]
        ), f"Missing required columns in {filename}"
