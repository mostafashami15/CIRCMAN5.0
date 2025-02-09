"""
Test suite for the LCA data generation functionality.
This ensures our synthetic data generation produces realistic and consistent values.
"""

import pytest
import pandas as pd
from datetime import datetime
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture
def data_generator():
    """Create a data generator instance for testing."""
    return ManufacturingDataGenerator(
        start_date="2024-01-01",
        days=5,  # Using a smaller number of days for faster testing
    )


def test_lca_material_data_generation(data_generator):
    """
    Test that material data generation produces valid and consistent results.
    We check for proper data structure and realistic value ranges.
    """
    material_data = data_generator.generate_material_flow_data()

    # Check that we have the expected data structure
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

    if not all(acceptable_rows):
        problem_rows = material_data[~acceptable_rows]
        print("\nProblem rows found:")
        print(problem_rows[["waste_generated", "recycled_amount"]])

    assert all(acceptable_rows), "Recycled amount cannot exceed waste generated"


def test_lca_energy_data_generation(data_generator):
    """
    Test energy data generation for LCA calculations.
    Verifies energy consumption patterns and source distributions.
    """
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

    # Verify energy source distribution
    energy_sources = energy_data["energy_source"].unique()
    assert all(source in ["grid", "solar", "wind"] for source in energy_sources)

    # Check energy consumption values
    assert (
        energy_data["energy_consumption"].min() >= 0
    ), "Energy consumption should be non-negative"
    assert all(
        0 <= energy_data["efficiency_rate"]
    ), "Efficiency rate should be non-negative"
    assert all(
        energy_data["efficiency_rate"] <= 1.0
    ), "Efficiency rate should not exceed 1.0"


def test_lca_process_data_generation(data_generator):
    """
    Test process data generation functionality.
    Ensures process steps and timings are properly generated.
    """
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

    # Verify process times are within expected range
    assert (
        process_data["process_time"].min() >= 45
    ), "Process time should not be below minimum"
    assert (
        process_data["process_time"].max() <= 75
    ), "Process time should not exceed maximum"


def test_complete_lca_dataset_generation(data_generator):
    """
    Test the generation of a complete LCA dataset.
    Verifies that all components are present and properly structured.
    """
    complete_dataset = data_generator.generate_complete_lca_dataset()

    # Check that all expected components are present
    expected_components = {
        "material_flow",
        "energy_consumption",
        "process_data",
        "production_data",
    }
    assert set(complete_dataset.keys()) == expected_components

    # Verify each component is a DataFrame
    for component in complete_dataset.values():
        assert isinstance(component, pd.DataFrame)
        assert not component.empty


def test_data_consistency_across_components(data_generator):
    """
    Test that data is consistent across different components.
    Checks for matching timestamps and production lines.
    """
    # Generate individual datasets
    production_data = data_generator.generate_production_data()
    energy_data = data_generator.generate_energy_data()
    material_data = data_generator.generate_material_flow_data()

    # Check all datasets have timestamps
    assert "timestamp" in production_data.columns
    assert "timestamp" in energy_data.columns
    assert "timestamp" in material_data.columns

    # Verify time ranges are consistent
    prod_range = (
        production_data["timestamp"].min(),
        production_data["timestamp"].max(),
    )
    energy_range = (energy_data["timestamp"].min(), energy_data["timestamp"].max())
    material_range = (
        material_data["timestamp"].min(),
        material_data["timestamp"].max(),
    )

    # All data should fall within the same date range
    assert prod_range[0] <= energy_range[0] and prod_range[1] >= energy_range[1]
    assert prod_range[0] <= material_range[0] and prod_range[1] >= material_range[1]

    # Check production line consistency where applicable
    if (
        "production_line" in production_data.columns
        and "production_line" in energy_data.columns
    ):
        prod_lines = set(production_data["production_line"])
        energy_lines = set(energy_data["production_line"])
        assert (
            prod_lines == energy_lines
        ), "Production lines should match across datasets"
