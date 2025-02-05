"""
Test suite for the LCA data generation functionality.
This ensures our synthetic data generation produces realistic and consistent values.
"""

import pytest
import pandas as pd
from datetime import datetime
from src.circman5.test_data_generator import ManufacturingDataGenerator


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
    material_data = data_generator.generate_lca_material_data()

    # Check that we have the expected data structure
    assert isinstance(material_data, pd.DataFrame)
    expected_columns = {
        "timestamp",
        "production_line",
        "panel_type",
        "material_type",
        "quantity_used",
        "waste_generated",
        "recycled_amount",
        "batch_id",
    }
    assert set(material_data.columns) == expected_columns

    # Verify data constraints
    assert not material_data.empty
    assert material_data["quantity_used"].min() > 0
    assert all(material_data["waste_generated"] <= material_data["quantity_used"])
    assert all(material_data["recycled_amount"] <= material_data["waste_generated"])


def test_lca_energy_data_generation(data_generator):
    """
    Test energy data generation for LCA calculations.
    Verifies energy consumption patterns and source distributions.
    """
    energy_data = data_generator.generate_lca_energy_data()

    # Check structure
    assert isinstance(energy_data, pd.DataFrame)
    expected_columns = {
        "timestamp",
        "production_line",
        "energy_source",
        "energy_consumption",
        "carbon_intensity",
        "batch_id",
    }
    assert set(energy_data.columns) == expected_columns

    # Verify energy source distribution
    energy_sources = energy_data["energy_source"].unique()
    assert all(source in ["grid", "solar", "wind"] for source in energy_sources)

    # Check energy consumption values
    assert energy_data["energy_consumption"].min() >= 0
    assert energy_data["carbon_intensity"].min() >= 0


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
        "impact_factor",
        "batch_id",
    }
    assert set(process_data.columns) == expected_columns

    # Verify process times are within expected range
    assert process_data["process_time"].min() >= 45  # Minimum process time
    assert process_data["process_time"].max() <= 75  # Maximum process time


def test_complete_lca_dataset_generation(data_generator):
    """
    Test the generation of a complete LCA dataset.
    Verifies that all components are present and properly structured.
    """
    complete_dataset = data_generator.generate_complete_lca_dataset()

    # Check that all expected components are present
    expected_components = {
        "material_data",
        "energy_data",
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
    Checks for matching timestamps, batch IDs, and production lines.
    """
    complete_dataset = data_generator.generate_complete_lca_dataset()

    # Get unique batch IDs from each component
    batch_ids = {
        component: set(df["batch_id"].unique())
        for component, df in complete_dataset.items()
        if "batch_id" in df.columns
    }

    # Check that batch IDs are consistent across components
    first_component = list(batch_ids.keys())[0]
    for component in batch_ids:
        assert batch_ids[component].intersection(
            batch_ids[first_component]
        ), f"Batch IDs in {component} don't match other components"
