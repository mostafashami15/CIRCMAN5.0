# tests/unit/manufacturing/digital_twin/test_lca_integration.py

"""Unit tests for Digital Twin LCA integration."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from circman5.manufacturing.lifecycle.lca_analyzer import LifeCycleImpact
from circman5.utils.results_manager import results_manager


def test_extract_material_data(lca_integration, sample_lca_state):
    """Test extraction of material data from digital twin state."""
    material_data = lca_integration.extract_material_data_from_state(sample_lca_state)

    # Verify dataframe structure
    assert isinstance(material_data, pd.DataFrame)
    assert not material_data.empty
    assert all(
        col in material_data.columns
        for col in [
            "batch_id",
            "timestamp",
            "material_type",
            "quantity_used",
            "waste_generated",
            "recycled_amount",
        ]
    )

    # Verify material types
    material_types = material_data["material_type"].unique()
    expected_materials = [
        "silicon_wafer",
        "solar_glass",
        "eva_sheet",
        "backsheet",
        "aluminum_frame",
    ]
    assert all(material in material_types for material in expected_materials)

    # Verify quantities
    assert (material_data["quantity_used"] > 0).all()
    assert (material_data["waste_generated"] >= 0).all()
    assert (material_data["recycled_amount"] >= 0).all()


def test_extract_energy_data(lca_integration, sample_lca_state):
    """Test extraction of energy data from digital twin state."""
    energy_data = lca_integration.extract_energy_data_from_state(sample_lca_state)

    # Verify dataframe structure
    assert isinstance(energy_data, pd.DataFrame)
    assert not energy_data.empty
    assert all(
        col in energy_data.columns
        for col in [
            "batch_id",
            "timestamp",
            "energy_source",
            "energy_consumption",
            "process_stage",
        ]
    )

    # Verify energy data
    assert "grid_electricity" in energy_data["energy_source"].values
    assert (energy_data["energy_consumption"] > 0).all()


def test_perform_lca_analysis(lca_integration, sample_lca_state):
    """Test performing LCA analysis from digital twin state."""
    # Set up test output directory
    output_dir = results_manager.get_path("lca_results")

    # Perform LCA analysis
    impact = lca_integration.perform_lca_analysis(
        state=sample_lca_state,
        batch_id="test_analysis",
        save_results=True,
        output_dir=output_dir,
    )

    # Verify impact results
    assert isinstance(impact, LifeCycleImpact)
    assert hasattr(impact, "manufacturing_impact")
    assert hasattr(impact, "use_phase_impact")
    assert hasattr(impact, "end_of_life_impact")
    assert hasattr(impact, "total_carbon_footprint")

    # Verify results are in the correct range
    assert impact.manufacturing_impact > 0
    assert impact.use_phase_impact < 0  # Should be negative (environmental benefit)
    assert isinstance(impact.total_carbon_footprint, float)


def test_aggregate_material_quantities(lca_integration):
    """Test aggregation of material quantities."""
    # Create sample material data
    material_data = pd.DataFrame(
        {
            "batch_id": ["batch1", "batch1", "batch1", "batch1"],
            "timestamp": [datetime.now()] * 4,
            "material_type": [
                "silicon_wafer",
                "solar_glass",
                "silicon_wafer",
                "aluminum_frame",
            ],
            "quantity_used": [100.0, 150.0, 50.0, 80.0],
            "waste_generated": [5.0, 7.5, 2.5, 4.0],
            "recycled_amount": [4.0, 6.0, 2.0, 3.5],
        }
    )

    # Aggregate quantities
    material_quantities = lca_integration._aggregate_material_quantities(material_data)

    # Verify results
    assert isinstance(material_quantities, dict)
    assert "silicon_wafer" in material_quantities
    assert "solar_glass" in material_quantities
    assert "aluminum_frame" in material_quantities
    assert material_quantities["silicon_wafer"] == 150.0  # 100 + 50
    assert material_quantities["solar_glass"] == 150.0
    assert material_quantities["aluminum_frame"] == 80.0


def test_null_state_handling(lca_integration, monkeypatch):
    """Test handling of null states."""
    # Mock the get_current_state to return None
    monkeypatch.setattr(lca_integration.digital_twin, "get_current_state", lambda: None)

    # Test with None state
    material_data = lca_integration.extract_material_data_from_state(None)
    energy_data = lca_integration.extract_energy_data_from_state(None)

    # Verify empty dataframes are returned
    assert isinstance(material_data, pd.DataFrame)
    assert isinstance(energy_data, pd.DataFrame)
    assert material_data.empty
    assert energy_data.empty

    # Test performing LCA with None state
    impact = lca_integration.perform_lca_analysis(state=None, save_results=False)

    # Verify empty impact object
    assert isinstance(impact, LifeCycleImpact)
    assert impact.manufacturing_impact == 0.0
    assert impact.use_phase_impact == 0.0
    assert impact.end_of_life_impact == 0.0
