# tests/unit/manufacturing/lifecycle/test_lca_core.py
"""Unit tests for LCA analysis module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from circman5.manufacturing.lifecycle import LCAAnalyzer, LifeCycleImpact
from circman5.utils.results_manager import results_manager
from circman5.manufacturing.lifecycle.impact_factors import (
    MATERIAL_IMPACT_FACTORS,
    ENERGY_IMPACT_FACTORS,
    RECYCLING_BENEFIT_FACTORS,
)


@pytest.fixture
def sample_material_inputs():
    """Fixture providing sample material inputs."""
    return {
        "silicon_wafer": 100.0,
        "solar_glass": 150.0,
        "eva_sheet": 50.0,
        "backsheet": 30.0,
        "aluminum_frame": 80.0,
    }


def test_manufacturing_impact_calculation(lca_analyzer, sample_material_data):
    """Test manufacturing phase impact calculations."""
    material_inputs = {
        "silicon_wafer": 100.0,
        "solar_glass": 150.0,
        "eva_sheet": 50.0,
        "backsheet": 30.0,
        "aluminum_frame": 80.0,
    }
    energy_consumption = 1000.0  # kWh

    impact = lca_analyzer.calculate_manufacturing_impact(
        material_inputs, energy_consumption
    )

    assert impact > 0
    assert isinstance(impact, float)
    # Verify impact calculation using known factors
    expected_impact = sum(
        qty * MATERIAL_IMPACT_FACTORS.get(mat, 0)
        for mat, qty in material_inputs.items()
    )
    expected_impact += energy_consumption * ENERGY_IMPACT_FACTORS["grid_electricity"]
    assert (
        abs(impact - expected_impact) < 0.01
    )  # Allow for small floating point differences


def test_use_phase_impact_calculation(lca_analyzer):
    """Test use phase impact calculations without degradation."""
    annual_generation = 1000.0
    lifetime = 25
    grid_intensity = 0.5

    impact = lca_analyzer._calculate_use_phase_impact(
        annual_generation=annual_generation,
        lifetime=lifetime,
        grid_intensity=grid_intensity,
    )

    # Calculate expected total lifetime generation (without degradation effects)
    total_generation = annual_generation * lifetime
    expected_impact = -1.0 * total_generation * grid_intensity

    assert impact < 0  # Should be negative (environmental benefit)
    assert isinstance(impact, float)
    assert abs(impact - expected_impact) < 1.0  # Allow for small difference

    # Additional validation of the calculated values
    assert abs(impact) > abs(
        annual_generation * grid_intensity
    )  # Impact should exceed single year
    assert abs(impact) <= abs(
        annual_generation * lifetime * grid_intensity
    )  # Impact should not exceed theoretical maximum


def test_use_phase_impact_with_degradation(lca_analyzer):
    """Test use phase impact calculations with degradation rates."""
    annual_generation = 1200.0
    lifetime = 20
    grid_intensity = 0.4
    degradation_rate = 0.005  # from DEGRADATION_RATES["mono_perc"]

    impact = lca_analyzer._calculate_use_phase_impact(
        annual_generation=annual_generation,
        lifetime=lifetime,
        grid_intensity=grid_intensity,
        degradation_rate=degradation_rate,
    )

    # Calculate expected total generation with degradation
    total_generation = sum(
        annual_generation * ((1 - degradation_rate) ** year) for year in range(lifetime)
    )
    expected_impact = -1.0 * total_generation * grid_intensity

    # Verify results
    assert impact < 0  # Should be negative (environmental benefit)
    assert isinstance(impact, float)
    assert (
        abs(impact - expected_impact) < 1.0
    )  # Allow for small floating point differences


def test_end_of_life_impact_calculation(lca_analyzer):
    """Test end-of-life phase impact calculations."""
    material_inputs = {
        "silicon_wafer": 100.0,
        "solar_glass": 150.0,
        "aluminum_frame": 80.0,
    }
    recycling_rates = {"silicon_wafer": 0.8, "solar_glass": 0.9, "aluminum_frame": 0.95}
    transport_distance = 100.0  # km

    impact = lca_analyzer.calculate_end_of_life_impact(
        material_inputs, recycling_rates, transport_distance
    )

    assert isinstance(impact, float)

    # Calculate expected recycling benefit correctly
    expected_recycling_benefit = sum(
        qty * recycling_rates[mat] * RECYCLING_BENEFIT_FACTORS.get(mat, 0)
        for mat, qty in material_inputs.items()
    )

    # Note: Impact will be greater than recycling benefit due to transport
    assert impact > expected_recycling_benefit


def test_full_lca_calculation(lca_analyzer):
    """Test complete LCA calculation."""
    material_inputs = {
        "silicon_wafer": 100.0,
        "solar_glass": 150.0,
        "aluminum_frame": 80.0,
    }

    result = lca_analyzer.perform_full_lca(
        material_inputs=material_inputs,
        energy_consumption=1000.0,
        lifetime_years=25,
        annual_energy_generation=1000.0,
        grid_carbon_intensity=0.5,
        recycling_rates={
            "silicon_wafer": 0.8,
            "solar_glass": 0.9,
            "aluminum_frame": 0.95,
        },
        transport_distance=100.0,
    )

    assert isinstance(result, LifeCycleImpact)
    assert hasattr(result, "manufacturing_impact")
    assert hasattr(result, "use_phase_impact")
    assert hasattr(result, "end_of_life_impact")
    assert hasattr(result, "total_carbon_footprint")
    assert result.manufacturing_impact > 0
    assert result.use_phase_impact < 0  # Should be negative (benefit)
    assert isinstance(result.total_carbon_footprint, float)


def test_lifecycle_impact_total_calculation():
    """Test LifeCycleImpact total calculation."""
    impact = LifeCycleImpact(
        manufacturing_impact=100.0,
        use_phase_impact=-200.0,
        end_of_life_impact=50.0,
    )

    assert impact.total_carbon_footprint == -50.0
    assert impact.total_impact == -50.0
    # Test dictionary conversion
    impact_dict = impact.to_dict()
    assert all(
        k in impact_dict
        for k in [
            "Manufacturing Impact",
            "Use Phase Impact",
            "End of Life Impact",
            "Total Carbon Footprint",
        ]
    )


def test_lca_results_saving(lca_analyzer):
    """Test saving LCA results to file."""
    impact = LifeCycleImpact(
        manufacturing_impact=100.0,
        use_phase_impact=-200.0,
        end_of_life_impact=50.0,
    )

    batch_id = "TEST_001"

    # Get reports directory from ResultsManager
    reports_dir = results_manager.get_path("reports")

    # Save results with explicit output directory
    lca_analyzer.save_results(
        impact,
        batch_id=batch_id,
        output_dir=reports_dir,
    )

    # Verify file exists in correct location
    expected_path = reports_dir / f"lca_impact_{batch_id}.xlsx"
    assert expected_path.exists()

    # Verify file contents
    df = pd.read_excel(expected_path)
    assert "Manufacturing Impact" in df.columns
    assert "Use Phase Impact" in df.columns
    assert "End of Life Impact" in df.columns
    assert "Total Carbon Footprint" in df.columns


def test_error_handling(lca_analyzer):
    """Test error handling for invalid inputs."""
    # Test with empty material inputs
    with pytest.raises(ValueError, match="Material inputs dictionary cannot be empty"):
        lca_analyzer.calculate_manufacturing_impact({}, 1000.0)

    # Test with negative energy consumption
    with pytest.raises(ValueError, match="Energy consumption cannot be negative"):
        lca_analyzer.calculate_manufacturing_impact({"silicon_wafer": 100.0}, -1000.0)

    # Test with negative material quantity
    with pytest.raises(ValueError, match="Material quantity cannot be negative"):
        lca_analyzer.calculate_manufacturing_impact({"silicon_wafer": -100.0}, 1000.0)

    # Test with invalid use phase inputs
    with pytest.raises(ValueError, match="Annual energy generation cannot be negative"):
        lca_analyzer._calculate_use_phase_impact(
            annual_generation=-1000.0, lifetime=25, grid_intensity=0.5
        )


def test_detailed_impacts_calculation(lca_analyzer):
    """Test detailed environmental impacts calculation."""
    material_inputs = {
        "silicon_wafer": 100.0,
        "solar_glass": 150.0,
        "aluminum_frame": 80.0,
    }
    energy_consumption = 1000.0

    impacts = lca_analyzer.calculate_detailed_impacts(
        material_inputs=material_inputs, energy_consumption=energy_consumption
    )

    # Verify dictionary structure
    assert isinstance(impacts, dict)
    expected_keys = {
        "ghg_emissions",
        "water_consumption",
        "energy_consumption",
        "material_intensity",
        "waste_generation",
    }
    assert all(key in impacts for key in expected_keys)

    # Verify values
    assert impacts["energy_consumption"] == energy_consumption
    assert impacts["material_intensity"] == sum(material_inputs.values())
    assert impacts["ghg_emissions"] > 0
    assert impacts["water_consumption"] > 0
    assert impacts["waste_generation"] > 0

    # Verify GHG emissions calculation
    expected_material_emissions = sum(
        qty * MATERIAL_IMPACT_FACTORS.get(mat, 0)
        for mat, qty in material_inputs.items()
    )
    expected_energy_emissions = (
        energy_consumption * ENERGY_IMPACT_FACTORS["grid_electricity"]
    )
    expected_total_emissions = expected_material_emissions + expected_energy_emissions

    assert abs(impacts["ghg_emissions"] - expected_total_emissions) < 0.01
