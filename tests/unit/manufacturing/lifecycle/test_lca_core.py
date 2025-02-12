# tests/unit/lca/test_lca_core.py
"""Unit tests for LCA analysis module."""

import pytest
from pathlib import Path
from circman5.manufacturing.lifecycle import LCAAnalyzer, LifeCycleImpact
from circman5.config.project_paths import project_paths
from circman5.manufacturing.lifecycle.impact_factors import (
    MATERIAL_IMPACT_FACTORS,
    ENERGY_IMPACT_FACTORS,
    RECYCLING_BENEFIT_FACTORS,
)


@pytest.fixture
def lca_analyzer():
    """Create LCAAnalyzer instance for testing."""
    return LCAAnalyzer()


def test_manufacturing_impact_calculation(lca_analyzer):
    """Test manufacturing phase impact calculations."""
    material_inputs = {"silicon_wafer": 100.0, "solar_glass": 150.0, "eva_sheet": 50.0}
    energy_consumption = 1000.0  # kWh

    impact = lca_analyzer.calculate_manufacturing_impact(
        material_inputs, energy_consumption
    )

    assert impact > 0
    assert isinstance(impact, float)


def test_use_phase_impact_calculation(lca_analyzer):
    """Test use phase impact calculations."""
    impact = lca_analyzer.calculate_use_phase_impact(
        lifetime_years=25, annual_energy_generation=1000.0, grid_carbon_intensity=0.5
    )

    # Use phase impact should be negative (environmental benefit)
    assert impact < 0
    assert isinstance(impact, float)


def test_end_of_life_impact_calculation(lca_analyzer):
    """Test end-of-life phase impact calculations."""
    material_quantities = {"silicon_wafer": 100.0, "solar_glass": 150.0}
    recycling_rates = {"silicon_wafer": 0.8, "solar_glass": 0.9}
    transport_distance = 100.0

    impact = lca_analyzer.calculate_end_of_life_impact(
        material_quantities, recycling_rates, transport_distance
    )

    assert isinstance(impact, float)


def test_full_lca_calculation(lca_analyzer):
    """Test complete LCA calculation."""
    material_inputs = {"silicon_wafer": 100.0, "solar_glass": 150.0}

    result = lca_analyzer.perform_full_lca(
        material_inputs=material_inputs,
        energy_consumption=1000.0,
        lifetime_years=25,
        annual_energy_generation=1000.0,
        grid_carbon_intensity=0.5,
        recycling_rates={"silicon_wafer": 0.8, "solar_glass": 0.9},
        transport_distance=100.0,
    )

    assert isinstance(result, LifeCycleImpact)
    assert hasattr(result, "manufacturing_impact")
    assert hasattr(result, "use_phase_impact")
    assert hasattr(result, "end_of_life_impact")
    assert hasattr(result, "total_carbon_footprint")


def test_lifecycle_impact_total_calculation():
    """Test LifeCycleImpact total calculation."""
    impact = LifeCycleImpact(
        manufacturing_impact=100.0,
        use_phase_impact=-200.0,
        end_of_life_impact=50.0,
        total_carbon_footprint=-50.0,
    )

    assert impact.total_impact == -50.0


def test_lca_results_saving(lca_analyzer):
    """Test saving LCA results to file."""
    impact = LifeCycleImpact(
        manufacturing_impact=100.0,
        use_phase_impact=200.0,
        end_of_life_impact=50.0,
        total_carbon_footprint=350.0,
    )

    lca_analyzer.save_results(impact, "TEST_001")
    run_dir = project_paths.get_run_directory()
    assert (run_dir / "reports" / "lca_impact_TEST_001.xlsx").exists()
