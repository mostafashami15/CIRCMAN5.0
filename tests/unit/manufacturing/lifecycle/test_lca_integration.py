# tests/unit/manufacturing/lifecycle/test_lca_integration.py
"""Integration tests for LCA functionality in manufacturing analysis."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from circman5.utils.errors import ProcessError


def test_lca_data_loading(
    manufacturing_analyzer, sample_lca_data, input_data_dir, copy_synthetic_input_files
):
    # Now the input_data_dir contains material_data.csv, energy_data.csv, process_data.csv.
    material_path = input_data_dir / "material_data.csv"
    energy_path = input_data_dir / "energy_data.csv"
    process_path = input_data_dir / "process_data.csv"

    manufacturing_analyzer.load_lca_data(
        material_data_path=str(material_path),
        energy_data_path=str(energy_path),
        process_data_path=str(process_path),
    )

    assert not manufacturing_analyzer.lca_data["material_flow"].empty
    assert not manufacturing_analyzer.lca_data["energy_consumption"].empty
    assert not manufacturing_analyzer.lca_data["process_data"].empty


def test_recycling_rates_calculation(manufacturing_analyzer, sample_lca_data):
    """Test calculation of recycling rates from material data."""
    manufacturing_analyzer.lca_data = sample_lca_data

    # Calculate recycling rates
    recycling_rates = manufacturing_analyzer.lca_analyzer.calculate_recycling_rates(
        manufacturing_analyzer.lca_data["material_flow"]
    )

    # Verify results
    assert isinstance(recycling_rates, dict)
    assert all(0 <= rate <= 1 for rate in recycling_rates.values())
    assert len(recycling_rates) > 0


def test_lifecycle_assessment(manufacturing_analyzer, sample_lca_data):
    """Test full lifecycle assessment calculation."""
    manufacturing_analyzer.lca_data = sample_lca_data

    # Perform lifecycle assessment
    impact = manufacturing_analyzer.perform_lifecycle_assessment()

    # Verify results
    assert hasattr(impact, "manufacturing_impact")
    assert hasattr(impact, "use_phase_impact")
    assert hasattr(impact, "end_of_life_impact")
    assert hasattr(impact, "total_carbon_footprint")

    # Verify logical relationships
    assert impact.manufacturing_impact > 0  # Manufacturing should have positive impact
    assert impact.use_phase_impact < 0  # Use phase should show environmental benefit
    assert isinstance(impact.total_carbon_footprint, float)


def test_lca_report_generation(
    manufacturing_analyzer, sample_lca_data, reports_dir, lca_results_dir
):
    """Test generation of LCA report."""
    manufacturing_analyzer.lca_data = sample_lca_data

    # First generate LCA report
    impact = manufacturing_analyzer.perform_lifecycle_assessment()
    manufacturing_analyzer.lca_analyzer.save_results(impact, output_dir=lca_results_dir)
    lca_report_path = lca_results_dir / "lca_impact.xlsx"
    assert lca_report_path.exists()

    # Then generate comprehensive report
    manufacturing_analyzer.generate_reports(output_dir=reports_dir)
    # Updated expected filename:
    report_path = reports_dir / "analysis_report.xlsx"
    assert report_path.exists()


def test_batch_specific_assessment(
    manufacturing_analyzer, sample_lca_data, test_data_generator, lca_results_dir
):
    """Test lifecycle assessment for specific batch."""
    # Ensure material data has meaningful solar glass quantity
    material_data = sample_lca_data["material_flow"].copy()
    material_data.loc[0, "material_type"] = "solar_glass"
    material_data.loc[0, "quantity_used"] = 1000.0
    sample_lca_data["material_flow"] = material_data

    manufacturing_analyzer.lca_data = sample_lca_data

    # Get a test batch ID
    test_batch = material_data["batch_id"].iloc[0]

    # Perform assessment for specific batch
    impact = manufacturing_analyzer.perform_lifecycle_assessment(
        batch_id=test_batch, output_dir=lca_results_dir  # Add output directory
    )

    # Verify results
    assert hasattr(impact, "manufacturing_impact")
    assert impact.manufacturing_impact > 0
    assert impact.use_phase_impact < 0
    assert isinstance(impact.total_carbon_footprint, float)

    # Verify outputs are in correct directory
    expected_files = [
        f"lifecycle_comparison_{test_batch}.png",
        f"material_flow_{test_batch}.png",
        f"energy_trends_{test_batch}.png",
    ]
    for filename in expected_files:
        file_path = lca_results_dir / filename
        assert file_path.exists(), f"Expected file not found: {file_path}"


def test_visualization_generation(
    manufacturing_analyzer, sample_lca_data, visualizations_dir
):
    """Test generation of LCA visualizations."""
    manufacturing_analyzer.lca_data = sample_lca_data

    # Perform assessment to generate data for visualization
    impact = manufacturing_analyzer.perform_lifecycle_assessment()

    # Generate visualizations
    manufacturing_analyzer.lca_visualizer.create_comprehensive_report(
        impact.to_dict(),
        manufacturing_analyzer.lca_data["material_flow"],
        manufacturing_analyzer.lca_data["energy_consumption"],
        output_dir=visualizations_dir,
    )

    # Verify files exist in correct location
    for viz_file in [
        "impact_distribution.png",
        "lifecycle_comparison.png",
        "material_flow.png",
        "energy_trends.png",
    ]:
        viz_path = visualizations_dir / viz_file
        assert viz_path.exists(), f"Visualization not found: {viz_path}"


def test_error_handling(manufacturing_analyzer):
    """Test error handling for invalid data scenarios."""
    # Test with empty data
    manufacturing_analyzer.lca_data = {
        "material_flow": pd.DataFrame(),
        "energy_consumption": pd.DataFrame(),
        "process_data": pd.DataFrame(),
    }

    # Verify appropriate error handling
    with pytest.raises(ProcessError) as exc_info:
        manufacturing_analyzer.perform_lifecycle_assessment()
    assert "No material flow data available" in str(exc_info.value)


def test_data_validation(manufacturing_analyzer, sample_lca_data):
    """Test data validation during LCA calculations."""
    manufacturing_analyzer.lca_data = sample_lca_data

    # Verify data validation
    material_data = manufacturing_analyzer.lca_data["material_flow"]
    assert not material_data.empty
    assert all(
        col in material_data.columns
        for col in [
            "material_type",
            "quantity_used",
            "waste_generated",
            "recycled_amount",
        ]
    )

    # Verify numerical constraints
    assert (material_data["quantity_used"] >= 0).all()
    assert (material_data["waste_generated"] >= 0).all()
    assert (material_data["recycled_amount"] >= 0).all()
