# tests/integration/test_data_pipeline.py
"""Integration tests for analysis module pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from circman5.utils.results_manager import results_manager
from circman5.manufacturing.analyzers.efficiency import EfficiencyAnalyzer
from circman5.manufacturing.analyzers.quality import QualityAnalyzer
from circman5.manufacturing.analyzers.sustainability import SustainabilityAnalyzer
from circman5.manufacturing.lifecycle import LCAAnalyzer
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture(scope="module")
def analysis_pipeline():
    """Create instances of all analyzers."""
    return {
        "efficiency": EfficiencyAnalyzer(),
        "quality": QualityAnalyzer(),
        "sustainability": SustainabilityAnalyzer(),
        "lca": LCAAnalyzer(),
    }


@pytest.fixture(scope="module")
def test_run_dir():
    """Get test run directory from ResultsManager."""
    # Get paths for different outputs
    run_dir = results_manager.get_run_dir()
    paths = {
        "analysis": run_dir / "analysis",
        "results": run_dir / "results",
        "reports": run_dir / "reports",
    }

    # Create directories
    for dir_path in paths.values():
        dir_path.mkdir(exist_ok=True)

    return run_dir


def test_efficiency_analysis(analysis_pipeline, test_data):
    """Test efficiency analysis pipeline."""
    efficiency = analysis_pipeline["efficiency"]

    # Analyze batch efficiency
    metrics = efficiency.analyze_batch_efficiency(test_data["production"])

    # Verify metrics
    assert "yield_rate" in metrics, "Missing yield rate"
    assert "cycle_time_efficiency" in metrics, "Missing cycle time efficiency"
    assert "energy_efficiency" in metrics, "Missing energy efficiency"

    # Validate metric values
    assert 0 <= metrics["yield_rate"] <= 100, "Invalid yield rate"
    assert metrics["cycle_time_efficiency"] > 0, "Invalid cycle time efficiency"
    assert metrics["energy_efficiency"] > 0, "Invalid energy efficiency"


def test_quality_analysis(analysis_pipeline, test_data):
    """Test quality analysis pipeline."""
    quality = analysis_pipeline["quality"]

    # Analyze defect rates
    metrics = quality.analyze_defect_rates(test_data["quality"])

    # Verify metrics
    assert "avg_defect_rate" in metrics, "Missing defect rate"
    assert "efficiency_score" in metrics, "Missing efficiency score"
    assert "uniformity_score" in metrics, "Missing uniformity score"

    # Validate metric ranges
    assert 0 <= metrics["avg_defect_rate"] <= 100, "Invalid defect rate"
    assert 0 <= metrics["efficiency_score"] <= 100, "Invalid efficiency score"
    assert 0 <= metrics["uniformity_score"] <= 100, "Invalid uniformity score"

    # Test quality trends
    trends = quality.identify_quality_trends(test_data["quality"])
    assert len(trends) > 0, "No quality trends identified"


def test_sustainability_analysis(analysis_pipeline, test_data):
    """Test sustainability analysis pipeline."""
    sustainability = analysis_pipeline["sustainability"]

    # Calculate carbon footprint
    carbon_footprint = sustainability.calculate_carbon_footprint(test_data["energy"])
    assert carbon_footprint >= 0, "Invalid carbon footprint"

    # Analyze material efficiency
    material_metrics = sustainability.analyze_material_efficiency(test_data["material"])

    # Verify material metrics
    assert "material_efficiency" in material_metrics, "Missing material efficiency"
    assert "recycling_rate" in material_metrics, "Missing recycling rate"
    assert "waste_reduction" in material_metrics, "Missing waste reduction"

    # Calculate overall sustainability score
    score = sustainability.calculate_sustainability_score(
        material_metrics["material_efficiency"],
        material_metrics["recycling_rate"],
        90.0,  # Example energy efficiency
    )
    assert 0 <= score <= 100, "Invalid sustainability score"


def test_lca_analysis(analysis_pipeline, test_data, test_run_dir):
    """Test Life Cycle Assessment analysis pipeline."""
    lca = analysis_pipeline["lca"]
    lca_data = test_data["lca"]

    # Prepare material inputs
    material_inputs = {
        "silicon": lca_data["material_flow"]["quantity_used"].sum(),
        "glass": lca_data["material_flow"]["quantity_used"].sum(),
    }

    # Perform full LCA
    impact = lca.perform_full_lca(
        material_inputs=material_inputs,
        energy_consumption=lca_data["energy_consumption"]["energy_consumption"].sum(),
        lifetime_years=25.0,
        annual_energy_generation=2000.0,
        grid_carbon_intensity=0.5,
        recycling_rates={"silicon": 0.8, "glass": 0.9},
        transport_distance=100.0,
    )

    # Create the results DataFrame
    results_df = pd.DataFrame(
        {
            "Parameter": ["Manufacturing Impact", "Use Phase Impact", "Total Impact"],
            "Value": [
                impact.manufacturing_impact,
                impact.use_phase_impact,
                impact.total_impact,
            ],
        }
    )

    # Get results directory and save file
    results_dir = results_manager.get_path("lca_results")
    output_path = results_dir / "lca_impact.xlsx"

    # Save results
    results_df.to_excel(output_path, index=False)

    # Verify impact calculations
    assert impact.manufacturing_impact >= 0, "Invalid manufacturing impact"
    assert (
        impact.use_phase_impact < 0
    ), "Invalid use phase impact"  # Should be negative (benefit)
    assert isinstance(impact.total_impact, float), "Invalid total impact"
    assert output_path.exists(), "LCA results file not created"


def test_analysis_integration(analysis_pipeline, test_data, test_run_dir):
    """Test integration between different analysis components."""
    # Get efficiency metrics
    efficiency_metrics = analysis_pipeline["efficiency"].analyze_batch_efficiency(
        test_data["production"]
    )

    # Use efficiency results in sustainability calculation
    sustainability_score = analysis_pipeline[
        "sustainability"
    ].calculate_sustainability_score(
        material_efficiency=90.0,
        recycling_rate=85.0,
        energy_efficiency=efficiency_metrics["energy_efficiency"],
    )
    assert 0 <= sustainability_score <= 100, "Invalid integrated sustainability score"

    # Combine with quality metrics
    quality_metrics = analysis_pipeline["quality"].analyze_defect_rates(
        test_data["quality"]
    )

    # Create combined analysis
    output_data = pd.DataFrame(
        {
            "efficiency_score": [efficiency_metrics["yield_rate"]],
            "quality_score": [quality_metrics["efficiency_score"]],
            "sustainability_score": [sustainability_score],
        }
    )

    # Save using ResultsManager
    metrics_path = results_manager.get_path("metrics") / "combined_metrics.csv"
    output_data.to_csv(metrics_path, index=False)
    assert metrics_path.exists(), "Combined metrics file not created"


def test_data_consistency(analysis_pipeline, test_data):
    """Test consistency of analysis results across modules."""
    # Get metrics from all analyzers
    efficiency_metrics = analysis_pipeline["efficiency"].analyze_batch_efficiency(
        test_data["production"]
    )
    quality_metrics = analysis_pipeline["quality"].analyze_defect_rates(
        test_data["quality"]
    )
    material_metrics = analysis_pipeline["sustainability"].analyze_material_efficiency(
        test_data["material"]
    )

    # Verify metric ranges are consistent
    for metric, value in efficiency_metrics.items():
        if not isinstance(value, (int, float)):
            continue

        if metric == "energy_efficiency":
            # Energy efficiency is a ratio between 0 and 1
            assert 0 <= value <= 1, f"Energy efficiency {value} not in range [0,1]"
        elif metric == "input_amount":
            # Allow a larger tolerance for input_amount
            tolerance = 5.0  # Increased tolerance
            assert (
                0 <= value <= 105
            ), f"Metric {metric} value {value} not in expected range"
