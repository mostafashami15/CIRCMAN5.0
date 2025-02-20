# tests/unit/manufacturing/analyzers/test_efficiency.py

"""Test suite for efficiency analyzer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from circman5.manufacturing.analyzers.efficiency import EfficiencyAnalyzer
from circman5.utils.errors import ValidationError
from circman5.utils.results_manager import results_manager


@pytest.fixture
def sample_production_data():
    """Create sample production data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")
    return pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(10)],
            "timestamp": dates,
            "output_amount": np.linspace(80, 100, 10),
            "input_amount": np.linspace(90, 110, 10),
            "cycle_time": np.random.uniform(45, 55, 10),
            "energy_used": np.random.uniform(140, 160, 10),
        }
    )


@pytest.fixture
def analyzer():
    return EfficiencyAnalyzer()


class TestEfficiencyAnalyzer:
    """Test suite for efficiency analyzer."""

    def test_batch_efficiency_complete(self, analyzer, sample_production_data):
        """Test complete efficiency analysis pipeline."""
        metrics = analyzer.analyze_batch_efficiency(sample_production_data)

        required_metrics = {"yield_rate", "cycle_time_efficiency", "energy_efficiency"}
        assert all(metric in metrics for metric in required_metrics)

        # Verify calculations
        yield_rate = metrics["yield_rate"]
        assert 0 <= yield_rate <= 100
        assert isinstance(yield_rate, float)

        # Save metrics report
        temp_path = Path("efficiency_metrics.xlsx")
        pd.DataFrame([metrics]).to_excel(temp_path)
        results_manager.save_file(temp_path, "reports")
        temp_path.unlink()

    def test_validation_handling(self, analyzer):
        """Test validation of input data."""
        invalid_data = pd.DataFrame(
            {
                "batch_id": ["TEST"],
                "output_amount": [-100],  # Invalid negative value
                "input_amount": [0],  # Invalid zero value
            }
        )

        # Test validation error
        with pytest.raises(ValidationError):
            analyzer.analyze_batch_efficiency(invalid_data)

        # Save validation test results
        temp_path = Path("validation_test.xlsx")
        pd.DataFrame(
            {
                "test_case": ["negative_output", "zero_input"],
                "value": [-100, 0],
                "expected_result": ["ValidationError", "ValidationError"],
            }
        ).to_excel(temp_path)
        results_manager.save_file(temp_path, "reports")
        temp_path.unlink()

    def test_edge_cases(self, analyzer):
        """Test handling of edge cases."""
        # Empty DataFrame
        assert analyzer.analyze_batch_efficiency(pd.DataFrame()) == {}

        # Missing optional columns
        minimal_data = pd.DataFrame(
            {
                "batch_id": ["TEST"],
                "output_amount": [90],
                "input_amount": [100],
            }
        )
        metrics = analyzer.analyze_batch_efficiency(minimal_data)
        assert "yield_rate" in metrics
        assert "energy_efficiency" not in metrics  # Optional metric

        # Save edge case results
        temp_path = Path("edge_case_results.xlsx")
        report_data = {
            "test_case": ["empty_dataframe", "minimal_data"],
            "metrics_present": [str({}), str(metrics)],
            "yield_rate": [None, metrics.get("yield_rate")],
            "has_energy_efficiency": [False, "energy_efficiency" in metrics],
        }
        pd.DataFrame(report_data).to_excel(temp_path)
        results_manager.save_file(temp_path, "reports")
        temp_path.unlink()

    def test_efficiency_calculations(self, analyzer, sample_production_data):
        """Test detailed efficiency calculations and visualization."""
        metrics = analyzer.analyze_batch_efficiency(sample_production_data)

        # Calculate expected yield rate manually
        expected_yield = (
            sample_production_data["output_amount"].mean()
            / sample_production_data["input_amount"].mean()
            * 100
        )

        assert abs(metrics["yield_rate"] - expected_yield) < 0.01

        # Save calculation details
        temp_path = Path("efficiency_calculations.xlsx")
        pd.DataFrame(
            {
                "metric": ["yield_rate", "expected_yield", "difference"],
                "value": [
                    metrics["yield_rate"],
                    expected_yield,
                    abs(metrics["yield_rate"] - expected_yield),
                ],
            }
        ).to_excel(temp_path)
        results_manager.save_file(temp_path, "reports")
        if temp_path.exists():
            temp_path.unlink()

        # Generate and save visualization
        if hasattr(analyzer, "plot_efficiency_trends"):
            temp_viz = Path("efficiency_trends.png")
            analyzer.plot_efficiency_trends(sample_production_data, str(temp_viz))
            # The file will be automatically cleaned up by plot_efficiency_trends

    def test_time_series_analysis(self, analyzer, sample_production_data):
        """Test efficiency analysis over time."""
        # Group by hour and calculate metrics
        hourly_metrics = []
        for _, group in sample_production_data.groupby(
            pd.Grouper(key="timestamp", freq="H")
        ):
            if not group.empty:
                metrics = analyzer.analyze_batch_efficiency(group)
                hourly_metrics.append(
                    {"timestamp": group["timestamp"].iloc[0], **metrics}
                )

        # Save time series analysis
        if hourly_metrics:
            temp_path = Path("efficiency_time_series.xlsx")
            pd.DataFrame(hourly_metrics).to_excel(temp_path)
            results_manager.save_file(temp_path, "reports")
            temp_path.unlink()

    def test_visualization_output(self, analyzer, sample_production_data):
        """Test that visualizations are saved to the correct directory."""
        temp_viz = Path("efficiency_test.png")

        # Generate visualization
        analyzer.plot_efficiency_trends(sample_production_data, str(temp_viz))
        if temp_viz.exists():  # Only try to save and delete if file exists
            results_manager.save_file(temp_viz, "visualizations")
            temp_viz.unlink()

        # Test empty data handling
        empty_df = pd.DataFrame()
        analyzer.plot_efficiency_trends(empty_df, str(temp_viz))
        if temp_viz.exists():  # Only try to save and delete if file exists
            results_manager.save_file(temp_viz, "visualizations")
            temp_viz.unlink()

    def test_visualization_features(self, analyzer, sample_production_data):
        """Test specific visualization features and options."""
        # Test with complete data
        temp_viz = Path("efficiency_features_test.png")
        analyzer.plot_efficiency_trends(sample_production_data, str(temp_viz))
        # No need to unlink as plot_efficiency_trends handles cleanup

        # Test with subset of columns
        subset_data = sample_production_data[
            ["timestamp", "input_amount", "output_amount"]
        ]
        temp_subset = Path("efficiency_subset_test.png")
        analyzer.plot_efficiency_trends(subset_data, str(temp_subset))
        # No need to unlink as plot_efficiency_trends handles cleanup
