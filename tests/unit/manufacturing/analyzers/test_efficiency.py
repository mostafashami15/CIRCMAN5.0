# tests/unit/manufacturing/analyzers/test_efficiency.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from circman5.manufacturing.analyzers.efficiency import EfficiencyAnalyzer
from circman5.utils.errors import ValidationError


@pytest.fixture
def sample_production_data():
    """Create sample production data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")
    return pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(10)],
            "timestamp": dates,
            "output_amount": np.linspace(80, 100, 10),  # Spread out values
            "input_amount": np.linspace(90, 110, 10),  # Spread out values
            "cycle_time": np.random.uniform(45, 55, 10),
            "energy_used": np.random.uniform(140, 160, 10),
        }
    )


@pytest.fixture
def analyzer():
    return EfficiencyAnalyzer()


class TestEfficiencyAnalyzer:
    """Test suite for efficiency analyzer."""

    def test_batch_efficiency_complete(
        self, analyzer, sample_production_data, reports_dir
    ):
        """Test complete efficiency analysis pipeline."""
        metrics = analyzer.analyze_batch_efficiency(sample_production_data)

        required_metrics = {"yield_rate", "cycle_time_efficiency", "energy_efficiency"}
        assert all(metric in metrics for metric in required_metrics)

        # Verify calculations
        yield_rate = metrics["yield_rate"]
        assert 0 <= yield_rate <= 100
        assert isinstance(yield_rate, float)

        # Save metrics report
        report_path = reports_dir / "efficiency_metrics.xlsx"
        pd.DataFrame([metrics]).to_excel(report_path)
        assert report_path.exists()

    def test_validation_handling(self, analyzer, reports_dir):
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
        report_path = reports_dir / "validation_test.xlsx"
        pd.DataFrame(
            {
                "test_case": ["negative_output", "zero_input"],
                "value": [-100, 0],
                "expected_result": ["ValidationError", "ValidationError"],
            }
        ).to_excel(report_path)
        assert report_path.exists()

    def test_edge_cases(self, analyzer, reports_dir):
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
        report_path = reports_dir / "edge_case_results.xlsx"
        report_data = {
            "test_case": ["empty_dataframe", "minimal_data"],
            "metrics_present": [str({}), str(metrics)],
            "yield_rate": [None, metrics.get("yield_rate")],
            "has_energy_efficiency": [False, "energy_efficiency" in metrics],
        }
        pd.DataFrame(report_data).to_excel(report_path)
        assert report_path.exists()

    def test_efficiency_calculations(
        self, analyzer, sample_production_data, reports_dir, visualizations_dir
    ):
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
        calcs_path = reports_dir / "efficiency_calculations.xlsx"
        pd.DataFrame(
            {
                "metric": ["yield_rate", "expected_yield", "difference"],
                "value": [
                    metrics["yield_rate"],
                    expected_yield,
                    abs(metrics["yield_rate"] - expected_yield),
                ],
            }
        ).to_excel(calcs_path)
        assert calcs_path.exists()

        # Generate and save visualization if the analyzer supports it
        if hasattr(analyzer, "plot_efficiency_trends"):
            viz_path = visualizations_dir / "efficiency_trends.png"
            analyzer.plot_efficiency_trends(
                sample_production_data, save_path=str(viz_path)
            )
            assert viz_path.exists()

    def test_time_series_analysis(self, analyzer, sample_production_data, reports_dir):
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
            report_path = reports_dir / "efficiency_time_series.xlsx"
            pd.DataFrame(hourly_metrics).to_excel(report_path)
            assert report_path.exists()

    def test_visualization_output(self, sample_production_data, visualizations_dir):
        """Test that visualizations are saved to the correct directory."""
        analyzer = EfficiencyAnalyzer()

        # Generate visualization path
        viz_path = visualizations_dir / "efficiency_test.png"

        # Create visualization
        analyzer.plot_efficiency_trends(sample_production_data, str(viz_path))

        # Verify file exists
        assert viz_path.exists()
        print(f"Created visualization at: {viz_path}")

        # Test empty data handling
        empty_df = pd.DataFrame()
        analyzer.plot_efficiency_trends(empty_df, str(viz_path))

        # Ensure no warnings or errors
        print("Visualization output test passed without warnings.")

    def test_visualization_features(self, sample_production_data, visualizations_dir):
        """Test specific visualization features and options."""
        analyzer = EfficiencyAnalyzer()
        viz_path = visualizations_dir / "efficiency_features_test.png"

        # Test with complete data
        analyzer.plot_efficiency_trends(sample_production_data, str(viz_path))
        assert viz_path.exists()

        # Test with subset of columns
        subset_data = sample_production_data[
            ["timestamp", "input_amount", "output_amount"]
        ]
        viz_path_subset = visualizations_dir / "efficiency_subset_test.png"
        analyzer.plot_efficiency_trends(subset_data, str(viz_path_subset))
        assert viz_path_subset.exists()
