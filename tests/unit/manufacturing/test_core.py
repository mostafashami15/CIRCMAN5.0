# tests/unit/manufacturing/test_core.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.manufacturing.lifecycle.lca_analyzer import LifeCycleImpact, LCAAnalyzer
from circman5.manufacturing.analyzers.efficiency import EfficiencyAnalyzer
from circman5.manufacturing.analyzers.quality import QualityAnalyzer
from circman5.manufacturing.analyzers.sustainability import SustainabilityAnalyzer
from circman5.utils.errors import ProcessError, DataError


@pytest.fixture
def test_data():
    """Generate test data for manufacturing tests."""
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=5)
    return {
        "production": generator.generate_production_data(),
        "quality": generator.generate_quality_data(),
        "energy": generator.generate_energy_data(),
        "material": generator.generate_material_flow_data(),
        "process": generator.generate_lca_process_data(),
    }


@pytest.fixture
def analyzer():
    """Create analyzer instance for testing."""
    return SoliTekManufacturingAnalysis()


class TestManufacturingCore:
    """Test core manufacturing analysis functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.analyzer = SoliTekManufacturingAnalysis()

    def initialize_test_data(self, analyzer, test_data):
        """Helper method to initialize test data."""
        analyzer.production_data = test_data["production"]
        analyzer.quality_data = test_data["quality"]
        analyzer.energy_data = test_data["energy"]
        analyzer.material_flow = test_data["material"]
        analyzer.lca_data = {
            "material_flow": test_data["material"],
            "energy_consumption": test_data["energy"],
            "process_data": test_data["process"],
        }

    def test_initialization(self):
        """Test proper analyzer initialization."""
        assert isinstance(self.analyzer.efficiency_analyzer, EfficiencyAnalyzer)
        assert isinstance(self.analyzer.quality_analyzer, QualityAnalyzer)
        assert isinstance(self.analyzer.sustainability_analyzer, SustainabilityAnalyzer)
        assert isinstance(self.analyzer.lca_analyzer, LCAAnalyzer)
        assert not self.analyzer.is_optimizer_trained

    def test_data_loading(self, analyzer, test_data, tmp_path):
        """Test data loading from files."""
        # Save test data to temporary files
        paths = {}
        for data_type, df in test_data.items():
            path = tmp_path / f"{data_type}.csv"
            df.to_csv(path, index=False)
            paths[f"{data_type}_path"] = str(path)

        # Test loading
        analyzer.load_data(**paths)

        assert not analyzer.production_data.empty
        assert not analyzer.quality_data.empty
        assert not analyzer.energy_data.empty
        assert not analyzer.material_flow.empty
        assert "batch_id" in analyzer.production_data.columns

    def test_manufacturing_analysis(self, analyzer, test_data):
        """Test complete manufacturing analysis pipeline."""
        self.initialize_test_data(analyzer, test_data)

        # Perform analysis
        results = analyzer.analyze_manufacturing_performance()

        # Verify results structure and content
        assert isinstance(results, dict)
        assert all(
            key in results for key in ["efficiency", "quality", "sustainability"]
        )

        # Verify metric values
        efficiency = results["efficiency"]
        assert 0 <= efficiency.get("yield_rate", 0) <= 100
        assert efficiency.get("energy_efficiency", 0) > 0

        quality = results["quality"]
        assert 0 <= quality.get("defect_rate", 0) <= 100
        assert 0 <= quality.get("efficiency_score", 0) <= 100

    def test_lifecycle_assessment(self, analyzer, test_data):
        """Test lifecycle assessment functionality."""
        self.initialize_test_data(analyzer, test_data)

        # Test without batch_id
        impact = analyzer.perform_lifecycle_assessment()

        assert isinstance(impact, LifeCycleImpact)
        assert isinstance(impact.manufacturing_impact, float)
        assert isinstance(impact.use_phase_impact, float)
        assert isinstance(impact.end_of_life_impact, float)
        assert impact.total_carbon_footprint > 0

        # Test with specific batch_id
        batch_id = test_data["production"]["batch_id"].iloc[0]
        batch_impact = analyzer.perform_lifecycle_assessment(batch_id=batch_id)
        assert isinstance(batch_impact, LifeCycleImpact)

    def test_optimization_workflow(self, analyzer, test_data):
        """Test complete optimization workflow."""
        self.initialize_test_data(analyzer, test_data)

        # Train model
        metrics = analyzer.train_optimization_model()
        assert metrics["r2"] > 0
        assert analyzer.is_optimizer_trained

        # Test optimization
        current_params = {
            "input_amount": 100.0,
            "energy_used": 150.0,
            "cycle_time": 50.0,
            "efficiency": 95.0,
            "defect_rate": 2.0,
            "thickness_uniformity": 98.0,
        }

        optimized = analyzer.optimize_process_parameters(current_params)
        assert isinstance(optimized, dict)
        assert all(param in optimized for param in current_params)
        assert all(isinstance(value, float) for value in optimized.values())

        # Test prediction
        predictions = analyzer.predict_batch_outcomes(current_params)
        assert isinstance(predictions, dict)
        assert "predicted_output" in predictions
        assert "predicted_quality" in predictions

    def test_report_generation(self, analyzer, test_data, tmp_path):
        """Test report generation functionality."""
        self.initialize_test_data(analyzer, test_data)

        # Test comprehensive report
        report_path = tmp_path / "analysis_report.xlsx"
        analyzer.generate_comprehensive_report(str(report_path))
        assert report_path.exists()

        # Test analysis report export
        export_path = tmp_path / "export_report.xlsx"
        analyzer.export_analysis_report(export_path)
        assert export_path.exists()

    def test_visualization_generation(self, analyzer, test_data, tmp_path):
        """Test visualization generation for different metric types."""
        self.initialize_test_data(analyzer, test_data)

        for metric_type in ["production", "energy", "quality", "sustainability"]:
            viz_path = tmp_path / f"{metric_type}.png"
            analyzer.generate_visualization(metric_type, str(viz_path))
            assert viz_path.exists()

    def test_empty_data_handling(self, analyzer):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame()

        analyzer.production_data = empty_df
        with pytest.raises(DataError):
            analyzer.analyze_efficiency()

        analyzer.quality_data = empty_df
        with pytest.raises(DataError):
            analyzer.analyze_quality_metrics()

        analyzer.lca_data = {
            "material_flow": empty_df,
            "energy_consumption": empty_df,
            "process_data": empty_df,
        }
        with pytest.raises(ValueError):
            analyzer.perform_lifecycle_assessment()

    def test_invalid_data_handling(self, analyzer):
        """Test handling of invalid data."""
        invalid_data = pd.DataFrame(
            {
                "batch_id": ["B1"],
                "input_amount": [-100],  # Invalid negative value
                "output_amount": [50],
            }
        )

        analyzer.production_data = invalid_data
        with pytest.raises(ProcessError):
            analyzer.analyze_efficiency()

    def test_parameter_validation(self, analyzer):
        """Test validation of input parameters."""
        with pytest.raises(ValueError):
            analyzer.optimize_process_parameters({})  # Empty parameters

        with pytest.raises(ValueError):
            analyzer.predict_batch_outcomes({"invalid_param": 100})

    def test_batch_specific_analysis(self, analyzer, test_data):
        """Test batch-specific analysis capabilities."""
        self.initialize_test_data(analyzer, test_data)

        batch_id = test_data["production"]["batch_id"].iloc[0]

        # Test LCA for specific batch
        impact = analyzer.perform_lifecycle_assessment(batch_id=batch_id)
        assert isinstance(impact, LifeCycleImpact)

        # Verify batch filtering
        filtered = analyzer._filter_batch_data(test_data["production"], batch_id)
        assert all(filtered["batch_id"] == batch_id)

    def test_end_to_end_workflow(self, analyzer, test_data, tmp_path):
        """Test complete analysis workflow."""
        self.initialize_test_data(analyzer, test_data)

        # Perform analyses
        performance = analyzer.analyze_manufacturing_performance()
        impact = analyzer.perform_lifecycle_assessment()

        # Generate reports and visualizations
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        analyzer.generate_reports(report_dir)

        # Verify outputs
        assert (report_dir / "analysis_report.xlsx").exists()
        assert performance["efficiency"]["yield_rate"] > 0
        assert impact.total_carbon_footprint > 0

    @pytest.mark.performance
    def test_performance_metrics(self, analyzer, test_data):
        """Test performance of key operations."""
        self.initialize_test_data(analyzer, test_data)

        start_time = datetime.now()
        analyzer.analyze_manufacturing_performance()
        analysis_time = (datetime.now() - start_time).total_seconds()

        assert analysis_time < 5.0  # Analysis should complete within 5 seconds

    def test_data_consistency(self, analyzer, test_data):
        """Test data consistency across operations."""
        self.initialize_test_data(analyzer, test_data)

        # Get metrics through different methods
        performance = analyzer.analyze_manufacturing_performance()
        efficiency = analyzer.analyze_efficiency()
        quality = analyzer.analyze_quality_metrics()

        # Verify consistency
        assert performance["efficiency"]["yield_rate"] == efficiency["yield_rate"]
        assert performance["quality"]["defect_rate"] == quality["avg_defect_rate"]
