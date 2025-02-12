# tests/unit/manufacturing/test_core.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.config.project_paths import project_paths
from circman5.utils.errors import ProcessError


@pytest.fixture
def test_data():
    """Generate test data for manufacturing tests."""
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=5)
    return {
        "production": generator.generate_production_data(),
        "quality": generator.generate_quality_data(),
        "energy": generator.generate_energy_data(),
        "material": generator.generate_material_flow_data(),
    }


@pytest.fixture
def analyzer():
    """Create analyzer instance for testing."""
    return SoliTekManufacturingAnalysis()


class TestManufacturingCore:
    """Test core manufacturing analysis functionality."""

    def setup_method(self):
        self.analyzer = SoliTekManufacturingAnalysis()

    def initialize_test_data(self, analyzer, test_data):
        """Helper method to initialize test data."""
        analyzer.production_data = test_data["production"]
        analyzer.quality_data = test_data["quality"]
        analyzer.energy_data = test_data["energy"]
        analyzer.material_flow = test_data["material"]

    def test_data_loading(self, analyzer, test_data, tmp_path):
        """Test data loading from files."""
        # Save test data to temporary files
        production_path = tmp_path / "production.csv"
        test_data["production"].to_csv(production_path, index=False)

        # Test loading
        analyzer.load_data(production_path=str(production_path))
        assert not analyzer.production_data.empty
        assert "batch_id" in analyzer.production_data.columns

    def test_manufacturing_analysis(self, analyzer, test_data):
        """Test complete manufacturing analysis pipeline."""
        self.initialize_test_data(analyzer, test_data)

        # Perform analysis
        results = analyzer.analyze_manufacturing_performance()

        # Verify results structure
        assert isinstance(results, dict)
        assert "efficiency" in results
        assert "quality" in results
        assert "sustainability" in results

        # Verify metric values
        efficiency = results["efficiency"]
        assert 0 <= efficiency.get("yield_rate", 0) <= 100
        assert efficiency.get("energy_efficiency", 0) > 0

    def test_report_generation(self, analyzer, test_data, tmp_path):
        """Test report generation functionality."""
        self.initialize_test_data(analyzer, test_data)

        # Generate report
        report_path = tmp_path / "analysis_report.xlsx"
        analyzer.export_analysis_report(report_path)

        # Verify report
        assert report_path.exists()

    def test_visualization_generation(self, analyzer, test_data, tmp_path):
        """Test visualization generation."""
        self.initialize_test_data(analyzer, test_data)

        viz_path = tmp_path / "production.png"
        analyzer.generate_visualization("production", str(viz_path))
        assert viz_path.exists()

    def test_error_handling(self, analyzer):
        """Test error handling for invalid inputs."""
        with pytest.raises(ProcessError) as exc_info:
            analyzer.generate_visualization("invalid_type", "test.png")
        # Optionally verify the error message
        assert "Unknown metric type" in str(exc_info.value)
