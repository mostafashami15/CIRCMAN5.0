"""Test suite for SoliTek Manufacturing Analysis System."""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator


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


class TestPVManufacturing:
    """Test cases for PV Manufacturing System"""

    def setup_method(self):
        """Setup test cases"""
        self.pv_system = SoliTekManufacturingAnalysis()
        self.test_batch_id = "TEST_BATCH_001"

    def initialize_test_data(self, test_data):
        """Helper method to initialize test data."""
        self.pv_system.production_data = test_data["production"]
        self.pv_system.quality_data = test_data["quality"]
        self.pv_system.energy_data = test_data["energy"]
        self.pv_system.material_flow = test_data["material"]

    def test_batch_initialization(self, test_data):
        """Test batch creation and initialization"""
        self.initialize_test_data(test_data)
        test_data = self.pv_system.production_data.copy()

        # Verify initial state
        assert not test_data.empty, "Production data should not be empty"
        assert "batch_id" in test_data.columns, "batch_id column should exist"
        assert "stage" in test_data.columns, "stage column should exist"
        assert len(test_data) > 0, "Should contain production records"

    def test_quality_control(self, test_data):
        """Test quality control measurements"""
        self.initialize_test_data(test_data)

        # Analyze quality metrics
        quality_metrics = self.pv_system.analyze_quality_metrics()

        # Assert quality measurements are calculated
        assert isinstance(quality_metrics, dict), "Should return metrics dictionary"
        assert not "error" in quality_metrics, "Should not return error"
        assert "avg_defect_rate" in quality_metrics, "Should include defect rate"
        assert "efficiency_score" in quality_metrics, "Should include efficiency score"
        assert (
            quality_metrics["avg_defect_rate"] >= 0
        ), "Defect rate should be non-negative"
        assert (
            quality_metrics["efficiency_score"] >= 0
        ), "Efficiency score should be non-negative"

    def test_sustainability_metrics(self, test_data):
        """Test sustainability metrics recording and calculation"""
        self.initialize_test_data(test_data)

        # Calculate sustainability metrics
        metrics = self.pv_system.calculate_sustainability_metrics()

        # Verify metrics
        assert isinstance(metrics, dict), "Should return metrics dictionary"
        assert not "error" in metrics, "Should not return error"
        assert "material_efficiency" in metrics, "Should include material efficiency"
        assert "recycling_rate" in metrics, "Should include recycling rate"
        assert (
            metrics["material_efficiency"] >= 0
        ), "Material efficiency should be non-negative"
        assert metrics["recycling_rate"] >= 0, "Recycling rate should be non-negative"

    def test_efficiency_analysis(self, test_data):
        """Test efficiency analysis process"""
        self.initialize_test_data(test_data)

        # Perform efficiency analysis
        efficiency_metrics = self.pv_system.analyze_efficiency()

        # Verify analysis results
        assert isinstance(efficiency_metrics, dict), "Should return metrics dictionary"
        assert not "error" in efficiency_metrics, "Should not return error"
        assert "yield_rate" in efficiency_metrics, "Should include yield rate"
        assert (
            efficiency_metrics["yield_rate"] >= 0
        ), "Yield rate should be non-negative"
        assert (
            "energy_efficiency" in efficiency_metrics
        ), "Should include energy efficiency"

    def test_visualization_generation(self, test_data, tmp_path):
        """Test visualization generation"""
        self.initialize_test_data(test_data)
        test_path = tmp_path / "test_visualization.png"

        # Generate visualization
        self.pv_system.generate_visualization("production", str(test_path))

        # Verify file was created
        assert test_path.exists(), "Visualization file should be created"

    def test_error_handling(self, test_data, tmp_path):
        """Test error handling in the system"""
        self.initialize_test_data(test_data)

        # Test invalid metric type
        with pytest.raises(Exception) as exc_info:
            self.pv_system.generate_visualization(
                "invalid_metric", str(tmp_path / "test.png")
            )
        assert "Unknown metric type" in str(exc_info.value)

        # Test invalid file path
        with pytest.raises(Exception) as exc_info:
            self.pv_system.load_production_data("/nonexistent/path/data.csv")
        assert "not found" in str(exc_info.value)

    def test_report_generation(self, test_data, tmp_path):
        """Test analysis report generation"""
        self.initialize_test_data(test_data)

        # Generate report
        report_path = tmp_path / "analysis_report.xlsx"
        self.pv_system.export_analysis_report(str(report_path))

        # Verify report exists
        assert report_path.exists(), "Report file should be created"
