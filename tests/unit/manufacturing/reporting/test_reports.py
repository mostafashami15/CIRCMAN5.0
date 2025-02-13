# tests/unit/manufacturing/reporting/test_reports.py

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from circman5.manufacturing.reporting.reports import ReportGenerator


@pytest.fixture
def report_generator():
    return ReportGenerator()


@pytest.fixture
def sample_metrics():
    return {
        "efficiency": {
            "yield_rate": 95.5,
            "output_amount": 1000,
            "cycle_time_efficiency": 0.85,
        },
        "quality": {
            "defect_rate": 2.5,
            "efficiency_score": 98.0,
            "uniformity_score": 96.5,
        },
    }


def test_generate_comprehensive_report(report_generator, sample_metrics, tmp_path):
    """Test comprehensive report generation."""
    output_file = tmp_path / "test_report.xlsx"
    report_generator.generate_comprehensive_report(sample_metrics, output_file)

    assert output_file.exists()
    # Verify Excel file contains expected sheets
    df_dict = pd.read_excel(output_file, sheet_name=None)
    assert "efficiency" in df_dict
    assert "quality" in df_dict


def test_export_analysis_report(report_generator, sample_metrics, tmp_path):
    """Test analysis report export."""
    output_file = tmp_path / "analysis_report.xlsx"
    report_generator.export_analysis_report(sample_metrics, output_file)

    assert output_file.exists()
    df_dict = pd.read_excel(output_file, sheet_name=None)
    assert len(df_dict) == len(sample_metrics)


def test_generate_lca_report(report_generator, tmp_path):
    """Test LCA report generation."""
    impact_data = {
        "manufacturing": {"total_impact": 1000.0, "energy_usage": 500.0},
        "use_phase": {"carbon_savings": -2000.0},
    }

    report_generator.generate_lca_report(impact_data, "TEST_001")

    # Verify report was created in reports directory
    report_path = report_generator.reports_dir / "lca_report_TEST_001.xlsx"
    assert report_path.exists()


def test_save_performance_metrics(report_generator):
    """Test saving performance metrics."""
    metrics = {"efficiency": 95.5, "quality_score": 98.0, "resource_efficiency": 92.5}

    report_generator.save_performance_metrics(metrics)

    # Verify metrics file was created
    metrics_path = report_generator.reports_dir / "performance_metrics.xlsx"
    assert metrics_path.exists()
