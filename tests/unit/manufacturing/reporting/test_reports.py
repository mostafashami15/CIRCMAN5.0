# tests/unit/manufacturing/reporting/test_reports.py

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from circman5.manufacturing.reporting.reports import ReportGenerator
from circman5.utils.results_manager import results_manager


@pytest.fixture(scope="module")
def reports_dir():
    """Get reports directory from ResultsManager."""
    return results_manager.get_path("reports")


@pytest.fixture
def report_generator():
    return ReportGenerator()


@pytest.fixture
def sample_metrics():
    """Generate sample metrics data."""
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


def test_generate_comprehensive_report(report_generator, sample_metrics, reports_dir):
    """Test comprehensive report generation."""
    output_file = reports_dir / "comprehensive_test_report.xlsx"
    report_generator.generate_comprehensive_report(sample_metrics, output_file)

    assert output_file.exists()
    df_dict = pd.read_excel(output_file, sheet_name=None)
    assert "efficiency" in df_dict
    assert "quality" in df_dict


def test_export_analysis_report(report_generator, sample_metrics, reports_dir):
    """Test analysis report export."""
    output_file = reports_dir / "analysis_test_report.xlsx"
    report_generator.export_analysis_report(sample_metrics, output_file)

    assert output_file.exists()
    df_dict = pd.read_excel(output_file, sheet_name=None)
    assert len(df_dict) == len(sample_metrics)


def test_generate_lca_report(report_generator, reports_dir):
    """Test LCA report generation."""
    impact_data = {
        "manufacturing": {"total_impact": 1000.0, "energy_usage": 500.0},
        "use_phase": {"carbon_savings": -2000.0},
    }
    batch_id = "TEST_001"

    output_file = reports_dir / f"lca_report_{batch_id}.xlsx"
    report_generator.generate_lca_report(impact_data, batch_id)

    assert output_file.exists()
    df_dict = pd.read_excel(output_file, sheet_name=None)
    assert "manufacturing" in df_dict
    assert "use_phase" in df_dict


def test_save_performance_metrics(report_generator, reports_dir):
    """Test saving performance metrics."""
    metrics = {"efficiency": 95.5, "quality_score": 98.0, "resource_efficiency": 92.5}

    output_file = reports_dir / "performance_metrics.xlsx"
    report_generator.save_performance_metrics(metrics)

    assert output_file.exists()
    df = pd.read_excel(output_file)
    assert all(metric in df.columns for metric in metrics.keys())


def test_generate_performance_report(report_generator, reports_dir):
    """Test performance report generation."""
    metrics = {
        "efficiency": 95.5,
        "quality_score": 98.0,
        "resource_efficiency": 92.5,
        "energy_efficiency": 94.0,
    }

    output_file = reports_dir / "performance_report.xlsx"
    report_generator.generate_performance_report(metrics, output_file)

    assert output_file.exists()
    df_dict = pd.read_excel(output_file, sheet_name=None)
    assert "Overall Metrics" in df_dict
    assert "Detailed Analysis" in df_dict
