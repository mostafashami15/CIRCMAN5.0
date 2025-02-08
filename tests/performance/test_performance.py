"""Performance tests for the manufacturing analysis system."""

import pytest
import time
import psutil
import os
from pathlib import Path

from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.config.project_paths import project_paths


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return SoliTekManufacturingAnalysis()


@pytest.fixture
def test_data():
    """Generate test data."""
    generator = ManufacturingDataGenerator(days=30)
    return {
        "production": generator.generate_production_data(),
        "quality": generator.generate_quality_data(),
        "energy": generator.generate_energy_data(),
        "material": generator.generate_material_flow_data(),
    }


def test_data_loading_performance(analyzer, test_data):
    """Test data loading performance."""
    start_time = time.time()

    # Get run directory for this test
    run_dir = project_paths.get_run_directory()
    data_dir = run_dir / "input_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save and load test data
    data_path = data_dir / "test_production_data.csv"
    test_data["production"].to_csv(data_path, index=False)

    analyzer.load_production_data(data_path)

    load_time = time.time() - start_time
    assert load_time < 5.0, f"Data loading took {load_time:.2f} seconds"
    assert data_path.exists(), "Test data file not saved"


def test_analysis_performance(analyzer, test_data):
    """Test analysis performance."""
    # Load test data
    analyzer.production_data = test_data["production"]
    analyzer.quality_data = test_data["quality"]
    analyzer.energy_data = test_data["energy"]
    analyzer.material_flow = test_data["material"]

    # Get run directory for results
    run_dir = project_paths.get_run_directory()
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Perform analysis
    analyzer.analyze_efficiency()
    analyzer.analyze_quality_metrics()
    analyzer.calculate_sustainability_metrics()

    # Generate report
    report_path = results_dir / "performance_test_report.xlsx"
    analyzer.export_analysis_report(str(report_path))

    analysis_time = time.time() - start_time
    assert analysis_time < 10.0, f"Analysis took {analysis_time:.2f} seconds"
    assert report_path.exists(), "Analysis report not generated"


def test_visualization_performance(analyzer, test_data):
    """Test visualization generation performance."""
    # Load ALL required data
    analyzer.production_data = test_data["production"]
    analyzer.quality_data = test_data["quality"]
    analyzer.energy_data = test_data["energy"]
    analyzer.material_flow = test_data["material"]  # Added this line

    # Get run directory for visualizations
    run_dir = project_paths.get_run_directory()
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Generate visualizations
    for metric_type in ["production", "quality", "sustainability"]:
        viz_path = viz_dir / f"{metric_type}_analysis.png"
        analyzer.generate_visualization(metric_type, str(viz_path))
        assert viz_path.exists(), f"{metric_type} visualization not generated"

    viz_time = time.time() - start_time
    assert viz_time < 15.0, f"Visualization generation took {viz_time:.2f} seconds"


def test_memory_usage(analyzer, test_data):
    """Test memory usage during analysis."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Load data and perform analysis
    analyzer.production_data = test_data["production"]
    analyzer.quality_data = test_data["quality"]

    # Get run directory
    run_dir = project_paths.get_run_directory()
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Perform analysis operations
    analyzer.analyze_efficiency()
    analyzer.analyze_quality_metrics()

    # Generate report
    report_path = results_dir / "memory_test_report.xlsx"
    analyzer.export_analysis_report(str(report_path))

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f} MB"
    assert report_path.exists(), "Memory test report not generated"
