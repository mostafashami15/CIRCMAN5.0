"""Performance tests for the CIRCMAN5.0 system."""

import pytest
import time
import psutil
import numpy as np
from pathlib import Path

from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.config.project_paths import project_paths


@pytest.fixture(scope="module")
def large_test_data():
    """Generate large test dataset for performance testing."""
    return ManufacturingDataGenerator(start_date="2024-01-01", days=365)


@pytest.fixture(scope="module")
def test_run_dir():
    """Create test run directory."""
    return project_paths.get_run_directory()


def measure_execution_time(func):
    """Decorator to measure function execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        return result, duration

    return wrapper


def measure_memory_usage(func):
    """Decorator to measure function memory usage."""

    def wrapper(*args, **kwargs):
        process = psutil.Process()
        memory_before = process.memory_info().rss
        result = func(*args, **kwargs)
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before
        return result, memory_used

    return wrapper


@measure_execution_time
def test_data_generation_performance(large_test_data):
    """Test performance of data generation."""
    production_data = large_test_data.generate_production_data()
    quality_data = large_test_data.generate_quality_data()
    energy_data = large_test_data.generate_energy_data()
    material_data = large_test_data.generate_material_flow_data()

    assert not production_data.empty
    assert not quality_data.empty
    assert not energy_data.empty
    assert not material_data.empty

    return {
        "production_size": len(production_data),
        "quality_size": len(quality_data),
        "energy_size": len(energy_data),
        "material_size": len(material_data),
    }


@measure_memory_usage
def test_analysis_memory_usage(large_test_data):
    """Test memory usage during analysis."""
    analyzer = SoliTekManufacturingAnalysis()

    # Load large datasets
    analyzer.production_data = large_test_data.generate_production_data()
    analyzer.quality_data = large_test_data.generate_quality_data()
    analyzer.energy_data = large_test_data.generate_energy_data()
    analyzer.material_flow = large_test_data.generate_material_flow_data()

    # Perform analysis
    efficiency_metrics = analyzer.analyze_efficiency()
    quality_metrics = analyzer.analyze_quality_metrics()
    sustainability_metrics = analyzer.calculate_sustainability_metrics()

    return {
        "efficiency": efficiency_metrics,
        "quality": quality_metrics,
        "sustainability": sustainability_metrics,
    }


@measure_execution_time
def test_optimization_performance(large_test_data):
    """Test performance of AI optimization."""
    analyzer = SoliTekManufacturingAnalysis()

    # Load data
    analyzer.production_data = large_test_data.generate_production_data()
    analyzer.quality_data = large_test_data.generate_quality_data()

    # Train model
    metrics = analyzer.train_optimization_model()

    # Test optimization
    current_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    optimized_params = analyzer.optimize_process_parameters(current_params)

    return {"training_metrics": metrics, "optimization_result": optimized_params}


@measure_execution_time
def test_visualization_performance(large_test_data, test_run_dir):
    """Test performance of visualization generation."""
    analyzer = SoliTekManufacturingAnalysis()

    # Load data
    analyzer.production_data = large_test_data.generate_production_data()
    analyzer.quality_data = large_test_data.generate_quality_data()
    analyzer.energy_data = large_test_data.generate_energy_data()
    analyzer.material_flow = large_test_data.generate_material_flow_data()

    # Generate visualizations
    for metric_type in ["production", "energy", "quality", "sustainability"]:
        viz_path = test_run_dir / "visualizations" / f"{metric_type}_analysis.png"
        analyzer.generate_visualization(metric_type, str(viz_path))
        assert viz_path.exists()

    return {"visualization_count": 4}


def test_performance_logging(test_run_dir):
    """Log performance test results."""
    results = []

    # Run data generation test
    data_result, data_time = test_data_generation_performance(
        ManufacturingDataGenerator()
    )
    results.append(("Data Generation", data_time))

    # Run analysis test
    analysis_result, memory_usage = test_analysis_memory_usage(
        ManufacturingDataGenerator()
    )
    # Ensure memory_usage is a number before division
    memory_usage_mb = float(memory_usage) / (1024 * 1024)  # Convert to MB
    results.append(("Analysis Memory", memory_usage_mb))

    # Run optimization test
    opt_result, opt_time = test_optimization_performance(ManufacturingDataGenerator())
    results.append(("Optimization", opt_time))

    # Run visualization test
    viz_result, viz_time = test_visualization_performance(
        ManufacturingDataGenerator(), test_run_dir
    )
    results.append(("Visualization", viz_time))

    # Save results
    performance_log = test_run_dir / "performance_results.txt"
    with open(performance_log, "w") as f:
        f.write("=== Performance Test Results ===\n\n")
        for name, value in results:
            if name == "Analysis Memory":
                f.write(f"{name}: {value:.2f} MB\n")
            else:
                f.write(f"{name}: {value:.2f} seconds\n")

    assert performance_log.exists()
