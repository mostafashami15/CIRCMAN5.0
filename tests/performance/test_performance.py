"""Performance tests for the CIRCMAN5.0 system."""

import pytest
import time
import psutil
import numpy as np
from pathlib import Path
import logging

from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.config.project_paths import project_paths
from circman5.utils.logging_config import setup_logger


# Setup logging
logger = setup_logger("performance_tests")


@pytest.fixture(scope="module")
def analyzer():
    """Create analyzer instance."""
    return SoliTekManufacturingAnalysis()


@pytest.fixture(scope="module")
def large_data_generator():
    """Create data generator for large datasets."""
    # Reduced from 365 to 30 days for optimization tests
    return ManufacturingDataGenerator(start_date="2024-01-01", days=30)


@pytest.fixture(scope="module")
def test_run_dir():
    """Create test run directory."""
    run_dir = project_paths.get_run_directory()
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    return run_dir


def test_data_generation_performance(large_data_generator):
    """Test performance of data generation."""
    logger.info("Starting data generation performance test")
    start_time = time.time()

    # Generate data
    production_data = large_data_generator.generate_production_data()
    quality_data = large_data_generator.generate_quality_data()
    energy_data = large_data_generator.generate_energy_data()
    material_data = large_data_generator.generate_material_flow_data()

    generation_time = time.time() - start_time
    logger.info(f"Data generation completed in {generation_time:.2f} seconds")

    # Verify data generation
    assert not production_data.empty, "Production data should not be empty"
    assert not quality_data.empty, "Quality data should not be empty"
    assert not energy_data.empty, "Energy data should not be empty"
    assert not material_data.empty, "Material data should not be empty"

    # Check performance
    assert (
        generation_time < 30.0
    ), f"Data generation took too long: {generation_time:.2f} seconds"

    return {
        "production_size": len(production_data),
        "quality_size": len(quality_data),
        "energy_size": len(energy_data),
        "material_size": len(material_data),
        "generation_time": generation_time,
    }


def test_analysis_memory_usage(analyzer, large_data_generator):
    """Test memory usage during analysis."""
    logger.info("Starting memory usage test")
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

    # Load data
    logger.info("Loading test data")
    analyzer.production_data = large_data_generator.generate_production_data()
    analyzer.quality_data = large_data_generator.generate_quality_data()
    analyzer.energy_data = large_data_generator.generate_energy_data()
    analyzer.material_flow = large_data_generator.generate_material_flow_data()

    # Perform analysis
    logger.info("Performing analysis")
    efficiency_metrics = analyzer.analyze_efficiency()
    quality_metrics = analyzer.analyze_quality_metrics()
    sustainability_metrics = analyzer.calculate_sustainability_metrics()

    final_memory = process.memory_info().rss / (1024 * 1024)  # MB
    memory_increase = final_memory - initial_memory
    logger.info(f"Memory increase: {memory_increase:.1f} MB")

    # Verify metrics
    assert isinstance(efficiency_metrics, dict), "Should return efficiency metrics"
    assert isinstance(quality_metrics, dict), "Should return quality metrics"
    assert isinstance(
        sustainability_metrics, dict
    ), "Should return sustainability metrics"

    # Check memory usage
    assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.1f} MB"

    return {
        "initial_memory": initial_memory,
        "final_memory": final_memory,
        "memory_increase": memory_increase,
    }


def test_optimization_performance(analyzer, large_data_generator):
    """Test performance of AI optimization."""
    logger.info("Starting optimization performance test")
    start_time = time.time()

    try:
        # Load smaller dataset for optimization
        logger.info("Loading data for optimization")
        analyzer.production_data = large_data_generator.generate_production_data()
        analyzer.quality_data = large_data_generator.generate_quality_data()

        # Train model with timeout
        logger.info("Training optimization model")
        metrics = analyzer.train_optimization_model()
        logger.info(f"Model training completed: {metrics}")

        # Test optimization
        logger.info("Testing parameter optimization")
        current_params = {
            "input_amount": 100.0,
            "energy_used": 150.0,
            "cycle_time": 50.0,
            "efficiency": 21.0,
            "defect_rate": 2.0,
            "thickness_uniformity": 95.0,
        }

        optimized_params = analyzer.optimize_process_parameters(current_params)
        optimization_time = time.time() - start_time
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")

        # Verify optimization
        assert metrics["r2"] > 0, "Model training failed"
        assert isinstance(optimized_params, dict), "Optimization failed"
        assert (
            optimization_time < 60.0
        ), f"Optimization took too long: {optimization_time:.2f} seconds"

        return {
            "training_metrics": metrics,
            "optimization_result": optimized_params,
            "optimization_time": optimization_time,
        }

    except Exception as e:
        logger.error(f"Optimization test failed: {str(e)}")
        raise


def test_visualization_performance(analyzer, large_data_generator, test_run_dir):
    """Test performance of visualization generation."""
    logger.info("Starting visualization performance test")
    start_time = time.time()

    # Load data
    logger.info("Loading data for visualization")
    analyzer.production_data = large_data_generator.generate_production_data()
    analyzer.quality_data = large_data_generator.generate_quality_data()
    analyzer.energy_data = large_data_generator.generate_energy_data()
    analyzer.material_flow = large_data_generator.generate_material_flow_data()

    # Generate visualizations
    logger.info("Generating visualizations")
    viz_dir = test_run_dir / "visualizations"
    for metric_type in ["production", "energy", "quality", "sustainability"]:
        viz_path = viz_dir / f"{metric_type}_analysis.png"
        analyzer.generate_visualization(metric_type, str(viz_path))
        assert viz_path.exists(), f"Failed to generate {metric_type} visualization"

    visualization_time = time.time() - start_time
    logger.info(f"Visualization completed in {visualization_time:.2f} seconds")
    assert (
        visualization_time < 30.0
    ), f"Visualization took too long: {visualization_time:.2f} seconds"

    return {"visualization_count": 4, "visualization_time": visualization_time}


def test_performance_logging(test_run_dir):
    """Log performance test results."""
    logger.info("Writing performance test results")
    results_file = test_run_dir / "performance_results.txt"

    with open(results_file, "w") as f:
        f.write("=== Performance Test Results ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Process ID: {psutil.Process().pid}\n")
        f.write(
            f"Memory Usage: {psutil.Process().memory_info().rss / (1024 * 1024):.1f} MB\n"
        )

    assert results_file.exists(), "Performance results not logged"
    logger.info("Performance results logged successfully")
