"""
Performance tests for the manufacturing analysis system.
Tests system performance with various data sizes and processing scenarios.
"""

import pytest
import time
import psutil
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Any, Optional
from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator  # Updated import

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def generate_large_dataset(num_days: int = 30) -> dict:
    """Generate a large dataset for performance testing."""
    try:
        generator = ManufacturingDataGenerator(start_date="2024-01-01", days=num_days)

        return {
            "production": generator.generate_production_data(),
            "energy": generator.generate_energy_data(),
            "quality": generator.generate_quality_data(),
            "material": generator.generate_material_flow_data(),
        }
    except Exception as e:
        pytest.fail(f"Failed to generate dataset: {str(e)}")
        return {}


@pytest.fixture(scope="module")
def large_dataset():
    """Create large dataset fixture."""
    return generate_large_dataset()


def measure_execution_time(func):
    """Decorator to measure function execution time."""

    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            return (result, execution_time)
        except Exception as e:
            pytest.fail(f"Function execution failed: {str(e)}")
            return (None, 0.0)

    return wrapper


@pytest.mark.performance
def test_data_loading_performance(large_dataset, tmp_path):
    """Test performance of data loading operations."""
    analyzer = SoliTekManufacturingAnalysis()

    @measure_execution_time
    def load_data() -> dict:  # Changed return type to dict
        analyzer.production_data = large_dataset["production"]
        analyzer.energy_data = large_dataset["energy"]
        analyzer.quality_data = large_dataset["quality"]
        analyzer.material_flow = large_dataset["material"]
        return {"success": True}  # Return a dict instead of bool

    result = load_data()  # Don't try to unpack here
    if not result[0].get("success", False):  # Check the result properly
        pytest.fail("Data loading failed")
    load_time = result[1]  # Get the time separately
    print(f"\nData loading time: {load_time:.2f} seconds")
    assert isinstance(load_time, float)
    assert (
        load_time < 5.0
    ), f"Data loading took {load_time:.2f} seconds, exceeding 5 second threshold"


@pytest.mark.performance
def test_analysis_performance(large_dataset):
    """Test performance of analysis operations."""
    analyzer = SoliTekManufacturingAnalysis()
    analyzer.production_data = large_dataset["production"]
    analyzer.energy_data = large_dataset["energy"]
    analyzer.quality_data = large_dataset["quality"]
    analyzer.material_flow = large_dataset["material"]

    # Test efficiency analysis performance
    @measure_execution_time
    def run_efficiency_analysis():
        return analyzer.analyze_efficiency()

    (result, efficiency_time) = run_efficiency_analysis()
    if result is None:
        pytest.fail("Efficiency analysis failed")
    print(f"\nEfficiency analysis time: {efficiency_time:.2f} seconds")
    assert isinstance(efficiency_time, float)
    assert (
        efficiency_time < 2.0
    ), f"Efficiency analysis took {efficiency_time:.2f} seconds"

    # Test quality analysis performance
    @measure_execution_time
    def run_quality_analysis():
        return analyzer.analyze_quality_metrics()

    (result, quality_time) = run_quality_analysis()
    if result is None:
        pytest.fail("Quality analysis failed")
    print(f"Quality analysis time: {quality_time:.2f} seconds")
    assert isinstance(quality_time, float)
    assert quality_time < 2.0, f"Quality analysis took {quality_time:.2f} seconds"

    # Test sustainability analysis performance
    @measure_execution_time
    def run_sustainability_analysis():
        return analyzer.calculate_sustainability_metrics()

    (result, sustainability_time) = run_sustainability_analysis()
    if result is None:
        pytest.fail("Sustainability analysis failed")
    print(f"Sustainability analysis time: {sustainability_time:.2f} seconds")
    assert isinstance(sustainability_time, float)
    assert (
        sustainability_time < 2.0
    ), f"Sustainability analysis took {sustainability_time:.2f} seconds"


@pytest.mark.performance
def test_visualization_performance(large_dataset, tmp_path):
    """Test performance of visualization generation."""
    analyzer = SoliTekManufacturingAnalysis()
    analyzer.production_data = large_dataset["production"]
    analyzer.energy_data = large_dataset["energy"]
    analyzer.quality_data = large_dataset["quality"]
    analyzer.material_flow = large_dataset["material"]

    for metric_type in ["production", "quality", "sustainability"]:

        @measure_execution_time
        def generate_vis() -> str:
            vis_path = tmp_path / f"test_{metric_type}.png"
            analyzer.generate_visualization(metric_type, str(vis_path))
            return str(vis_path)

        (path, vis_time) = generate_vis()
        if path is None:
            pytest.fail(f"{metric_type} visualization failed")
        print(f"\n{metric_type} visualization time: {vis_time:.2f} seconds")
        assert isinstance(vis_time, float)
        assert (
            vis_time < 3.0
        ), f"{metric_type} visualization took {vis_time:.2f} seconds"
        assert Path(path).exists()


@pytest.mark.performance
def test_memory_usage(large_dataset):
    """Test memory usage with large datasets."""

    def get_memory_usage() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f}MB")

    analyzer = SoliTekManufacturingAnalysis()
    analyzer.production_data = large_dataset["production"]
    analyzer.energy_data = large_dataset["energy"]
    analyzer.quality_data = large_dataset["quality"]
    analyzer.material_flow = large_dataset["material"]

    final_memory = get_memory_usage()
    print(f"Final memory usage: {final_memory:.2f}MB")
    memory_increase = final_memory - initial_memory
    print(f"Memory increase: {memory_increase:.2f}MB")

    # Assert memory increase is within acceptable range (less than 500MB)
    assert memory_increase < 500, f"Memory usage increased by {memory_increase:.2f}MB"


if __name__ == "__main__":
    pytest.main(["-v", "-m", "performance"])
