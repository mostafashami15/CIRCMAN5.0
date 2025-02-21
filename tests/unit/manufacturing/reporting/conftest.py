# tests/unit/manufacturing/reporting/conftest.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from circman5.utils.results_manager import results_manager


@pytest.fixture(scope="session")
def test_data_dir():
    """Get input data directory from ResultsManager."""
    return results_manager.get_path("input_data")


@pytest.fixture(scope="module")
def viz_dir():
    """Get visualization directory from ResultsManager."""
    return results_manager.get_path("visualizations")


@pytest.fixture(scope="module")
def reports_dir():
    """Get reports directory from ResultsManager."""
    return results_manager.get_path("reports")


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
        "sustainability": {
            "material_efficiency": 92.0,
            "energy_efficiency": 88.5,
            "waste_reduction": 94.0,
        },
    }


@pytest.fixture
def sample_monitor_data():
    """Generate sample monitoring data."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")

    efficiency_data = pd.DataFrame(
        {
            "timestamp": dates,
            "production_rate": np.random.uniform(80, 100, 10),
            "energy_efficiency": np.random.uniform(85, 95, 10),
        }
    )

    quality_data = pd.DataFrame(
        {
            "timestamp": dates,
            "quality_score": np.random.uniform(90, 98, 10),
            "defect_rate": np.random.uniform(1, 5, 10),
        }
    )

    resource_data = pd.DataFrame(
        {
            "timestamp": dates,
            "material_consumption": np.random.uniform(900, 1100, 10),
            "resource_efficiency": np.random.uniform(85, 95, 10),
        }
    )

    return {
        "efficiency": efficiency_data,
        "quality": quality_data,
        "resources": resource_data,
    }


@pytest.fixture
def cleanup_test_files():
    """Cleanup temporary test files after tests."""
    yield
