# tests/unit/manufacturing/analyzers/conftest.py

"""Shared test fixtures for analyzer tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from circman5.utils.results_manager import results_manager


@pytest.fixture
def test_output_dir():
    """Create and return test output directory."""
    run_dir = results_manager.get_run_dir()
    print(f"Test output directory: {run_dir}")  # Debug print
    return run_dir


@pytest.fixture
def reports_dir():
    """Get reports directory for current test run."""
    return results_manager.get_path("reports")


@pytest.fixture
def visualizations_dir():
    """Get visualizations directory for current test run."""
    return results_manager.get_path("visualizations")


@pytest.fixture
def sample_energy_data():
    """Generate sample energy consumption data."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-10", freq="H"),
            "energy_source": np.random.choice(["grid", "solar", "wind"], 240),
            "energy_consumption": np.random.uniform(40, 60, 240),
            "efficiency_rate": np.random.uniform(0.85, 0.95, 240),
        }
    )


@pytest.fixture
def sample_material_data():
    """Generate sample material flow data."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-10", freq="H"),
            "material_type": np.random.choice(["silicon", "glass", "metal"], 240),
            "quantity_used": np.random.uniform(80, 120, 240),
            "waste_generated": np.random.uniform(2, 8, 240),
            "recycled_amount": np.random.uniform(1, 6, 240),
        }
    )
