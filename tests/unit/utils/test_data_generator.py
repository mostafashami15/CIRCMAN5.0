"""Tests for test data generator."""

import pytest
from pathlib import Path
import pandas as pd
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.config.project_paths import project_paths


@pytest.fixture
def generator():
    """Create data generator instance."""
    return ManufacturingDataGenerator()


def test_data_generation_paths(generator):
    """Test data files are saved in correct locations."""
    generator.save_generated_data()

    # Check file locations
    synthetic_dir = Path(project_paths.get_path("SYNTHETIC_DATA"))
    expected_files = [
        "test_energy_data.csv",
        "test_material_data.csv",
        "test_process_data.csv",
    ]

    for file in expected_files:
        file_path = synthetic_dir / file
        assert file_path.exists()
        assert pd.read_csv(file_path).shape[0] > 0


def test_data_quality(generator):
    """Test generated data has correct structure."""
    efficiency_data = generator.generate_production_data()
    assert not efficiency_data.empty
    assert all(
        col in efficiency_data.columns
        for col in ["timestamp", "batch_id", "input_amount", "output_amount"]
    )
