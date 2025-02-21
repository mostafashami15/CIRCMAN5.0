# tests/unit/utils/test_data_generator.py

import pytest
import pandas as pd
from pathlib import Path
import shutil
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.results_manager import ResultsManager


@pytest.fixture
def results_manager():
    """Create ResultsManager instance."""
    return ResultsManager()


@pytest.fixture
def generator():
    """Create data generator instance."""
    return ManufacturingDataGenerator()


def test_data_generation_paths(generator, results_manager):
    """Test data files are saved in correct locations."""
    # Generate and save data
    generator.save_generated_data()

    # Check file locations
    synthetic_dir = results_manager.get_path("SYNTHETIC_DATA")
    expected_files = [
        "test_energy_data.csv",
        "test_material_data.csv",
        "test_process_data.csv",
        "test_production_data.csv",
    ]

    for file in expected_files:
        file_path = synthetic_dir / file
        assert file_path.exists(), f"File {file} not found"
        df = pd.read_csv(file_path)
        assert not df.empty, f"File {file} is empty"


def test_data_quality(generator):
    """Test generated data has correct structure."""
    efficiency_data = generator.generate_production_data()
    assert not efficiency_data.empty
    assert all(
        col in efficiency_data.columns
        for col in ["timestamp", "batch_id", "input_amount", "output_amount"]
    )
