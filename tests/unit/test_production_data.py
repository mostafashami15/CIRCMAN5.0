import pytest
import pandas as pd
import os
from pathlib import Path
from circman5.manufacturing.data_loader import ManufacturingDataLoader
from circman5.utils.errors import DataError, ValidationError


@pytest.fixture
def test_data_path(tmp_path):
    """Create a temporary test CSV file with all required columns"""
    data = {
        "batch_id": ["BATCH_001", "BATCH_002"],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
        "stage": ["silicon_purification", "silicon_purification"],
        "input_amount": [100.0, 150.0],
        "output_amount": [90.0, 140.0],
        "energy_used": [50.0, 75.0],
        # Add new required columns
        "product_type": ["solar_panel", "solar_panel"],
        "production_line": ["LINE_A", "LINE_A"],
        "output_quantity": [85.0, 135.0],
        "cycle_time": [45.0, 48.0],
        "yield_rate": [0.90, 0.93],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_production_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_valid_production_data_loading(test_data_path):
    """Test loading valid production data"""
    analyzer = ManufacturingDataLoader()
    data = analyzer.load_production_data(test_data_path)
    assert not data.empty
    assert len(data) == 2
    assert all(
        col in data.columns
        for col in [
            "batch_id",
            "product_type",
            "production_line",
            "output_quantity",
            "cycle_time",
            "yield_rate",
        ]
    )


# Rest of the test file remains the same...
