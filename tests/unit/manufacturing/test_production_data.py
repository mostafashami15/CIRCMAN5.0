# tests/unit/manufacturing/test_production_data.py

import pytest
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime

from circman5.manufacturing.data_loader import ManufacturingDataLoader
from circman5.utils.errors import DataError, ValidationError
from circman5.utils.results_manager import results_manager


@pytest.fixture(scope="module")
def input_data_dir():
    """Get input data directory from ResultsManager."""
    return results_manager.get_path("input_data")


@pytest.fixture
def data_loader():
    """Create data loader instance."""
    return ManufacturingDataLoader()


@pytest.fixture
def valid_production_data(input_data_dir):
    """Create valid test production data."""
    data = {
        "batch_id": ["BATCH_001", "BATCH_002"],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
        "stage": ["silicon_purification", "silicon_purification"],
        "input_amount": [100.0, 150.0],
        "output_amount": [90.0, 140.0],
        "energy_used": [50.0, 75.0],
        "product_type": ["solar_panel", "solar_panel"],
        "production_line": ["LINE_A", "LINE_A"],
        "output_quantity": [85.0, 135.0],
        "cycle_time": [45.0, 48.0],
        "yield_rate": [0.90, 0.93],
    }
    df = pd.DataFrame(data)
    filename = "test_production_data.csv"
    save_path = input_data_dir / filename
    df.to_csv(save_path, index=False)
    return save_path


def test_valid_production_data_loading(data_loader, valid_production_data):
    """Test loading valid production data."""
    data = data_loader.load_production_data(valid_production_data)
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


def test_missing_columns(data_loader, input_data_dir):
    """Test handling of missing required columns."""
    # Create data missing required columns
    invalid_data = pd.DataFrame(
        {"batch_id": ["BATCH_001"], "timestamp": ["2024-01-01 10:00:00"]}
    )

    filename = "invalid_production_data.csv"
    save_path = input_data_dir / filename
    invalid_data.to_csv(save_path, index=False)

    with pytest.raises(ValidationError, match="Missing required columns"):
        data_loader.load_production_data(save_path)


def test_invalid_data_types(data_loader, input_data_dir):
    """Test handling of invalid data types."""
    invalid_data = pd.DataFrame(
        {
            "batch_id": ["BATCH_001"],
            "timestamp": ["2024-01-01 10:00:00"],
            "input_amount": ["invalid"],  # Should be float
            "output_amount": [90.0],
            "energy_used": [50.0],
            "product_type": ["solar_panel"],
            "production_line": ["LINE_A"],
            "output_quantity": [85.0],
            "cycle_time": [45.0],
            "yield_rate": [0.90],
        }
    )

    filename = "invalid_types_data.csv"
    save_path = input_data_dir / filename
    invalid_data.to_csv(save_path, index=False)

    with pytest.raises(ValidationError, match="Invalid data type"):
        data_loader.load_production_data(save_path)


def test_validate_production_data(data_loader, valid_production_data):
    """Test production data validation."""
    data = pd.read_csv(valid_production_data)
    assert data_loader.validate_production_data(data)


def test_negative_values(data_loader, input_data_dir):
    """Test handling of negative values."""
    invalid_data = pd.DataFrame(
        {
            "batch_id": ["BATCH_001"],
            "timestamp": ["2024-01-01 10:00:00"],
            "input_amount": [-100.0],  # Invalid negative value
            "output_amount": [90.0],
            "energy_used": [50.0],
            "product_type": ["solar_panel"],
            "production_line": ["LINE_A"],
            "output_quantity": [85.0],
            "cycle_time": [45.0],
            "yield_rate": [0.90],
        }
    )

    # Validate directly without saving to file
    with pytest.raises(ValidationError, match="Input amounts cannot be negative"):
        data_loader.validate_production_data(invalid_data)


def test_excessive_output(data_loader, input_data_dir):
    """Test handling of output exceeding input."""
    invalid_data = pd.DataFrame(
        {
            "batch_id": ["BATCH_001"],
            "timestamp": ["2024-01-01 10:00:00"],
            "input_amount": [100.0],
            "output_amount": [200.0],  # Exceeds input by more than 10%
            "energy_used": [50.0],
            "product_type": ["solar_panel"],
            "production_line": ["LINE_A"],
            "output_quantity": [85.0],
            "cycle_time": [45.0],
            "yield_rate": [0.90],
        }
    )

    # Validate directly without saving to file
    with pytest.raises(
        ValidationError, match="Output amount cannot significantly exceed input amount"
    ):
        data_loader.validate_production_data(invalid_data)


def test_empty_file_handling(data_loader, input_data_dir):
    """Test handling of empty files."""
    empty_df = pd.DataFrame(
        columns=list(data_loader.production_schema.keys())
    )  # Add columns
    filename = "empty_data.csv"
    save_path = input_data_dir / filename
    empty_df.to_csv(save_path, index=False)

    with pytest.raises(DataError, match="data file is empty"):
        data_loader.load_production_data(save_path)


def test_missing_file_handling(data_loader):
    """Test handling of missing files."""
    with pytest.raises(DataError, match="data file not found"):
        data_loader.load_production_data("nonexistent_file.csv")


def test_file_saving(data_loader, valid_production_data, input_data_dir):
    """Test that loaded data is properly saved."""
    data_loader.load_production_data(valid_production_data)
    saved_file = input_data_dir / "production_data.csv"
    assert saved_file.exists()
