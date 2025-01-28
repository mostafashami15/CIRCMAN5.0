import pytest
import pandas as pd
import os
from pathlib import Path
from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis, ValidationError


@pytest.fixture
def test_data_path(tmp_path):
    """Create a temporary test CSV file"""
    data = {
        "batch_id": ["BATCH_001", "BATCH_002"],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
        "stage": ["silicon_purification", "silicon_purification"],
        "input_amount": [100.0, 150.0],
        "output_amount": [90.0, 140.0],
        "energy_used": [50.0, 75.0],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_production_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_valid_production_data_loading(test_data_path):
    """Test loading valid production data"""
    analyzer = SoliTekManufacturingAnalysis()
    analyzer.load_production_data(test_data_path)
    assert not analyzer.production_data.empty
    assert len(analyzer.production_data) == 2
    assert "batch_id" in analyzer.production_data.columns


def test_invalid_file_path():
    """Test loading from non-existent file"""
    analyzer = SoliTekManufacturingAnalysis()
    with pytest.raises(FileNotFoundError):
        analyzer.load_production_data("nonexistent_file.csv")


def test_invalid_data_validation():
    """Test loading data that fails validation"""
    # Create test data with invalid values
    data = {
        "batch_id": ["BATCH_001"],
        "timestamp": ["2024-01-01 10:00:00"],
        "stage": ["silicon_purification"],
        "input_amount": [100.0],
        "output_amount": [150.0],  # Greater than input_amount, should fail validation
        "energy_used": [50.0],
    }
    df = pd.DataFrame(data)

    test_file = "test_invalid_data.csv"
    df.to_csv(test_file, index=False)

    analyzer = SoliTekManufacturingAnalysis()
    with pytest.raises(ValidationError):
        analyzer.load_production_data(test_file)

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
