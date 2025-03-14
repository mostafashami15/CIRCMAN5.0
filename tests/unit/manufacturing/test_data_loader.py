import pytest
import pandas as pd
from pathlib import Path

from circman5.manufacturing.data_loader import ManufacturingDataLoader
from circman5.utils.errors import DataError, ValidationError


@pytest.fixture
def data_loader():
    """Create a data loader instance for testing."""
    return ManufacturingDataLoader()


class TestManufacturingDataLoader:
    """Test data loading and processing functionality."""

    def test_initialization(self, data_loader):
        """Test proper initialization."""
        assert hasattr(data_loader, "production_schema")
        assert hasattr(data_loader, "energy_schema")
        assert hasattr(data_loader, "quality_schema")
        assert hasattr(data_loader, "material_schema")

    def test_load_real_time_data(self, data_loader, mocker):
        """Test real-time data streaming configuration."""
        # Test with default parameters
        stream_config = data_loader.load_real_time_data()
        assert isinstance(stream_config, dict)
        assert "buffer_size" in stream_config
        assert stream_config["buffer_size"] == 100
        assert "active" in stream_config
        assert stream_config["active"] is True

        # Test with custom parameters
        custom_source = {"type": "test", "url": "test://localhost"}
        custom_config = data_loader.load_real_time_data(
            data_source=custom_source, buffer_size=200
        )
        assert custom_config["buffer_size"] == 200
        assert custom_config["data_source"] == custom_source

    def test_validate_streaming_data(self, data_loader):
        """Test validation of streaming data points."""
        # Valid data point
        valid_data = {
            "timestamp": "2025-01-01T10:00:00",
            "batch_id": "BATCH001",
            "input_amount": 100.0,
            "energy_used": 50.0,
            "cycle_time": 30.0,
        }
        assert data_loader.validate_streaming_data(valid_data) is True

        # Invalid - missing fields
        invalid_data1 = {
            "timestamp": "2025-01-01T10:00:00",
            "batch_id": "BATCH001",
            # Missing required fields
        }
        assert data_loader.validate_streaming_data(invalid_data1) is False

        # Invalid - negative values
        invalid_data2 = {
            "timestamp": "2025-01-01T10:00:00",
            "batch_id": "BATCH001",
            "input_amount": -10.0,  # Negative
            "energy_used": 50.0,
            "cycle_time": 30.0,
        }
        assert data_loader.validate_streaming_data(invalid_data2) is False

        # Invalid - wrong types
        invalid_data3 = {
            "timestamp": "2025-01-01T10:00:00",
            "batch_id": "BATCH001",
            "input_amount": "not a number",  # Wrong type
            "energy_used": 50.0,
            "cycle_time": 30.0,
        }
        assert data_loader.validate_streaming_data(invalid_data3) is False

    def test_integrate_external_sources(self, data_loader, mocker):
        """Test integration of external data sources."""
        # Valid source config
        valid_source = {
            "name": "test_source",
            "type": "database",
            "connection_params": {
                "host": "localhost",
                "port": 5432,
                "username": "test",
                "password": "test",
                "database": "test_db",
            },
        }

        # Test successful integration
        assert data_loader.integrate_external_sources(valid_source) is True

        # Test with invalid config (should not raise error but return False)
        invalid_source = {
            "type": "unknown"
            # Missing required fields
        }
        assert data_loader.integrate_external_sources(invalid_source) is False
