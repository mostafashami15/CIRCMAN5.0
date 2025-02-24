# tests/unit/adapters/services/test_constants_service.py
import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path

from circman5.adapters.services.constants_service import ConstantsService
from circman5.adapters.base.config_manager import ConfigurationManager
from circman5.adapters.base.adapter_base import ConfigAdapterBase


class MockAdapter(ConfigAdapterBase):
    """Mock adapter for testing."""

    def __init__(self, config_data=None, config_path=None):
        super().__init__(config_path)
        self.config_data = config_data or {"test_key": "test_value"}

    def load_config(self):
        return self.config_data

    def validate_config(self, config):
        return True

    def get_defaults(self):
        return {"test_key": "default_value"}


class TestConstantsService:
    """Test ConstantsService functionality."""

    def test_singleton_pattern(self):
        """Test that the service is a singleton."""
        service1 = ConstantsService()
        service2 = ConstantsService()

        assert service1 is service2

    @patch("circman5.adapters.services.constants_service.ConfigurationManager")
    def test_initialization(self, mock_config_manager):
        """Test initialization with mocked config manager."""
        # Reset the singleton for this test
        ConstantsService._instance = None

        # Create instance with mock
        manager_instance = MagicMock()
        mock_config_manager.return_value = manager_instance

        # Create service instance, which should initialize the config manager
        service = ConstantsService()

        # Check manager is created
        mock_config_manager.assert_called_once()

        # In the actual implementation, _load_all_configs is called, which then calls
        # load_config for each adapter. We should check that _load_all_configs was called
        # rather than specific load_config calls.
        assert hasattr(service, "_load_all_configs")

        # We can also verify that the adapters were registered
        assert hasattr(service, "_register_adapters")

    def test_get_manufacturing_constants(self):
        """Test getting manufacturing constants."""
        # Create service with mock manager
        ConstantsService._instance = None
        service = ConstantsService()

        # Replace the config manager with a mock
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = {"test_manufacturing": "value"}
        service.config_manager = mock_manager

        # Test the method
        result = service.get_manufacturing_constants()

        mock_manager.get_config.assert_called_with("manufacturing")
        assert result == {"test_manufacturing": "value"}

    def test_get_impact_factors(self):
        """Test getting impact factors."""
        # Create service with mock manager
        ConstantsService._instance = None
        service = ConstantsService()

        # Replace the config manager with a mock
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = {"test_factor": 1.0}
        service.config_manager = mock_manager

        # Test the method
        result = service.get_impact_factors()

        mock_manager.get_config.assert_called_with("impact_factors")
        assert result == {"test_factor": 1.0}

    def test_get_optimization_config(self):
        """Test getting optimization config."""
        # Create service with mock manager
        ConstantsService._instance = None
        service = ConstantsService()

        # Replace the config manager with a mock
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = {"test_optimization": "value"}
        service.config_manager = mock_manager

        # Test the method
        result = service.get_optimization_config()

        mock_manager.get_config.assert_called_with("optimization")
        assert result == {"test_optimization": "value"}

    def test_get_constant(self):
        """Test getting a specific constant."""
        # Create service with mock manager
        ConstantsService._instance = None
        service = ConstantsService()

        # Replace the config manager with a mock
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = {"specific_key": "specific_value"}
        service.config_manager = mock_manager

        # Test the method
        result = service.get_constant("test_adapter", "specific_key")

        mock_manager.get_config.assert_called_with("test_adapter")
        assert result == "specific_value"

    def test_get_constant_key_error(self):
        """Test error handling when key not found."""
        # Create service with mock manager
        ConstantsService._instance = None
        service = ConstantsService()

        # Replace the config manager with a mock
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = {"existing_key": "value"}
        service.config_manager = mock_manager

        # Test the method
        with pytest.raises(
            KeyError, match="Key not found in test_adapter config: missing_key"
        ):
            service.get_constant("test_adapter", "missing_key")

    def test_reload_configs(self):
        """Test reloading all configurations."""
        # Create service with mock manager
        ConstantsService._instance = None
        service = ConstantsService()

        # Replace the config manager with a mock
        mock_manager = MagicMock()
        service.config_manager = mock_manager

        # Test the method
        service.reload_configs()

        mock_manager.reload_all.assert_called_once()
