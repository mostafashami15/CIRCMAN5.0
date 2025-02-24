# tests/unit/adapters/base/test_config_manager.py
import pytest
from pathlib import Path

from circman5.adapters.base.config_manager import ConfigurationManager
from circman5.adapters.base.adapter_base import ConfigAdapterBase


class MockAdapter(ConfigAdapterBase):
    """Mock adapter for testing."""

    def __init__(self, config_path=None, mock_config=None):
        super().__init__(config_path)
        self.mock_config = mock_config or {"test_key": "test_value"}

    def load_config(self):
        return self.mock_config

    def validate_config(self, config):
        return "test_key" in config

    def get_defaults(self):
        return {"test_key": "default_value"}


class TestConfigurationManager:
    """Test ConfigurationManager functionality."""

    def test_initialization(self):
        """Test basic initialization."""
        manager = ConfigurationManager()
        assert hasattr(manager, "adapters")
        assert manager.adapters == {}

    def test_register_adapter(self):
        """Test adapter registration."""
        manager = ConfigurationManager()
        adapter = MockAdapter()

        manager.register_adapter("test", adapter)
        assert "test" in manager.adapters
        assert manager.adapters["test"] == adapter

    def test_load_config(self):
        """Test loading configuration."""
        manager = ConfigurationManager()
        adapter = MockAdapter()
        manager.register_adapter("test", adapter)

        config = manager.load_config("test")
        assert config == {"test_key": "test_value"}

    def test_get_config(self):
        """Test getting configuration."""
        manager = ConfigurationManager()
        adapter = MockAdapter()
        manager.register_adapter("test", adapter)

        # First call loads the config
        config = manager.get_config("test")
        assert config == {"test_key": "test_value"}

        # Second call returns cached config
        config = manager.get_config("test")
        assert config == {"test_key": "test_value"}

    def test_unknown_adapter(self):
        """Test error handling for unknown adapter."""
        manager = ConfigurationManager()

        with pytest.raises(ValueError, match="Unknown adapter: unknown"):
            manager.load_config("unknown")

    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        manager = ConfigurationManager()
        adapter = MockAdapter(mock_config={"invalid": "config"})
        manager.register_adapter("test", adapter)

        config = manager.load_config("test")
        assert config == {"test_key": "default_value"}  # Should use defaults
