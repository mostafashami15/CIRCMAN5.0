# tests/unit/adapters/base/test_adapter_base.py
import pytest
import tempfile
import json
from pathlib import Path

from circman5.adapters.base.adapter_base import ConfigAdapterBase


# Create concrete test implementation of the abstract base class
class TestAdapter(ConfigAdapterBase):
    """Concrete implementation of ConfigAdapterBase for testing."""

    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.config_path = config_path or Path("test_config.json")

    def load_config(self):
        if not self.config_path.exists():
            return self.get_defaults()
        return self._load_json_config(self.config_path)

    def validate_config(self, config):
        return "test_key" in config

    def get_defaults(self):
        return {"test_key": "test_value"}


class TestConfigAdapterBase:
    """Test the ConfigAdapterBase class functionality."""

    def test_initialization(self):
        """Test that adapter initializes properly."""
        adapter = TestAdapter()
        assert adapter.config_path == Path("test_config.json")
        assert hasattr(adapter, "logger")

    def test_load_json_config(self):
        """Test the _load_json_config helper method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test config file
            test_config = {"test_key": "test_value"}
            config_path = Path(tmp_dir) / "test_config.json"

            with open(config_path, "w") as f:
                json.dump(test_config, f)

            # Test loading the config
            adapter = TestAdapter(config_path)
            loaded_config = adapter._load_json_config(config_path)

            assert loaded_config == test_config

    def test_load_json_config_file_not_found(self):
        """Test handling of file not found errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_path = Path(tmp_dir) / "non_existent.json"
            adapter = TestAdapter(non_existent_path)

            with pytest.raises(FileNotFoundError):
                adapter._load_json_config(non_existent_path)

    def test_load_json_config_invalid_json(self):
        """Test handling of invalid JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create invalid JSON file
            config_path = Path(tmp_dir) / "invalid.json"

            with open(config_path, "w") as f:
                f.write("{invalid: json")

            adapter = TestAdapter(config_path)

            with pytest.raises(ValueError, match="Invalid JSON configuration"):
                adapter._load_json_config(config_path)
