# tests/unit/adapters/config/test_monitoring.py
import pytest
import json
from pathlib import Path
import tempfile

from circman5.adapters.config.monitoring import MonitoringAdapter


class TestMonitoringAdapter:
    """Test MonitoringAdapter functionality."""

    def test_initialization(self):
        """Test basic initialization."""
        adapter = MonitoringAdapter()
        assert adapter.config_path.name == "monitoring.json"

    def test_get_defaults(self):
        """Test default configuration."""
        adapter = MonitoringAdapter()
        defaults = adapter.get_defaults()

        assert "MONITORING_WEIGHTS" in defaults
        weights = defaults["MONITORING_WEIGHTS"]
        assert "defect" in weights
        assert "yield" in weights
        assert "uniformity" in weights

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        adapter = MonitoringAdapter()
        valid_config = adapter.get_defaults()

        assert adapter.validate_config(valid_config) is True

    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        adapter = MonitoringAdapter()
        invalid_config = {"invalid": "config"}

        assert adapter.validate_config(invalid_config) is False

    def test_validate_config_invalid_weights(self):
        """Test validation with invalid weights."""
        adapter = MonitoringAdapter()
        config = {
            "MONITORING_WEIGHTS": {
                "defect": 0.4,
                "yield": 0.4
                # Missing "uniformity"
            }
        }

        assert adapter.validate_config(config) is False

    def test_validate_config_non_numeric_weights(self):
        """Test validation with non-numeric weights."""
        adapter = MonitoringAdapter()
        config = {
            "MONITORING_WEIGHTS": {"defect": 0.4, "yield": "invalid", "uniformity": 0.2}
        }

        assert adapter.validate_config(config) is False

    def test_validate_config_weight_sum(self):
        """Test validation of weight sum."""
        adapter = MonitoringAdapter()
        config = {
            "MONITORING_WEIGHTS": {
                "defect": 0.5,
                "yield": 0.5,
                "uniformity": 0.5,  # Sum > 1.0
            }
        }

        assert adapter.validate_config(config) is False

    def test_load_config_missing_file(self):
        """Test loading configuration with missing file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_path = Path(tmp_dir) / "non_existent.json"
            adapter = MonitoringAdapter(config_path=non_existent_path)

            config = adapter.load_config()
            assert config == adapter.get_defaults()

    def test_load_config_valid_file(self):
        """Test loading configuration from a valid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_config = {
                "MONITORING_WEIGHTS": {"defect": 0.3, "yield": 0.3, "uniformity": 0.4}
            }

            test_path = Path(tmp_dir) / "test_monitoring.json"
            with open(test_path, "w") as f:
                json.dump(test_config, f)

            adapter = MonitoringAdapter(config_path=test_path)
            config = adapter.load_config()

            assert config == test_config
