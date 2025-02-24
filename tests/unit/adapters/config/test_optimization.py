# tests/unit/adapters/config/test_optimization.py
import pytest
import json
from pathlib import Path
import tempfile

from circman5.adapters.config.optimization import OptimizationAdapter


class TestOptimizationAdapter:
    """Test OptimizationAdapter functionality."""

    def test_initialization(self):
        """Test basic initialization."""
        adapter = OptimizationAdapter()
        assert adapter.config_path.name == "optimization.json"

    def test_get_defaults(self):
        """Test default configuration."""
        adapter = OptimizationAdapter()
        defaults = adapter.get_defaults()

        assert "MODEL_CONFIG" in defaults
        assert "FEATURE_COLUMNS" in defaults
        assert "OPTIMIZATION_CONSTRAINTS" in defaults
        assert "TRAINING_PARAMETERS" in defaults

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        adapter = OptimizationAdapter()
        valid_config = adapter.get_defaults()

        assert adapter.validate_config(valid_config) is True

    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        adapter = OptimizationAdapter()
        invalid_config = {"invalid": "config"}

        assert adapter.validate_config(invalid_config) is False

    def test_validate_config_missing_model_config(self):
        """Test validation with missing model config."""
        adapter = OptimizationAdapter()
        defaults = adapter.get_defaults()
        config = {
            "FEATURE_COLUMNS": defaults["FEATURE_COLUMNS"],
            "OPTIMIZATION_CONSTRAINTS": defaults["OPTIMIZATION_CONSTRAINTS"],
            "TRAINING_PARAMETERS": defaults["TRAINING_PARAMETERS"],
        }

        assert adapter.validate_config(config) is False

    def test_validate_config_missing_features(self):
        """Test validation with missing features."""
        adapter = OptimizationAdapter()
        config = adapter.get_defaults()
        # Make a copy of the config with incomplete features
        incomplete_features = config["FEATURE_COLUMNS"].copy()
        incomplete_features.remove("input_amount")

        test_config = config.copy()
        test_config["FEATURE_COLUMNS"] = incomplete_features

        assert adapter.validate_config(test_config) is False

    def test_load_config_missing_file(self):
        """Test loading configuration with missing file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_path = Path(tmp_dir) / "non_existent.json"
            adapter = OptimizationAdapter(config_path=non_existent_path)

            config = adapter.load_config()
            assert config == adapter.get_defaults()

    def test_load_config_valid_file(self):
        """Test loading configuration from a valid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = OptimizationAdapter()
            test_config = adapter.get_defaults()
            # Modify a value to verify it's loaded correctly
            test_config["MODEL_CONFIG"]["test_size"] = 0.3

            test_path = Path(tmp_dir) / "test_optimization.json"
            with open(test_path, "w") as f:
                json.dump(test_config, f)

            adapter = OptimizationAdapter(config_path=test_path)
            config = adapter.load_config()

            assert config == test_config
            assert config["MODEL_CONFIG"]["test_size"] == 0.3
