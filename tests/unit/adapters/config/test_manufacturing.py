# tests/unit/adapters/config/test_manufacturing.py
import pytest
import json
from pathlib import Path
import tempfile

from circman5.adapters.config.manufacturing import ManufacturingAdapter


class TestManufacturingAdapter:
    """Test ManufacturingAdapter functionality."""

    def test_initialization(self):
        """Test basic initialization."""
        adapter = ManufacturingAdapter()
        assert adapter.config_path.name == "manufacturing.json"

    def test_get_defaults(self):
        """Test default configuration."""
        adapter = ManufacturingAdapter()
        defaults = adapter.get_defaults()

        assert "MANUFACTURING_STAGES" in defaults
        assert "QUALITY_THRESHOLDS" in defaults
        assert "OPTIMIZATION_TARGETS" in defaults

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        adapter = ManufacturingAdapter()
        valid_config = adapter.get_defaults()

        assert adapter.validate_config(valid_config) is True

    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        adapter = ManufacturingAdapter()
        invalid_config = {"invalid": "config"}

        assert adapter.validate_config(invalid_config) is False

    def test_validate_config_missing_stages(self):
        """Test validation with missing stages."""
        adapter = ManufacturingAdapter()
        config = {
            "QUALITY_THRESHOLDS": adapter.get_defaults()["QUALITY_THRESHOLDS"],
            "OPTIMIZATION_TARGETS": adapter.get_defaults()["OPTIMIZATION_TARGETS"],
        }

        assert adapter.validate_config(config) is False

    def test_load_config_missing_file(self):
        """Test loading configuration with missing file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_path = Path(tmp_dir) / "non_existent.json"
            adapter = ManufacturingAdapter(config_path=non_existent_path)

            config = adapter.load_config()
            assert config == adapter.get_defaults()

    def test_load_config_valid_file(self):
        """Test loading configuration from a valid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_config = {
                "MANUFACTURING_STAGES": {
                    "test_stage": {
                        "input": "test_input",
                        "output": "test_output",
                        "expected_yield": 0.95,
                    }
                },
                "QUALITY_THRESHOLDS": {
                    "min_efficiency": 18.0,
                    "max_defect_rate": 5.0,
                    "min_thickness_uniformity": 90.0,
                    "max_contamination_level": 1.0,
                },
                "OPTIMIZATION_TARGETS": {
                    "min_yield_rate": 92.0,
                    "min_energy_efficiency": 0.7,
                    "min_water_reuse": 80.0,
                    "min_recycled_content": 30.0,
                },
            }

            test_path = Path(tmp_dir) / "test_manufacturing.json"
            with open(test_path, "w") as f:
                json.dump(test_config, f)

            adapter = ManufacturingAdapter(config_path=test_path)
            config = adapter.load_config()

            assert config == test_config
