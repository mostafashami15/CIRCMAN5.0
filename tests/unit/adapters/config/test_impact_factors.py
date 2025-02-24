# tests/unit/adapters/config/test_impact_factors.py
import pytest
import json
from pathlib import Path
import tempfile

from circman5.adapters.config.impact_factors import ImpactFactorsAdapter


class TestImpactFactorsAdapter:
    """Test ImpactFactorsAdapter functionality."""

    def test_initialization(self):
        """Test basic initialization."""
        adapter = ImpactFactorsAdapter()
        assert adapter.config_path.name == "impact_factors.json"

    def test_get_defaults(self):
        """Test default configuration."""
        adapter = ImpactFactorsAdapter()
        defaults = adapter.get_defaults()

        assert "MATERIAL_IMPACT_FACTORS" in defaults
        assert "ENERGY_IMPACT_FACTORS" in defaults
        assert "RECYCLING_BENEFIT_FACTORS" in defaults

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        adapter = ImpactFactorsAdapter()
        valid_config = adapter.get_defaults()

        assert adapter.validate_config(valid_config) is True

    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        adapter = ImpactFactorsAdapter()
        invalid_config = {"invalid": "config"}

        assert adapter.validate_config(invalid_config) is False

    def test_load_config_missing_file(self):
        """Test loading configuration with missing file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_path = Path(tmp_dir) / "non_existent.json"
            adapter = ImpactFactorsAdapter(config_path=non_existent_path)

            config = adapter.load_config()
            assert config == adapter.get_defaults()

    def test_load_config_valid_file(self):
        """Test loading configuration from a valid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_config = {
                "MATERIAL_IMPACT_FACTORS": {"test": 1.0},
                "ENERGY_IMPACT_FACTORS": {"test": 0.5},
                "RECYCLING_BENEFIT_FACTORS": {"test": -1.0},
                "TRANSPORT_IMPACT_FACTORS": {"test": 0.1},
                "PROCESS_IMPACT_FACTORS": {"test": 0.2},
                "GRID_CARBON_INTENSITIES": {"test": 0.3},
                "DEGRADATION_RATES": {"test": 0.4},
            }

            test_path = Path(tmp_dir) / "test_impact_factors.json"
            with open(test_path, "w") as f:
                json.dump(test_config, f)

            adapter = ImpactFactorsAdapter(config_path=test_path)
            config = adapter.load_config()

            assert config == test_config
