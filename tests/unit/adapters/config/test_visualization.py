# tests/unit/adapters/config/test_visualization.py
import pytest
import json
from pathlib import Path
import tempfile

from circman5.adapters.config.visualization import VisualizationAdapter


class TestVisualizationAdapter:
    """Test VisualizationAdapter functionality."""

    def test_initialization(self):
        """Test basic initialization."""
        adapter = VisualizationAdapter()
        assert adapter.config_path.name == "visualization.json"

    def test_get_defaults(self):
        """Test default configuration."""
        adapter = VisualizationAdapter()
        defaults = adapter.get_defaults()

        assert "DEFAULT_STYLE" in defaults
        assert "COLOR_PALETTE" in defaults
        assert defaults["COLOR_PALETTE"] == "husl"

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        adapter = VisualizationAdapter()
        valid_config = adapter.get_defaults()

        assert adapter.validate_config(valid_config) is True

    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        adapter = VisualizationAdapter()
        invalid_config = {"invalid": "config"}

        assert adapter.validate_config(invalid_config) is False

    def test_validate_config_invalid_style(self):
        """Test validation with invalid style."""
        adapter = VisualizationAdapter()
        config = {"DEFAULT_STYLE": "not_a_dict", "COLOR_PALETTE": "husl"}

        assert adapter.validate_config(config) is False

    def test_validate_config_invalid_palette(self):
        """Test validation with invalid color palette."""
        adapter = VisualizationAdapter()
        config = {
            "DEFAULT_STYLE": adapter.get_defaults()["DEFAULT_STYLE"],
            "COLOR_PALETTE": 42,  # Not a string
        }

        assert adapter.validate_config(config) is False

    def test_load_config_missing_file(self):
        """Test loading configuration with missing file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_path = Path(tmp_dir) / "non_existent.json"
            adapter = VisualizationAdapter(config_path=non_existent_path)

            config = adapter.load_config()
            assert config == adapter.get_defaults()

    def test_load_config_valid_file(self):
        """Test loading configuration from a valid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_config = {
                "DEFAULT_STYLE": {
                    "figure.figsize": [10, 6],
                    "axes.titlesize": 16,
                    "axes.labelsize": 14,
                    "axes.grid": True,
                },
                "COLOR_PALETTE": "viridis",
                "COLOR_PALETTE_SIZE": 10,
                "DEFAULT_DPI": 450,
            }

            test_path = Path(tmp_dir) / "test_visualization.json"
            with open(test_path, "w") as f:
                json.dump(test_config, f)

            adapter = VisualizationAdapter(config_path=test_path)
            config = adapter.load_config()

            assert config == test_config
            assert config["COLOR_PALETTE"] == "viridis"
            assert config["DEFAULT_DPI"] == 450
