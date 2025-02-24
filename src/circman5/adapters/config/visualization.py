# src/circman5/adapters/config/visualization.py

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json

from ..base.adapter_base import ConfigAdapterBase


class VisualizationAdapter(ConfigAdapterBase):
    """Adapter for visualization configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize visualization adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "visualization.json"
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load visualization configuration.

        Returns:
            Dict[str, Any]: Visualization configuration

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        if not self.config_path.exists():
            self.logger.warning(
                f"Config file not found: {self.config_path}, using defaults"
            )
            return self.get_defaults()

        return self._load_json_config(self.config_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate visualization configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        required_keys = {
            "DEFAULT_STYLE",
            "COLOR_PALETTE",
        }

        # Check required top-level keys
        if not all(key in config for key in required_keys):
            self.logger.error(
                f"Missing required keys: {required_keys - set(config.keys())}"
            )
            return False

        # Validate DEFAULT_STYLE has required attributes
        style = config.get("DEFAULT_STYLE", {})
        if not isinstance(style, dict):
            self.logger.error("DEFAULT_STYLE must be a dictionary")
            return False

        # Check color palette
        color_palette = config.get("COLOR_PALETTE")
        if not isinstance(color_palette, str):
            self.logger.error("COLOR_PALETTE must be a string")
            return False

        return True

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default visualization configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "DEFAULT_STYLE": {
                "figure.figsize": [12, 8],
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "axes.grid": True,
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
            },
            "COLOR_PALETTE": "husl",
            "COLOR_PALETTE_SIZE": 8,
            "DEFAULT_DPI": 300,
        }
