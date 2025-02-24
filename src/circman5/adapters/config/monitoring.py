# src/circman5/adapters/config/monitoring.py

from pathlib import Path
from typing import Dict, Any, Optional
import json

from ..base.adapter_base import ConfigAdapterBase


class MonitoringAdapter(ConfigAdapterBase):
    """Adapter for monitoring system configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize monitoring adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "monitoring.json"
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load monitoring configuration.

        Returns:
            Dict[str, Any]: Monitoring configuration

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
        Validate monitoring configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        required_keys = {
            "MONITORING_WEIGHTS",
        }

        # Check required top-level keys
        if not all(key in config for key in required_keys):
            self.logger.error(
                f"Missing required keys: {required_keys - set(config.keys())}"
            )
            return False

        # Validate monitoring weights
        weights = config.get("MONITORING_WEIGHTS", {})
        required_weights = {"defect", "yield", "uniformity"}

        if not all(weight in weights for weight in required_weights):
            self.logger.error("Missing required monitoring weights")
            return False

        # Validate that weights are numeric and sum to approximately 1.0
        if not all(isinstance(v, (int, float)) for v in weights.values()):
            self.logger.error("All weight values must be numeric")
            return False

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            self.logger.warning(
                f"Monitoring weights should sum to 1.0, but got {total_weight}"
            )
            return False

        return True

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default monitoring configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "MONITORING_WEIGHTS": {
                "defect": 0.4,
                "yield": 0.4,
                "uniformity": 0.2,
            },
        }
