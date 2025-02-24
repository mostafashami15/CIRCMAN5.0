# src/circman5/adapters/config/manufacturing.py

from pathlib import Path
from typing import Dict, Any, Optional
import json

from ..base.adapter_base import ConfigAdapterBase


class ManufacturingAdapter(ConfigAdapterBase):
    """Adapter for manufacturing constants and configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize manufacturing adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "manufacturing.json"
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load manufacturing configuration.

        Returns:
            Dict[str, Any]: Manufacturing configuration

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
        Validate manufacturing configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        required_keys = {
            "MANUFACTURING_STAGES",
            "QUALITY_THRESHOLDS",
            "OPTIMIZATION_TARGETS",
        }

        # Check required top-level keys
        if not all(key in config for key in required_keys):
            self.logger.error(
                f"Missing required keys: {required_keys - set(config.keys())}"
            )
            return False

        # Validate manufacturing stages
        stages = config.get("MANUFACTURING_STAGES", {})
        for stage, data in stages.items():
            if not all(key in data for key in ["input", "output", "expected_yield"]):
                self.logger.error(f"Invalid stage configuration: {stage}")
                return False

            if not isinstance(data["expected_yield"], (int, float)):
                self.logger.error(f"Invalid yield value for stage: {stage}")
                return False

        # Validate quality thresholds
        thresholds = config.get("QUALITY_THRESHOLDS", {})
        required_thresholds = {
            "min_efficiency",
            "max_defect_rate",
            "min_thickness_uniformity",
            "max_contamination_level",
        }

        if not all(key in thresholds for key in required_thresholds):
            self.logger.error("Missing required quality thresholds")
            return False

        # Validate optimization targets
        targets = config.get("OPTIMIZATION_TARGETS", {})
        required_targets = {
            "min_yield_rate",
            "min_energy_efficiency",
            "min_water_reuse",
            "min_recycled_content",
        }

        if not all(key in targets for key in required_targets):
            self.logger.error("Missing required optimization targets")
            return False

        return True

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default manufacturing configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "MANUFACTURING_STAGES": {
                "silicon_purification": {
                    "input": "raw_silicon",
                    "output": "purified_silicon",
                    "expected_yield": 0.90,
                },
                "wafer_production": {
                    "input": "purified_silicon",
                    "output": "silicon_wafer",
                    "expected_yield": 0.95,
                },
                "cell_production": {
                    "input": "silicon_wafer",
                    "output": "solar_cell",
                    "expected_yield": 0.98,
                },
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
