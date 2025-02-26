# src/circman5/adapters/config/digital_twin.py

"""
Configuration adapter for Digital Twin module.

This module provides configuration loading and validation for the Digital Twin system.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

from ..base.adapter_base import ConfigAdapterBase


class DigitalTwinAdapter(ConfigAdapterBase):
    """Adapter for Digital Twin configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Digital Twin adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "digital_twin.json"
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load Digital Twin configuration.

        Returns:
            Dict[str, Any]: Digital Twin configuration

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
        Validate Digital Twin configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        required_sections = {
            "DIGITAL_TWIN_CONFIG",
            "SIMULATION_PARAMETERS",
            "SYNCHRONIZATION_CONFIG",
            "STATE_MANAGEMENT",
        }

        # Check required top-level sections
        if not all(section in config for section in required_sections):
            self.logger.error(
                f"Missing required sections: {required_sections - set(config.keys())}"
            )
            return False

        # Validate digital twin config
        dt_config = config.get("DIGITAL_TWIN_CONFIG", {})
        required_dt_params = {
            "name",
            "update_frequency",
            "history_length",
            "simulation_steps",
            "synchronization_mode",
        }

        if not all(param in dt_config for param in required_dt_params):
            self.logger.error("Invalid digital twin configuration")
            return False

        # Validate simulation parameters
        sim_params = config.get("SIMULATION_PARAMETERS", {})
        required_sim_params = {
            "temperature_increment",
            "energy_consumption_increment",
            "production_rate_increment",
            "default_simulation_steps",
        }

        if not all(param in sim_params for param in required_sim_params):
            self.logger.error("Invalid simulation parameters")
            return False

        # Validate sync config
        sync_config = config.get("SYNCHRONIZATION_CONFIG", {})
        required_sync_params = {
            "default_sync_interval",
            "default_sync_mode",
            "retry_interval",
        }

        if not all(param in sync_config for param in required_sync_params):
            self.logger.error("Invalid synchronization configuration")
            return False

        return True

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default Digital Twin configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "DIGITAL_TWIN_CONFIG": {
                "name": "SoliTek_DigitalTwin",
                "update_frequency": 1.0,
                "history_length": 1000,
                "simulation_steps": 10,
                "data_sources": ["sensors", "manual_input", "manufacturing_system"],
                "synchronization_mode": "real_time",
                "log_level": "INFO",
            },
            "SIMULATION_PARAMETERS": {
                "temperature_increment": 0.5,
                "energy_consumption_increment": 2.0,
                "production_rate_increment": 0.2,
                "target_temperature": 22.5,
                "temperature_regulation": 0.1,
                "silicon_wafer_consumption_rate": 0.5,
                "solar_glass_consumption_rate": 0.2,
                "default_simulation_steps": 10,
            },
            "SYNCHRONIZATION_CONFIG": {
                "default_sync_interval": 1.0,
                "default_sync_mode": "real_time",
                "retry_interval": 1.0,
                "timeout": 10.0,
            },
            "STATE_MANAGEMENT": {
                "default_history_length": 1000,
                "validation_level": "standard",
                "auto_timestamp": True,
            },
            "SCENARIO_MANAGEMENT": {"max_scenarios": 100},
        }
