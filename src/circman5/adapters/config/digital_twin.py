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
                "default_simulation_steps": 10,
                "target_temperature": 22.5,
                "temperature_regulation": 0.1,
                "silicon_wafer_consumption_rate": 0.5,
                "solar_glass_consumption_rate": 0.2,
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
            "EVENT_NOTIFICATION": {
                "persistence_enabled": True,
                "max_events": 1000,
                "default_alert_severity": "warning",
                "publish_state_changes": True,
                "publish_threshold_breaches": True,
                "publish_simulation_results": True,
            },
            "PARAMETERS": {
                "PARAMETER_GROUPS": [
                    {
                        "name": "Process Control",
                        "description": "Manufacturing process control parameters",
                        "category": "process",
                        "parameters": [
                            {
                                "name": "target_temperature",
                                "description": "Target temperature for manufacturing process",
                                "type": "float",
                                "default_value": 22.5,
                                "path": "production_line.temperature",
                                "min_value": 18.0,
                                "max_value": 30.0,
                                "units": "°C",
                                "tags": ["temperature", "control"],
                            },
                            {
                                "name": "energy_limit",
                                "description": "Maximum energy consumption allowed",
                                "type": "float",
                                "default_value": 200.0,
                                "path": "production_line.energy_limit",
                                "min_value": 50.0,
                                "max_value": 500.0,
                                "units": "kWh",
                                "tags": ["energy", "limit"],
                            },
                            {
                                "name": "production_rate",
                                "description": "Target production rate",
                                "type": "float",
                                "default_value": 5.0,
                                "path": "production_line.production_rate",
                                "min_value": 1.0,
                                "max_value": 10.0,
                                "units": "units/hour",
                                "tags": ["production", "rate"],
                            },
                        ],
                    },
                    {
                        "name": "Quality Control",
                        "description": "Quality control parameters",
                        "category": "quality",
                        "parameters": [
                            {
                                "name": "defect_threshold",
                                "description": "Maximum allowed defect rate",
                                "type": "float",
                                "default_value": 0.05,
                                "path": "production_line.defect_threshold",
                                "min_value": 0.01,
                                "max_value": 0.1,
                                "units": "",
                                "tags": ["quality", "threshold"],
                            },
                            {
                                "name": "inspection_frequency",
                                "description": "Frequency of quality inspections",
                                "type": "integer",
                                "default_value": 10,
                                "path": "production_line.inspection_frequency",
                                "min_value": 1,
                                "max_value": 100,
                                "units": "units",
                                "tags": ["quality", "inspection"],
                            },
                        ],
                    },
                    {
                        "name": "System Settings",
                        "description": "General system settings",
                        "category": "system",
                        "parameters": [
                            {
                                "name": "update_frequency",
                                "description": "Digital twin update frequency",
                                "type": "float",
                                "default_value": 1.0,
                                "path": "",
                                "min_value": 0.1,
                                "max_value": 10.0,
                                "units": "Hz",
                                "tags": ["system", "performance"],
                                "requires_restart": True,
                            },
                            {
                                "name": "log_level",
                                "description": "Logging level",
                                "type": "enum",
                                "default_value": "INFO",
                                "path": "",
                                "enum_values": [
                                    "DEBUG",
                                    "INFO",
                                    "WARNING",
                                    "ERROR",
                                    "CRITICAL",
                                ],
                                "tags": ["system", "logging"],
                            },
                        ],
                    },
                ]
            },
            "PARAMETER_THRESHOLDS": {
                "production_line.temperature": {
                    "name": "Production Line Temperature",
                    "value": 25.0,
                    "comparison": "greater_than",
                    "severity": "WARNING",
                },
                "production_line.energy_consumption": {
                    "name": "Energy Consumption",
                    "value": 200.0,
                    "comparison": "greater_than",
                    "severity": "WARNING",
                },
                "production_line.defect_rate": {
                    "name": "Defect Rate",
                    "value": 0.1,
                    "comparison": "greater_than",
                    "severity": "ERROR",
                },
            },
            "AI_INTEGRATION": {
                "DEFAULT_PARAMETERS": {
                    "input_amount": 100.0,
                    "energy_used": 50.0,
                    "cycle_time": 30.0,
                    "efficiency": 0.9,
                    "defect_rate": 0.05,
                    "thickness_uniformity": 95.0,
                },
                "PARAMETER_MAPPING": {
                    "production_rate": "output_amount",
                    "energy_consumption": "energy_used",
                    "temperature": "temperature",
                    "cycle_time": "cycle_time",
                },
                "OPTIMIZATION_CONSTRAINTS": {
                    "energy_used": [10.0, 100.0],
                    "cycle_time": [20.0, 60.0],
                    "defect_rate": [0.01, 0.1],
                },
            },
        }
