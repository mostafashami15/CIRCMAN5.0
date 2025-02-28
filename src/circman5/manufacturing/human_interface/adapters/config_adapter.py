# src/circman5/manufacturing/human_interface/adapters/config_adapter.py

"""
Configuration adapter for CIRCMAN5.0 Human-Machine Interface.

This module provides configuration loading and validation for the Human-Machine
Interface system, following the adapter pattern used throughout the application.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from ....utils.logging_config import setup_logger
from circman5.adapters.base.adapter_base import ConfigAdapterBase


class HumanInterfaceAdapter(ConfigAdapterBase):
    """
    Adapter for Human-Machine Interface configuration.

    This class handles loading, validation, and access to configuration
    for the human interface system.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Human Interface adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path
            or Path(__file__).parent.parent.parent.parent.parent
            / "config"
            / "human_interface.json"
        )
        self.logger = setup_logger("human_interface_adapter")

    def load_config(self) -> Dict[str, Any]:
        """
        Load Human Interface configuration.

        Returns:
            Dict[str, Any]: Human Interface configuration

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
        Validate Human Interface configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        # Check required sections
        required_sections = {
            "INTERFACE_CONFIG",
            "DASHBOARD_CONFIG",
            "PANEL_TYPES",
            "THEME",
        }

        if not all(section in config for section in required_sections):
            self.logger.error(
                f"Missing required sections: {required_sections - set(config.keys())}"
            )
            return False

        # Validate interface config
        interface_config = config.get("INTERFACE_CONFIG", {})
        required_interface_params = {
            "refresh_interval",
            "default_view",
            "enable_animations",
        }

        if not all(param in interface_config for param in required_interface_params):
            self.logger.error("Invalid interface configuration")
            return False

        # Validate dashboard config
        dashboard_config = config.get("DASHBOARD_CONFIG", {})
        required_dashboard_params = {"default_layout", "panel_spacing", "auto_refresh"}

        if not all(param in dashboard_config for param in required_dashboard_params):
            self.logger.error("Invalid dashboard configuration")
            return False

        return True

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default Human Interface configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "INTERFACE_CONFIG": {
                "refresh_interval": 5.0,  # Seconds
                "default_view": "main_dashboard",
                "enable_animations": True,
                "enable_sounds": False,
                "log_level": "INFO",
            },
            "DASHBOARD_CONFIG": {
                "default_layout": "main_dashboard",
                "panel_spacing": 10,
                "auto_refresh": True,
                "refresh_interval": 5.0,  # Seconds
                "max_history_points": 100,
            },
            "PANEL_TYPES": {
                "status_panel": {
                    "name": "Status Panel",
                    "description": "Displays system status information",
                    "default_size": {"rows": 1, "cols": 1},
                },
                "kpi_panel": {
                    "name": "KPI Panel",
                    "description": "Displays key performance indicators",
                    "default_size": {"rows": 1, "cols": 1},
                },
                "process_panel": {
                    "name": "Process Panel",
                    "description": "Displays manufacturing process information",
                    "default_size": {"rows": 1, "cols": 2},
                },
                "alert_panel": {
                    "name": "Alert Panel",
                    "description": "Displays system alerts and notifications",
                    "default_size": {"rows": 1, "cols": 2},
                },
                "chart_panel": {
                    "name": "Chart Panel",
                    "description": "Displays charts and visualizations",
                    "default_size": {"rows": 1, "cols": 1},
                },
                "parameter_panel": {
                    "name": "Parameter Panel",
                    "description": "Displays and allows editing of parameters",
                    "default_size": {"rows": 1, "cols": 1},
                },
                "control_panel": {
                    "name": "Control Panel",
                    "description": "Provides process control interface",
                    "default_size": {"rows": 1, "cols": 2},
                },
            },
            "THEME": {
                "dark_mode": True,
                "colors": {
                    "primary": "#1976d2",
                    "secondary": "#dc004e",
                    "background": "#121212",
                    "surface": "#1e1e1e",
                    "error": "#cf6679",
                    "warning": "#ffb74d",
                    "info": "#64b5f6",
                    "success": "#81c784",
                    "text": {"primary": "#ffffff", "secondary": "#b0bec5"},
                },
                "panel": {
                    "border_radius": "4px",
                    "shadow": "0 2px 4px rgba(0,0,0,0.2)",
                    "header_bg": "#2c2c2c",
                },
            },
        }
