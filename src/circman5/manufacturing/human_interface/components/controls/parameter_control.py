# src/circman5/manufacturing/human_interface/components/controls/parameter_control.py

"""
Parameter control component for CIRCMAN5.0 Human-Machine Interface.

This module implements the parameter control interface, allowing users to view,
edit, and apply changes to system parameters through the digital twin's
configuration interface.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
import datetime
import threading

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.dashboard_manager import dashboard_manager
from ...core.interface_state import interface_state
from ...adapters.event_adapter import EventAdapter
from ....digital_twin.configuration.config_api import ConfigurationAPI
from ....digital_twin.configuration.parameter_definition import (
    ParameterDefinition,
    ParameterGroup,
    ParameterCategory,
    ParameterType,
)


class ParameterControl:
    """
    Parameter control component for the Human-Machine Interface.

    This component provides an interface for viewing, editing, and applying
    changes to system parameters through the digital twin's configuration interface.

    Attributes:
        state: Reference to interface state
        config_api: Reference to configuration API
        event_adapter: Event adapter for notifications
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the parameter control."""
        self.logger = setup_logger("parameter_control")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.config_api = ConfigurationAPI()  # Get instance
        self.event_adapter = EventAdapter()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        # Cache for parameter data
        self._parameter_cache: Dict[str, Any] = {}
        self._last_update = datetime.datetime.min
        self._cache_ttl = datetime.timedelta(seconds=5)  # Cache TTL of 5 seconds

        # Register with interface manager
        interface_manager.register_component("parameter_control", self)

        # Register for configuration events
        self.event_adapter.register_callback(
            self._on_parameter_change, category=None  # Register for all categories
        )

        self.logger.info("Parameter Control initialized")

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups.

        Returns:
            List[Dict[str, Any]]: Parameter groups data
        """
        # Check if cached data is still valid
        now = datetime.datetime.now()
        if (
            now - self._last_update < self._cache_ttl
            and "groups" in self._parameter_cache
        ):
            return self._parameter_cache["groups"].copy()

        # Get parameter groups from configuration API
        try:
            groups = self.config_api.get_parameter_groups()

            # Cache the groups
            self._parameter_cache["groups"] = groups
            self._last_update = now

            return groups

        except Exception as e:
            self.logger.error(f"Error getting parameter groups: {str(e)}")
            return []

    def get_parameters(self, group_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get parameters, optionally filtered by group.

        Args:
            group_name: Optional group name to filter by

        Returns:
            List[Dict[str, Any]]: Parameter data
        """
        # Get all parameters
        try:
            all_params = self.config_api.get_all_parameters()

            # Filter by group if specified
            if group_name:
                groups = self.get_parameter_groups()
                for group in groups:
                    if group["name"] == group_name:
                        return group["parameters"]

                # Group not found
                self.logger.warning(f"Parameter group not found: {group_name}")
                return []

            return all_params

        except Exception as e:
            self.logger.error(f"Error getting parameters: {str(e)}")
            return []

    def get_parameter(self, param_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific parameter by name.

        Args:
            param_name: Parameter name

        Returns:
            Optional[Dict[str, Any]]: Parameter data or None if not found
        """
        try:
            return self.config_api.get_parameter_info(param_name)
        except KeyError:
            self.logger.warning(f"Parameter not found: {param_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting parameter {param_name}: {str(e)}")
            return None

    def set_parameter_value(self, param_name: str, value: Any) -> Dict[str, Any]:
        """
        Set a parameter value.

        Args:
            param_name: Parameter name
            value: New parameter value

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            # Convert string values to appropriate types
            if isinstance(value, str):
                # Get parameter info to determine type
                param_info = self.get_parameter(param_name)
                if param_info:
                    param_type = param_info.get("type")
                    if param_type == "float":
                        value = float(value)
                    elif param_type == "integer":
                        value = int(value)
                    elif param_type == "boolean":
                        value = value.lower() in ("true", "yes", "1", "t", "y")

            # Set parameter value
            success, error = self.config_api.set_parameter_value(param_name, value)

            if success:
                self.logger.info(f"Parameter {param_name} set to {value}")

                # Invalidate cache
                self._last_update = datetime.datetime.min

                return {"success": True}
            else:
                self.logger.warning(f"Failed to set parameter {param_name}: {error}")
                return {"success": False, "error": error}

        except Exception as e:
            self.logger.error(f"Error setting parameter {param_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    def reset_parameter(self, param_name: str) -> Dict[str, Any]:
        """
        Reset a parameter to its default value.

        Args:
            param_name: Parameter name

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            self.config_api.reset_parameter(param_name)

            # Invalidate cache
            self._last_update = datetime.datetime.min

            self.logger.info(f"Parameter {param_name} reset to default")
            return {"success": True}

        except Exception as e:
            self.logger.error(f"Error resetting parameter {param_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    def reset_all_parameters(self) -> Dict[str, Any]:
        """
        Reset all parameters to their default values.

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            self.config_api.reset_all_parameters()

            # Invalidate cache
            self._last_update = datetime.datetime.min

            self.logger.info("All parameters reset to defaults")
            return {"success": True}

        except Exception as e:
            self.logger.error(f"Error resetting all parameters: {str(e)}")
            return {"success": False, "error": str(e)}

    def export_configuration(self) -> Dict[str, Any]:
        """
        Export current configuration as JSON.

        Returns:
            Dict[str, Any]: Result with success, config JSON, and error message if applicable
        """
        try:
            config_json = self.config_api.export_configuration()

            self.logger.info("Configuration exported")
            return {"success": True, "config": config_json}

        except Exception as e:
            self.logger.error(f"Error exporting configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def import_configuration(self, config_json: str) -> Dict[str, Any]:
        """
        Import configuration from JSON.

        Args:
            config_json: Configuration JSON string

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            success, error = self.config_api.import_configuration(config_json)

            if success:
                # Invalidate cache
                self._last_update = datetime.datetime.min

                self.logger.info("Configuration imported successfully")
                return {"success": True}
            else:
                self.logger.warning(f"Failed to import configuration: {error}")
                return {"success": False, "error": error}

        except Exception as e:
            self.logger.error(f"Error importing configuration: {str(e)}")
            return {"success": False, "error": str(e)}

    def _on_parameter_change(self, event: Any) -> None:
        """
        Handle parameter change events.

        Args:
            event: Event data
        """
        # Check if event has details and is a parameter change
        if hasattr(event, "details") and "parameter_name" in event.details:
            # Invalidate cache
            self._last_update = datetime.datetime.min

            param_name = event.details["parameter_name"]
            new_value = event.details["new_value"]

            self.logger.debug(f"Parameter changed: {param_name} = {new_value}")

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle parameter control commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "get_parameter_groups":
            groups = self.get_parameter_groups()
            return {"handled": True, "success": True, "groups": groups}

        elif command == "get_parameters":
            group_name = params.get("group_name")
            parameters = self.get_parameters(group_name)
            return {"handled": True, "success": True, "parameters": parameters}

        elif command == "get_parameter":
            param_name = params.get("param_name")
            if param_name:
                parameter = self.get_parameter(param_name)
                if parameter:
                    return {"handled": True, "success": True, "parameter": parameter}
                return {
                    "handled": True,
                    "success": False,
                    "error": f"Parameter not found: {param_name}",
                }
            return {
                "handled": True,
                "success": False,
                "error": "Missing param_name parameter",
            }

        elif command == "set_parameter":
            param_name = params.get("param_name")
            value = params.get("value")

            if not param_name or value is None:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing param_name or value parameter",
                }

            result = self.set_parameter_value(param_name, value)
            return {"handled": True, **result}

        elif command == "reset_parameter":
            param_name = params.get("param_name")

            if not param_name:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing param_name parameter",
                }

            result = self.reset_parameter(param_name)
            return {"handled": True, **result}

        elif command == "reset_all_parameters":
            result = self.reset_all_parameters()
            return {"handled": True, **result}

        elif command == "export_configuration":
            result = self.export_configuration()
            return {"handled": True, **result}

        elif command == "import_configuration":
            config_json = params.get("config")

            if not config_json:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing config parameter",
                }

            result = self.import_configuration(config_json)
            return {"handled": True, **result}

        # Not a parameter control command
        return {"handled": False}


# Create global instance
parameter_control = ParameterControl()
