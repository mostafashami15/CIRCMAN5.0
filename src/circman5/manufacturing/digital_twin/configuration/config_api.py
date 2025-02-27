# src/circman5/manufacturing/digital_twin/configuration/config_api.py

from typing import Dict, Any, List, Optional, Tuple
import json

from ....utils.logging_config import setup_logger
from .config_manager import (
    config_manager,
    ParameterDefinition,
    ParameterGroup,
    ParameterCategory,
)
from ..event_notification.subscribers import Subscriber
from ..event_notification.event_types import Event, EventCategory


class ConfigurationSubscriber(Subscriber):
    """Subscriber for configuration events."""

    def __init__(self, callback=None):
        """
        Initialize configuration subscriber.

        Args:
            callback: Optional callback for parameter change events
        """
        super().__init__(name="configuration")
        self.callback = callback

    def handle_event(self, event: Event) -> None:
        """
        Handle configuration events.

        Args:
            event: Event to handle
        """
        # Check if this is a parameter change event
        if event.category == EventCategory.SYSTEM and "parameter_name" in event.details:
            # Log the change
            self.logger.info(
                f"Parameter changed: {event.details['parameter_name']} = "
                f"{event.details['new_value']}"
            )

            # Call callback if provided
            if self.callback:
                try:
                    self.callback(
                        event.details["parameter_name"],
                        event.details["old_value"],
                        event.details["new_value"],
                    )
                except Exception as e:
                    self.logger.error(f"Error in callback: {str(e)}")


class ConfigurationAPI:
    """
    API for interacting with Digital Twin configuration.

    This class provides methods for the Human-Machine Interface
    to interact with the configuration system.
    """

    def __init__(self):
        """Initialize configuration API."""
        self.logger = setup_logger("config_api")

        # Get reference to configuration manager
        self.config_manager = config_manager

        # Initialize subscriber for configuration events
        self.subscriber = ConfigurationSubscriber(callback=self._on_parameter_change)
        self.subscriber.register(category=EventCategory.SYSTEM)

        self.logger.info("Configuration API initialized")

    def _on_parameter_change(
        self, param_name: str, old_value: Any, new_value: Any
    ) -> None:
        """
        Handle parameter change event.

        Args:
            param_name: Name of the parameter
            old_value: Previous value
            new_value: New value
        """
        # This is called when a parameter changes
        # Could trigger UI updates or other actions
        pass

    def get_parameter_value(self, param_name: str) -> Any:
        """
        Get current value of a parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            Any: Current value of the parameter

        Raises:
            KeyError: If parameter not found
        """
        return self.config_manager.get_parameter_value(param_name)

    def set_parameter_value(
        self, param_name: str, value: Any
    ) -> Tuple[bool, Optional[str]]:
        """
        Set value of a parameter.

        Args:
            param_name: Name of the parameter
            value: New value

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)

        Raises:
            KeyError: If parameter not found
        """
        return self.config_manager.set_parameter_value(param_name, value)

    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """
        Get information about a parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            Dict[str, Any]: Parameter information

        Raises:
            KeyError: If parameter not found
        """
        param_def = self.config_manager.get_parameter_definition(param_name)
        param_value = self.config_manager.get_parameter_value(param_name)

        return {
            "name": param_def.name,
            "description": param_def.description,
            "value": param_value,
            "default": param_def.default_value,
            "type": param_def.parameter_type.value,
            "min_value": param_def.min_value,
            "max_value": param_def.max_value,
            "enum_values": param_def.enum_values,
            "units": param_def.units,
            "requires_restart": param_def.requires_restart,
        }

    def get_all_parameters(self) -> List[Dict[str, Any]]:
        """
        Get information about all parameters.

        Returns:
            List[Dict[str, Any]]: List of parameter information
        """
        parameters = []

        for param_name in self.config_manager.parameter_definitions:
            parameters.append(self.get_parameter_info(param_name))

        return parameters

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Get information about parameter groups.

        Returns:
            List[Dict[str, Any]]: List of parameter group information
        """
        groups = []

        for group in self.config_manager.get_parameter_groups():
            # Get parameters in this group
            params = []
            for param in group.parameters:
                param_info = self.get_parameter_info(param.name)
                params.append(param_info)

            groups.append(
                {
                    "name": group.name,
                    "description": group.description,
                    "category": group.category.value,
                    "parameters": params,
                }
            )

        return groups

    def reset_parameter(self, param_name: str) -> None:
        """
        Reset parameter to its default value.

        Args:
            param_name: Name of the parameter

        Raises:
            KeyError: If parameter not found
        """
        self.config_manager.reset_parameter_to_default(param_name)

    def reset_all_parameters(self) -> None:
        """Reset all parameters to their default values."""
        self.config_manager.reset_all_parameters()

    def export_configuration(self) -> str:
        """
        Export current configuration as JSON.

        Returns:
            str: JSON string of current configuration
        """
        config = {
            "parameters": self.config_manager.get_all_parameter_values(),
            "timestamp": self.config_manager._instance._get_timestamp(),
        }

        return json.dumps(config, indent=2)

    def import_configuration(self, config_json: str) -> Tuple[bool, Optional[str]]:
        """
        Import configuration from JSON.

        Args:
            config_json: JSON string of configuration

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            config = json.loads(config_json)

            if "parameters" not in config:
                return False, "Invalid configuration format"

            # Validate and set parameters
            for param_name, value in config["parameters"].items():
                try:
                    success, error = self.set_parameter_value(param_name, value)
                    if not success:
                        self.logger.warning(
                            f"Failed to set parameter '{param_name}': {error}"
                        )
                except KeyError:
                    self.logger.warning(f"Unknown parameter in import: {param_name}")

            return True, None

        except json.JSONDecodeError:
            return False, "Invalid JSON format"
        except Exception as e:
            return False, f"Error importing configuration: {str(e)}"
