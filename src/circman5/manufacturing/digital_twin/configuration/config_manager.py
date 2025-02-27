# src/circman5/manufacturing/digital_twin/configuration/config_manager.py

from typing import Dict, Any, Optional, List, Set, Tuple
import json
import datetime
from pathlib import Path

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from .parameter_definition import ParameterDefinition, ParameterGroup, ParameterCategory
from ..event_notification.publishers import Publisher
from ..event_notification.event_types import EventCategory, EventSeverity, Event
from .parameter_definition import (
    ParameterDefinition,
    ParameterGroup,
    ParameterCategory,
    ParameterType,
)


class ParameterConfigEvent(Event):
    """Event for parameter configuration changes."""

    def __init__(self, parameter_name: str, old_value: Any, new_value: Any, **kwargs):
        # Extract details from kwargs
        original_details = kwargs.pop("details", {})

        # Create new details dictionary
        details = {
            "parameter_name": parameter_name,
            "old_value": old_value,
            "new_value": new_value,
            **original_details,
        }

        super().__init__(
            category=EventCategory.SYSTEM,
            severity=kwargs.pop("severity", EventSeverity.INFO),
            message=f"Parameter '{parameter_name}' changed from {old_value} to {new_value}",
            details=details,
            **kwargs,
        )


class ConfigurationPublisher(Publisher):
    """Publisher for configuration events."""

    def __init__(self):
        """Initialize configuration publisher."""
        super().__init__(source="configuration")

    def publish_parameter_change(
        self,
        parameter_name: str,
        old_value: Any,
        new_value: Any,
        parameter_path: Optional[str] = None,
        requires_restart: bool = False,
    ) -> None:
        """
        Publish a parameter change event.

        Args:
            parameter_name: Name of the parameter
            old_value: Previous value
            new_value: New value
            parameter_path: Path to the parameter in the state
            requires_restart: Whether this change requires restart
        """
        severity = EventSeverity.WARNING if requires_restart else EventSeverity.INFO

        event = ParameterConfigEvent(
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=new_value,
            severity=severity,
            details={
                "parameter_path": parameter_path,
                "requires_restart": requires_restart,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

        self.publish_event(event)


class ConfigurationManager:
    """
    Manager for Digital Twin configuration parameters.

    This class handles loading, validation, and persistence of
    configuration parameters for the Digital Twin.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize configuration manager."""
        if self._initialized:
            return

        self.logger = setup_logger("config_manager")
        self.constants = ConstantsService()

        # Initialize configuration data
        self.parameter_definitions: Dict[str, ParameterDefinition] = {}
        self.parameter_groups: Dict[str, ParameterGroup] = {}
        self.current_values: Dict[str, Any] = {}

        # Initialize event publisher
        self.publisher = ConfigurationPublisher()

        # Load configuration definitions
        self._load_parameter_definitions()

        # Load current values
        self._load_current_values()

        self._initialized = True
        self.logger.info("Configuration Manager initialized")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.datetime.now().isoformat()

    def _load_parameter_definitions(self) -> None:
        """Load parameter definitions from constants service."""
        try:
            # Get digital twin config
            dt_config = self.constants.get_digital_twin_config()

            # Load parameter definitions if available
            if "PARAMETERS" in dt_config:
                param_config = dt_config["PARAMETERS"]

                # Process parameter groups
                for group_data in param_config.get("PARAMETER_GROUPS", []):
                    group_name = group_data.get("name")
                    if not group_name:
                        continue

                    # Create parameter group
                    try:
                        category = ParameterCategory(
                            group_data.get("category", "system")
                        )
                    except (ValueError, TypeError):
                        category = ParameterCategory.SYSTEM

                    group = ParameterGroup(
                        name=group_name,
                        description=group_data.get("description", ""),
                        category=category,
                    )

                    # Add parameters to group
                    for param_data in group_data.get("parameters", []):
                        param_name = param_data.get("name")
                        if not param_name:
                            continue

                        try:
                            param_type = ParameterType(param_data.get("type", "float"))
                        except (ValueError, TypeError):
                            param_type = ParameterType.FLOAT

                        # Create parameter definition
                        param = ParameterDefinition(
                            name=param_name,
                            description=param_data.get("description", ""),
                            parameter_type=param_type,
                            default_value=param_data.get("default_value"),
                            path=param_data.get("path", ""),
                            min_value=param_data.get("min_value"),
                            max_value=param_data.get("max_value"),
                            enum_values=param_data.get("enum_values"),
                            pattern=param_data.get("pattern"),
                            units=param_data.get("units"),
                            tags=param_data.get("tags", []),
                            requires_restart=param_data.get("requires_restart", False),
                        )

                        # Add parameter to group
                        group.parameters.append(param)

                        # Add to definitions dictionary
                        self.parameter_definitions[param_name] = param

                    # Add group to groups dictionary
                    self.parameter_groups[group_name] = group

                self.logger.info(
                    f"Loaded {len(self.parameter_definitions)} parameter definitions "
                    f"in {len(self.parameter_groups)} groups"
                )
            else:
                self.logger.warning("No parameter definitions found in configuration")

        except Exception as e:
            self.logger.error(f"Error loading parameter definitions: {str(e)}")

    def _load_current_values(self) -> None:
        """Load current parameter values from saved configuration."""
        try:
            # Try to load from saved configuration
            config_dir = results_manager.get_path("config")
            config_file = config_dir / "digital_twin_parameters.json"

            if config_file.exists():
                with open(config_file, "r") as f:
                    saved_values = json.load(f)

                # Validate and set values
                for param_name, value in saved_values.items():
                    if param_name in self.parameter_definitions:
                        param_def = self.parameter_definitions[param_name]
                        is_valid, _ = param_def.validate(value)

                        if is_valid:
                            self.current_values[param_name] = value
                        else:
                            # Use default value if invalid
                            self.current_values[param_name] = param_def.default_value
                            self.logger.warning(
                                f"Invalid saved value for parameter '{param_name}', "
                                f"using default: {param_def.default_value}"
                            )
                    else:
                        # Ignore parameters not in definitions
                        self.logger.warning(
                            f"Unknown parameter in saved config: {param_name}"
                        )

                self.logger.info(f"Loaded {len(self.current_values)} parameter values")
            else:
                # Initialize with default values
                self._initialize_default_values()

        except Exception as e:
            self.logger.error(f"Error loading parameter values: {str(e)}")
            # Initialize with default values
            self._initialize_default_values()

    def _initialize_default_values(self) -> None:
        """Initialize parameters with default values."""
        for param_name, param_def in self.parameter_definitions.items():
            self.current_values[param_name] = param_def.default_value

        self.logger.info(
            f"Initialized {len(self.current_values)} parameters with default values"
        )

    def _save_current_values(self) -> None:
        """Save current parameter values to file."""
        try:
            # Create config directory if it doesn't exist
            config_dir = results_manager.get_path("config")
            config_dir.mkdir(parents=True, exist_ok=True)

            # Save to file
            config_file = config_dir / "digital_twin_parameters.json"
            with open(config_file, "w") as f:
                json.dump(self.current_values, f, indent=2)

            self.logger.info(f"Saved configuration to {config_file}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")

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
        if param_name not in self.parameter_definitions:
            raise KeyError(f"Unknown parameter: {param_name}")

        # Return current value or default if not set
        if param_name in self.current_values:
            return self.current_values[param_name]

        # Use default value if not set
        return self.parameter_definitions[param_name].default_value

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
        if param_name not in self.parameter_definitions:
            raise KeyError(f"Unknown parameter: {param_name}")

        # Get parameter definition
        param_def = self.parameter_definitions[param_name]

        # Validate value
        is_valid, error_message = param_def.validate(value)
        if not is_valid:
            return False, error_message

        # Get old value
        old_value = self.get_parameter_value(param_name)

        # Only update if value has changed
        if old_value != value:
            # Update value
            self.current_values[param_name] = value

            # Save configuration
            self._save_current_values()

            # Publish event
            self.publisher.publish_parameter_change(
                parameter_name=param_name,
                old_value=old_value,
                new_value=value,
                parameter_path=param_def.path,
                requires_restart=param_def.requires_restart,
            )

            self.logger.info(
                f"Parameter '{param_name}' changed from {old_value} to {value}"
            )

        return True, None

    def get_parameter_definition(self, param_name: str) -> ParameterDefinition:
        """
        Get definition of a parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            ParameterDefinition: Parameter definition

        Raises:
            KeyError: If parameter not found
        """
        if param_name not in self.parameter_definitions:
            raise KeyError(f"Unknown parameter: {param_name}")

        return self.parameter_definitions[param_name]

    def get_all_parameter_values(self) -> Dict[str, Any]:
        """
        Get all current parameter values.

        Returns:
            Dict[str, Any]: Dictionary of parameter names to values
        """
        # Start with current values
        all_values = self.current_values.copy()

        # Add default values for parameters not in current values
        for param_name, param_def in self.parameter_definitions.items():
            if param_name not in all_values:
                all_values[param_name] = param_def.default_value

        return all_values

    def get_parameter_groups(self) -> List[ParameterGroup]:
        """
        Get all parameter groups.

        Returns:
            List[ParameterGroup]: List of parameter groups
        """
        return list(self.parameter_groups.values())

    def get_parameters_by_category(
        self, category: ParameterCategory
    ) -> List[ParameterDefinition]:
        """
        Get parameters by category.

        Args:
            category: Category to filter by

        Returns:
            List[ParameterDefinition]: List of parameters in the category
        """
        params = []

        for group in self.parameter_groups.values():
            if group.category == category:
                params.extend(group.parameters)

        return params

    def get_parameters_by_tag(self, tag: str) -> List[ParameterDefinition]:
        """
        Get parameters by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List[ParameterDefinition]: List of parameters with the tag
        """
        return [
            param_def
            for param_def in self.parameter_definitions.values()
            if tag in param_def.tags
        ]

    def reset_parameter_to_default(self, param_name: str) -> None:
        """
        Reset parameter to its default value.

        Args:
            param_name: Name of the parameter

        Raises:
            KeyError: If parameter not found
        """
        if param_name not in self.parameter_definitions:
            raise KeyError(f"Unknown parameter: {param_name}")

        param_def = self.parameter_definitions[param_name]
        self.set_parameter_value(param_name, param_def.default_value)

    def reset_all_parameters(self) -> None:
        """Reset all parameters to their default values."""
        for param_name, param_def in self.parameter_definitions.items():
            self.current_values[param_name] = param_def.default_value

        # Save configuration
        self._save_current_values()

        self.logger.info("Reset all parameters to default values")

    def apply_parameters_to_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply current parameter values to a state dictionary.

        Args:
            state: State dictionary to update

        Returns:
            Dict[str, Any]: Updated state dictionary
        """
        updated_state = state.copy()

        for param_name, param_def in self.parameter_definitions.items():
            if not param_def.path:
                continue

            # Get current value
            value = self.get_parameter_value(param_name)

            # Parse path and set value in state
            path_parts = param_def.path.split(".")
            current = updated_state

            # Navigate to the correct location in the state
            for i, part in enumerate(path_parts):
                if i == len(path_parts) - 1:
                    # Set the value
                    current[part] = value
                else:
                    # Create intermediate dictionaries if needed
                    if part not in current or not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]

        return updated_state


# Create global instance
config_manager = ConfigurationManager()
