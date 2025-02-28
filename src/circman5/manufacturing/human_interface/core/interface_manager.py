# src/circman5/manufacturing/human_interface/core/interface_manager.py

"""
Human-Machine Interface (HMI) manager for CIRCMAN5.0.

This module acts as the central coordinator for the human interface system,
initializing components, managing user interactions, and coordinating the
various interface elements.
"""

from typing import Dict, Any, List, Optional, Set, Union, Callable
import datetime
import threading
from pathlib import Path

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from .interface_state import interface_state
from ..adapters.event_adapter import EventAdapter


class InterfaceManager:
    """
    Central manager for the Human-Machine Interface system.

    This class initializes and coordinates all HMI components, manages state
    and handles communication between the interface and the underlying systems.

    Attributes:
        state: Reference to interface state singleton
        components: Dictionary of registered interface components
        event_adapter: Adapter for the event notification system
        logger: Logger instance for this class
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(InterfaceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the interface manager."""
        if self._initialized:
            return

        self.logger = setup_logger("interface_manager")
        self.constants = ConstantsService()

        # Initialize components
        self.state = interface_state
        self.components: Dict[str, Any] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}

        # Create event adapter to interface with event notification system
        self.event_adapter = EventAdapter()

        # Thread safety
        self._lock = threading.RLock()

        self._initialized = True
        self.logger.info("Interface Manager initialized")

    def register_component(self, component_id: str, component: Any) -> None:
        """
        Register an interface component.

        Args:
            component_id: Unique identifier for the component
            component: Component instance

        Raises:
            ValueError: If component ID already exists
        """
        with self._lock:
            if component_id in self.components:
                raise ValueError(f"Component ID already registered: {component_id}")

            self.components[component_id] = component
            self.logger.debug(f"Component registered: {component_id}")

    def get_component(self, component_id: str) -> Any:
        """
        Get a registered component by ID.

        Args:
            component_id: Component identifier

        Returns:
            Any: Component instance

        Raises:
            KeyError: If component not found
        """
        if component_id not in self.components:
            raise KeyError(f"Component not found: {component_id}")

        return self.components[component_id]

    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register a handler for interface events.

        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        with self._lock:
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []

            self.event_handlers[event_type].append(handler)
            self.logger.debug(f"Event handler registered for {event_type}")

    def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Trigger an interface event.

        Args:
            event_type: Type of event to trigger
            event_data: Event data
        """
        handlers = self.event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {str(e)}")

    def change_view(self, view_name: str) -> None:
        """
        Change the active interface view.

        Args:
            view_name: Name of view to activate
        """
        self.state.set_active_view(view_name)

        # Trigger view change event
        self.trigger_event("view_changed", {"view_name": view_name})
        self.logger.info(f"View changed to: {view_name}")

    def handle_parameter_selection(self, parameter: str, selected: bool) -> None:
        """
        Handle parameter selection/deselection.

        Args:
            parameter: Parameter name
            selected: Whether parameter is selected
        """
        if selected:
            self.state.add_selected_parameter(parameter)
        else:
            self.state.remove_selected_parameter(parameter)

        # Trigger parameter selection event
        self.trigger_event(
            "parameter_selection_changed",
            {"parameter": parameter, "selected": selected},
        )

    def toggle_panel(self, panel_id: str) -> bool:
        """
        Toggle a panel's expanded state.

        Args:
            panel_id: Panel identifier

        Returns:
            bool: New expanded state
        """
        expanded = self.state.toggle_panel_expanded(panel_id)

        # Trigger panel toggle event
        self.trigger_event(
            "panel_toggled", {"panel_id": panel_id, "expanded": expanded}
        )

        return expanded

    def update_alert_settings(self, filters: Dict[str, Any]) -> None:
        """
        Update alert display settings.

        Args:
            filters: Alert filter settings
        """
        self.state.update_alert_filters(filters)

        # Trigger alert settings event
        self.trigger_event("alert_settings_changed", {"filters": filters})

    def save_custom_view(self, name: str, config: Dict[str, Any]) -> None:
        """
        Save a custom dashboard view configuration.

        Args:
            name: View name
            config: View configuration
        """
        self.state.save_custom_view(name, config)

        # Trigger custom view event
        self.trigger_event("custom_view_saved", {"name": name, "config": config})

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a command from the interface.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        self.logger.debug(f"Handling command: {command}, params: {params}")

        # Trigger command event
        self.trigger_event("command_executed", {"command": command, "params": params})

        # Process standard commands
        if command == "change_view":
            self.change_view(params.get("view_name", "main_dashboard"))
            return {"success": True}

        elif command == "reset_interface":
            self.state.reset_to_defaults()
            self.trigger_event("interface_reset", {})
            return {"success": True}

        elif command == "set_parameter_group":
            self.state.set_parameter_group(params.get("group_name", ""))
            return {"success": True}

        # If no standard command handled it, delegate to registered components
        for component in self.components.values():
            if hasattr(component, "handle_command"):
                try:
                    result = component.handle_command(command, params)
                    if result.get("handled", False):
                        return result
                except Exception as e:
                    self.logger.error(f"Error in component command handler: {str(e)}")

        return {"success": False, "error": "Unknown command"}

    def initialize(self) -> bool:
        """
        Initialize all interface components.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing interface components")

            # Initialize event adapter
            self.event_adapter.initialize()

            # Initialize all registered components
            for component_id, component in self.components.items():
                if hasattr(component, "initialize"):
                    try:
                        component.initialize()
                        self.logger.debug(f"Component initialized: {component_id}")
                    except Exception as e:
                        self.logger.error(
                            f"Error initializing component {component_id}: {str(e)}"
                        )

            return True

        except Exception as e:
            self.logger.error(f"Error initializing interface: {str(e)}")
            return False

    def shutdown(self) -> None:
        """Clean up resources and shut down interface."""
        self.logger.info("Shutting down interface")

        # Shutdown all registered components
        for component_id, component in self.components.items():
            if hasattr(component, "shutdown"):
                try:
                    component.shutdown()
                    self.logger.debug(f"Component shutdown: {component_id}")
                except Exception as e:
                    self.logger.error(
                        f"Error shutting down component {component_id}: {str(e)}"
                    )

        # Shutdown event adapter
        try:
            self.event_adapter.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down event adapter: {str(e)}")


# Create global instance
interface_manager = InterfaceManager()
