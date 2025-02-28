# src/circman5/manufacturing/human_interface/core/interface_state.py

"""
Human-Machine Interface (HMI) state management module for CIRCMAN5.0.

This module handles the state of the human interface, tracking the active views,
selected parameters, alerts, and other UI-specific state information.
"""

from typing import Dict, Any, List, Optional, Set, Union
import datetime
import threading
import copy
import json
from pathlib import Path

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService


class InterfaceState:
    """
    Manages the state of the Human-Machine Interface.

    This class maintains the UI state, including which components are active,
    what parameters are selected, and what alerts are being displayed.

    Attributes:
        active_view: Currently active dashboard view
        selected_parameters: Set of parameters currently selected for monitoring
        expanded_panels: Set of panels currently expanded
        alert_filters: Filtering criteria for alerts
        custom_views: User-defined custom views
        logger: Logger instance for this class
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(InterfaceState, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the interface state."""
        if self._initialized:
            return

        self.logger = setup_logger("interface_state")
        self.constants = ConstantsService()

        # Initialize state containers
        self.active_view = "main_dashboard"
        self.selected_parameters: Set[str] = set()
        self.expanded_panels: Set[str] = set(["status", "kpi"])
        self.alert_filters: Dict[str, Any] = {
            "severity_levels": ["critical", "error", "warning", "info"],
            "categories": ["system", "process", "user"],
            "show_acknowledged": False,
        }
        self.custom_views: Dict[str, Dict[str, Any]] = {}

        # For parameter control
        self.selected_parameter_group = ""
        self.parameter_edit_mode = False

        # For process control
        self.selected_process = ""
        self.process_control_mode = "monitor"  # Options: monitor, manual, automatic

        # Thread safety
        self._lock = threading.RLock()

        # Load state from persistence if available
        self._load_state()

        self._initialized = True
        self.logger.info("Interface State initialized")

    def set_active_view(self, view_name: str) -> None:
        """
        Set the active view in the interface.

        Args:
            view_name: Name of the view to set as active
        """
        with self._lock:
            self.active_view = view_name
            self.logger.debug(f"Active view set to: {view_name}")
            self._save_state()

    def get_active_view(self) -> str:
        """
        Get the currently active view.

        Returns:
            str: Name of the active view
        """
        return self.active_view

    def add_selected_parameter(self, parameter: str) -> None:
        """
        Add a parameter to the selected parameters set.

        Args:
            parameter: Parameter to select
        """
        with self._lock:
            self.selected_parameters.add(parameter)
            self.logger.debug(f"Parameter selected: {parameter}")
            self._save_state()

    def remove_selected_parameter(self, parameter: str) -> None:
        """
        Remove a parameter from the selected parameters set.

        Args:
            parameter: Parameter to deselect
        """
        with self._lock:
            if parameter in self.selected_parameters:
                self.selected_parameters.remove(parameter)
                self.logger.debug(f"Parameter deselected: {parameter}")
                self._save_state()

    def get_selected_parameters(self) -> Set[str]:
        """
        Get the set of currently selected parameters.

        Returns:
            Set[str]: Set of selected parameter names
        """
        return self.selected_parameters.copy()

    def toggle_panel_expanded(self, panel_id: str) -> bool:
        """
        Toggle the expanded state of a panel.

        Args:
            panel_id: ID of the panel to toggle

        Returns:
            bool: New expanded state (True if expanded)
        """
        with self._lock:
            if panel_id in self.expanded_panels:
                self.expanded_panels.remove(panel_id)
                self.logger.debug(f"Panel collapsed: {panel_id}")
                expanded = False
            else:
                self.expanded_panels.add(panel_id)
                self.logger.debug(f"Panel expanded: {panel_id}")
                expanded = True

            self._save_state()
            return expanded

    def is_panel_expanded(self, panel_id: str) -> bool:
        """
        Check if a panel is expanded.

        Args:
            panel_id: ID of the panel to check

        Returns:
            bool: True if panel is expanded
        """
        return panel_id in self.expanded_panels

    def update_alert_filters(self, filters: Dict[str, Any]) -> None:
        """
        Update alert filtering criteria.

        Args:
            filters: Dictionary of filter criteria
        """
        with self._lock:
            self.alert_filters.update(filters)
            self.logger.debug(f"Alert filters updated: {filters}")
            self._save_state()

    def get_alert_filters(self) -> Dict[str, Any]:
        """
        Get current alert filters.

        Returns:
            Dict[str, Any]: Current alert filters
        """
        return self.alert_filters.copy()

    def save_custom_view(self, view_name: str, view_config: Dict[str, Any]) -> None:
        """
        Save a custom view configuration.

        Args:
            view_name: Name of the custom view
            view_config: View configuration
        """
        with self._lock:
            self.custom_views[view_name] = view_config
            self.logger.info(f"Custom view saved: {view_name}")
            self._save_state()

    def delete_custom_view(self, view_name: str) -> bool:
        """
        Delete a custom view.

        Args:
            view_name: Name of the custom view to delete

        Returns:
            bool: True if view was deleted
        """
        with self._lock:
            if view_name in self.custom_views:
                del self.custom_views[view_name]
                self.logger.info(f"Custom view deleted: {view_name}")
                self._save_state()
                return True
            return False

    def get_custom_view(self, view_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a custom view configuration.

        Args:
            view_name: Name of the custom view

        Returns:
            Optional[Dict[str, Any]]: View configuration if exists
        """
        return self.custom_views.get(view_name)

    def get_all_custom_views(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all custom views.

        Returns:
            Dict[str, Dict[str, Any]]: All custom views
        """
        return self.custom_views.copy()

    def set_parameter_group(self, group_name: str) -> None:
        """
        Set the selected parameter group.

        Args:
            group_name: Name of the parameter group
        """
        with self._lock:
            self.selected_parameter_group = group_name
            self.logger.debug(f"Selected parameter group: {group_name}")
            self._save_state()

    def get_parameter_group(self) -> str:
        """
        Get the selected parameter group.

        Returns:
            str: Name of the selected parameter group
        """
        return self.selected_parameter_group

    def set_parameter_edit_mode(self, edit_mode: bool) -> None:
        """
        Set parameter edit mode.

        Args:
            edit_mode: Whether edit mode is enabled
        """
        with self._lock:
            self.parameter_edit_mode = edit_mode
            self.logger.debug(f"Parameter edit mode: {edit_mode}")
            self._save_state()

    def is_parameter_edit_mode(self) -> bool:
        """
        Check if parameter edit mode is enabled.

        Returns:
            bool: True if edit mode is enabled
        """
        return self.parameter_edit_mode

    def set_selected_process(self, process_name: str) -> None:
        """
        Set the selected process.

        Args:
            process_name: Name of the process
        """
        with self._lock:
            self.selected_process = process_name
            self.logger.debug(f"Selected process: {process_name}")
            self._save_state()

    def get_selected_process(self) -> str:
        """
        Get the selected process.

        Returns:
            str: Name of the selected process
        """
        return self.selected_process

    def set_process_control_mode(self, mode: str) -> None:
        """
        Set the process control mode.

        Args:
            mode: Control mode (monitor, manual, automatic)

        Raises:
            ValueError: If invalid mode
        """
        if mode not in ["monitor", "manual", "automatic"]:
            raise ValueError(f"Invalid process control mode: {mode}")

        with self._lock:
            self.process_control_mode = mode
            self.logger.debug(f"Process control mode: {mode}")
            self._save_state()

    def get_process_control_mode(self) -> str:
        """
        Get the process control mode.

        Returns:
            str: Current process control mode
        """
        return self.process_control_mode

    def reset_to_defaults(self) -> None:
        """Reset interface state to default values."""
        with self._lock:
            self.active_view = "main_dashboard"
            self.selected_parameters = set()
            self.expanded_panels = set(["status", "kpi"])
            self.alert_filters = {
                "severity_levels": ["critical", "error", "warning", "info"],
                "categories": ["system", "process", "user"],
                "show_acknowledged": False,
            }
            self.custom_views = {}
            self.selected_parameter_group = ""
            self.parameter_edit_mode = False
            self.selected_process = ""
            self.process_control_mode = "monitor"

            self.logger.info("Interface state reset to defaults")
            self._save_state()

    def _save_state(self) -> None:
        """Save interface state to persistent storage."""
        try:
            # Convert state to serializable format
            state_dict = {
                "active_view": self.active_view,
                "selected_parameters": list(self.selected_parameters),
                "expanded_panels": list(self.expanded_panels),
                "alert_filters": self.alert_filters,
                "custom_views": self.custom_views,
                "selected_parameter_group": self.selected_parameter_group,
                "parameter_edit_mode": self.parameter_edit_mode,
                "selected_process": self.selected_process,
                "process_control_mode": self.process_control_mode,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            # Get the interface directory
            interface_dir = results_manager.get_path("digital_twin")
            state_file = interface_dir / "interface_state.json"

            # Save state to file
            with open(state_file, "w") as f:
                json.dump(state_dict, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving interface state: {str(e)}")

    def _load_state(self) -> None:
        """Load interface state from persistent storage."""
        try:
            interface_dir = results_manager.get_path("digital_twin")
            state_file = interface_dir / "interface_state.json"

            if state_file.exists():
                with open(state_file, "r") as f:
                    state_dict = json.load(f)

                # Update state attributes
                self.active_view = state_dict.get("active_view", "main_dashboard")
                self.selected_parameters = set(
                    state_dict.get("selected_parameters", [])
                )
                self.expanded_panels = set(
                    state_dict.get("expanded_panels", ["status", "kpi"])
                )
                self.alert_filters = state_dict.get(
                    "alert_filters",
                    {
                        "severity_levels": ["critical", "error", "warning", "info"],
                        "categories": ["system", "process", "user"],
                        "show_acknowledged": False,
                    },
                )
                self.custom_views = state_dict.get("custom_views", {})
                self.selected_parameter_group = state_dict.get(
                    "selected_parameter_group", ""
                )
                self.parameter_edit_mode = state_dict.get("parameter_edit_mode", False)
                self.selected_process = state_dict.get("selected_process", "")
                self.process_control_mode = state_dict.get(
                    "process_control_mode", "monitor"
                )

                self.logger.info("Interface state loaded from persistent storage")
        except Exception as e:
            self.logger.warning(
                f"Could not load interface state, using defaults: {str(e)}"
            )


# Create global instance
interface_state = InterfaceState()
