# src/circman5/manufacturing/human_interface/components/dashboard/main_dashboard.py

"""
Main dashboard component for CIRCMAN5.0 Human-Machine Interface.

This module implements the primary dashboard layout and functionality,
integrating various panels and visualizations for monitoring the
manufacturing system.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
import datetime
import threading

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.dashboard_manager import dashboard_manager
from ...core.interface_state import interface_state
from ..controls.parameter_control import parameter_control
from ...adapters.event_adapter import EventAdapter


class MainDashboard:
    """
    Main dashboard component for the Human-Machine Interface.

    This class implements the primary dashboard layout and functionality,
    coordinating various panels for system monitoring.

    Attributes:
        state: Reference to interface state
        event_adapter: Event adapter for notifications
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the main dashboard."""
        self.logger = setup_logger("main_dashboard")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.event_adapter = EventAdapter()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        # Register with interface manager and dashboard manager
        interface_manager.register_component("main_dashboard", self)
        dashboard_manager.register_component("main_dashboard", self)

        # Register event handlers
        interface_manager.register_event_handler(
            "panel_toggled", self._on_panel_toggled
        )

        self.logger.info("Main Dashboard initialized")

    def render_dashboard(self) -> Dict[str, Any]:
        """
        Render the main dashboard.

        Returns:
            Dict[str, Any]: Dashboard data
        """
        # Use dashboard manager to render the dashboard
        return dashboard_manager.render_dashboard("main_dashboard")

    def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render a dashboard panel.

        Args:
            config: Panel configuration

        Returns:
            Dict[str, Any]: Panel data
        """
        # This is called by the dashboard manager when it needs to render
        # a panel of type "main_dashboard" (unlikely, but included for consistency)
        return {
            "type": "main_dashboard",
            "timestamp": datetime.datetime.now().isoformat(),
            "panels": self._get_default_panels(),
        }

    def update_layout(self, layout_config: Dict[str, Any]) -> bool:
        """
        Update dashboard layout configuration.

        Args:
            layout_config: New layout configuration

        Returns:
            bool: True if layout was updated
        """
        try:
            # Get existing layout
            layout = dashboard_manager.get_layout("main_dashboard")
            if not layout:
                self.logger.warning("Main dashboard layout not found")
                return False

            # Update layout configuration
            layout.layout_config = layout_config

            # Save the updated layout
            dashboard_manager.update_layout(layout)

            return True

        except Exception as e:
            self.logger.error(f"Error updating layout: {str(e)}")
            return False

    def update_panel_config(self, panel_id: str, config: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific panel.

        Args:
            panel_id: Panel identifier
            config: New panel configuration

        Returns:
            bool: True if panel was updated
        """
        try:
            # Get existing layout
            layout = dashboard_manager.get_layout("main_dashboard")
            if not layout:
                self.logger.warning("Main dashboard layout not found")
                return False

            # Update panel configuration
            if panel_id in layout.panels:
                layout.panels[panel_id].update(config)

                # Save the updated layout
                dashboard_manager.update_layout(layout)

                return True
            else:
                self.logger.warning(f"Panel not found: {panel_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error updating panel: {str(e)}")
            return False

    def add_panel(self, panel_id: str, config: Dict[str, Any]) -> bool:
        """
        Add a new panel to the dashboard.

        Args:
            panel_id: Panel identifier
            config: Panel configuration

        Returns:
            bool: True if panel was added
        """
        try:
            # Get existing layout
            layout = dashboard_manager.get_layout("main_dashboard")
            if not layout:
                self.logger.warning("Main dashboard layout not found")
                return False

            # Add new panel
            layout.panels[panel_id] = config

            # Save the updated layout
            dashboard_manager.update_layout(layout)

            return True

        except Exception as e:
            self.logger.error(f"Error adding panel: {str(e)}")
            return False

    def remove_panel(self, panel_id: str) -> bool:
        """
        Remove a panel from the dashboard.

        Args:
            panel_id: Panel identifier

        Returns:
            bool: True if panel was removed
        """
        try:
            # Get existing layout
            layout = dashboard_manager.get_layout("main_dashboard")
            if not layout:
                self.logger.warning("Main dashboard layout not found")
                return False

            # Remove panel
            if panel_id in layout.panels:
                del layout.panels[panel_id]

                # Save the updated layout
                dashboard_manager.update_layout(layout)

                return True
            else:
                self.logger.warning(f"Panel not found: {panel_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error removing panel: {str(e)}")
            return False

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle dashboard-related commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "render_main_dashboard":
            dashboard_data = self.render_dashboard()
            return {"handled": True, "success": True, "dashboard": dashboard_data}

        elif command == "update_panel":
            panel_id = params.get("panel_id")
            panel_config = params.get("config")

            if not panel_id or not panel_config:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing panel_id or config parameter",
                }

            success = self.update_panel_config(panel_id, panel_config)
            return {"handled": True, "success": success}

        elif command == "add_panel":
            panel_id = params.get("panel_id")
            panel_config = params.get("config")

            if not panel_id or not panel_config:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing panel_id or config parameter",
                }

            success = self.add_panel(panel_id, panel_config)
            return {"handled": True, "success": success}

        elif command == "remove_panel":
            panel_id = params.get("panel_id")

            if not panel_id:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing panel_id parameter",
                }

            success = self.remove_panel(panel_id)
            return {"handled": True, "success": success}

        # Not a main dashboard command
        return {"handled": False}

    def _on_panel_toggled(self, event_data: Dict[str, Any]) -> None:
        """
        Handle panel toggle events.

        Args:
            event_data: Event data
        """
        panel_id = event_data.get("panel_id")
        expanded = event_data.get("expanded", False)

        if panel_id:
            self.logger.debug(f"Panel {panel_id} toggled: expanded={expanded}")
            # Additional handling if needed

    def _get_default_panels(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default panel configurations.

        Returns:
            Dict[str, Dict[str, Any]]: Default panels
        """
        return {
            "status": {
                "type": "status_panel",
                "title": "System Status",
                "position": {"row": 0, "col": 0},
                "size": {"rows": 1, "cols": 1},
                "expanded": self.state.is_panel_expanded("status"),
            },
            "kpi": {
                "type": "kpi_panel",
                "title": "Key Performance Indicators",
                "position": {"row": 0, "col": 1},
                "size": {"rows": 1, "cols": 1},
                "expanded": self.state.is_panel_expanded("kpi"),
                "metrics": ["production_rate", "energy_consumption", "defect_rate"],
            },
            "process": {
                "type": "process_panel",
                "title": "Manufacturing Process",
                "position": {"row": 1, "col": 0},
                "size": {"rows": 1, "cols": 2},
                "expanded": self.state.is_panel_expanded("process"),
            },
        }


# Create global instance
main_dashboard = MainDashboard()
