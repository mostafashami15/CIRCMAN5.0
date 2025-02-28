# src/circman5/manufacturing/human_interface/components/dashboard/status_panel.py

"""
Status panel component for CIRCMAN5.0 Human-Machine Interface.

This module implements the status panel that displays the current system status,
including production line state, active processes, and operational metrics.
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
from ....digital_twin.core.twin_core import DigitalTwin


class StatusPanel:
    """
    Status panel component for the Human-Machine Interface.

    This panel displays the current system status, production line state,
    active processes, and operational metrics.

    Attributes:
        state: Reference to interface state
        digital_twin: Reference to digital twin
        event_adapter: Event adapter for notifications
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the status panel."""
        self.logger = setup_logger("status_panel")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.digital_twin = DigitalTwin()  # Get instance
        self.event_adapter = EventAdapter()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        # Cache for status data
        self._status_cache: Dict[str, Any] = {}
        self._last_update = datetime.datetime.min
        self._cache_ttl = datetime.timedelta(seconds=1)  # Cache TTL of 1 second

        # Register with interface manager and dashboard manager
        interface_manager.register_component("status_panel", self)
        dashboard_manager.register_component("status_panel", self)

        # Register event handlers for system state changes
        self.event_adapter.register_callback(
            self._on_system_state_change,
            category=None,  # Will be registered for all categories
            severity=None,
        )

        self.logger.info("Status Panel initialized")

    def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the status panel.

        Args:
            config: Panel configuration

        Returns:
            Dict[str, Any]: Panel data
        """
        # Get system status
        system_status = self._get_system_status()

        # Prepare panel data
        panel_data = {
            "type": "status_panel",
            "title": config.get("title", "System Status"),
            "timestamp": datetime.datetime.now().isoformat(),
            "expanded": self.state.is_panel_expanded(config.get("id", "status")),
            "system_status": system_status,
            "config": config,
        }

        return panel_data

    def _get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Dict[str, Any]: System status information
        """
        # Check if cached data is still valid
        now = datetime.datetime.now()
        if now - self._last_update < self._cache_ttl and self._status_cache:
            return self._status_cache.copy()

        # Get current state from digital twin
        try:
            current_state = self.digital_twin.get_current_state()

            # Extract status information
            status_data = {
                "timestamp": current_state.get("timestamp", now.isoformat()),
                "system_status": current_state.get("system_status", "unknown"),
                "production_line": {},
            }

            # Extract production line status if available
            if "production_line" in current_state:
                prod_line = current_state["production_line"]
                status_data["production_line"] = {
                    "status": prod_line.get("status", "unknown"),
                    "temperature": prod_line.get("temperature", 0.0),
                    "energy_consumption": prod_line.get("energy_consumption", 0.0),
                    "production_rate": prod_line.get("production_rate", 0.0),
                }

            # Extract material information if available
            if "materials" in current_state:
                materials = current_state["materials"]
                status_data["materials"] = {
                    name: {
                        "inventory": material.get("inventory", 0.0),
                        "quality": material.get("quality", 0.0),
                    }
                    for name, material in materials.items()
                    if isinstance(material, dict)
                }

            # Extract environment information if available
            if "environment" in current_state:
                status_data["environment"] = current_state["environment"]

            # Update cache
            self._status_cache = status_data
            self._last_update = now

            return status_data

        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")

            # Return basic error status
            return {
                "timestamp": now.isoformat(),
                "system_status": "error",
                "error": str(e),
            }

    def _on_system_state_change(self, event: Any) -> None:
        """
        Handle system state change events.

        Args:
            event: Event data
        """
        # Invalidate cache
        self._last_update = datetime.datetime.min

        # Check if event is a system state change
        if hasattr(event, "category") and event.category == "SYSTEM":
            self.logger.debug("System state change detected, invalidating status cache")

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle status panel commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "get_system_status":
            status_data = self._get_system_status()
            return {"handled": True, "success": True, "status": status_data}

        elif command == "refresh_status":
            # Invalidate cache
            self._last_update = datetime.datetime.min
            status_data = self._get_system_status()
            return {"handled": True, "success": True, "status": status_data}

        # Not a status panel command
        return {"handled": False}


# Create global instance
status_panel = StatusPanel()
