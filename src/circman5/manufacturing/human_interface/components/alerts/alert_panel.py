# src/circman5/manufacturing/human_interface/components/alerts/alert_panel.py

"""
Alert panel component for CIRCMAN5.0 Human-Machine Interface.

This module implements the alert panel that displays system alerts and
notifications from the event notification system, allowing users to
view, filter, and acknowledge alerts.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
import datetime
import threading
import time

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.dashboard_manager import dashboard_manager
from ...core.interface_state import interface_state
from ...adapters.event_adapter import EventAdapter
from ....digital_twin.event_notification.event_types import (
    Event,
    EventCategory,
    EventSeverity,
)


class AlertPanel:
    """
    Alert panel component for the Human-Machine Interface.

    This panel displays system alerts and notifications from the
    event notification system, allowing users to view, filter, and
    acknowledge alerts.

    Attributes:
        state: Reference to interface state
        event_adapter: Event adapter for notifications
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the alert panel."""
        self.logger = setup_logger("alert_panel")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.event_adapter = EventAdapter()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        # Alert cache and tracking
        self._alerts_cache: List[Dict[str, Any]] = []
        self._new_alerts_count = 0
        self._last_update = datetime.datetime.min
        self._cache_ttl = datetime.timedelta(seconds=10)  # Cache TTL of 10 seconds

        # Register with interface manager and dashboard manager
        interface_manager.register_component("alert_panel", self)
        dashboard_manager.register_component("alert_panel", self)

        # Register for events
        self.event_adapter.register_callback(
            self._on_event, category=None, severity=None  # Register for all categories
        )

        self.logger.info("Alert Panel initialized")

    def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the alert panel.

        Args:
            config: Panel configuration

        Returns:
            Dict[str, Any]: Panel data
        """
        # Get alerts based on current filter settings
        filter_settings = config.get("filter", {})

        # If no filter in config, use filter from interface state
        if not filter_settings:
            filter_settings = self.state.get_alert_filters()

        alerts = self.get_filtered_alerts(filter_settings)

        # Prepare panel data
        panel_data = {
            "type": "alert_panel",
            "title": config.get("title", "System Alerts"),
            "timestamp": datetime.datetime.now().isoformat(),
            "expanded": self.state.is_panel_expanded(config.get("id", "alerts")),
            "alerts": alerts,
            "new_alerts_count": self._new_alerts_count,
            "config": config,
            "filter_settings": filter_settings,
        }

        return panel_data

    def get_filtered_alerts(
        self, filter_settings: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get alerts filtered by settings.

        Args:
            filter_settings: Filter settings

        Returns:
            List[Dict[str, Any]]: Filtered alerts
        """
        # Check if we need to refresh the cache
        now = datetime.datetime.now()
        if now - self._last_update > self._cache_ttl or not self._alerts_cache:
            self._refresh_alerts_cache()

        # Apply filters
        filtered_alerts = self._alerts_cache.copy()

        # Filter by severity levels
        severity_levels = filter_settings.get("severity_levels", [])
        if severity_levels:
            filtered_alerts = [
                alert
                for alert in filtered_alerts
                if alert.get("severity", "").lower()
                in [s.lower() for s in severity_levels]
            ]

        # Filter by categories
        categories = filter_settings.get("categories", [])
        if categories:
            filtered_alerts = [
                alert
                for alert in filtered_alerts
                if alert.get("category", "").lower() in [c.lower() for c in categories]
            ]

        # Filter by acknowledgment status
        show_acknowledged = filter_settings.get("show_acknowledged", False)
        if not show_acknowledged:
            filtered_alerts = [
                alert
                for alert in filtered_alerts
                if not alert.get("acknowledged", False)
            ]

        return filtered_alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert/Event ID to acknowledge

        Returns:
            bool: True if alert was acknowledged
        """
        try:
            # Acknowledge in event system
            success = self.event_adapter.acknowledge_event(alert_id)

            if success:
                # Update cache
                with self._lock:
                    for alert in self._alerts_cache:
                        if alert.get("id") == alert_id:
                            alert["acknowledged"] = True

                self.logger.debug(f"Alert acknowledged: {alert_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {str(e)}")
            return False

    def acknowledge_all_visible(self, filter_settings: Dict[str, Any]) -> int:
        """
        Acknowledge all visible alerts based on current filters.

        Args:
            filter_settings: Filter settings

        Returns:
            int: Number of alerts acknowledged
        """
        try:
            # Get visible alerts
            visible_alerts = self.get_filtered_alerts(filter_settings)

            # Acknowledge each alert
            count = 0
            for alert in visible_alerts:
                alert_id = alert.get("id")
                if alert_id and not alert.get("acknowledged", False):
                    if self.acknowledge_alert(alert_id):
                        count += 1

            self.logger.info(f"Acknowledged {count} alerts")
            return count

        except Exception as e:
            self.logger.error(f"Error acknowledging all alerts: {str(e)}")
            return 0

    def update_filter_settings(self, filter_settings: Dict[str, Any]) -> None:
        """
        Update alert filter settings.

        Args:
            filter_settings: New filter settings
        """
        self.state.update_alert_filters(filter_settings)
        self.logger.debug(f"Updated alert filter settings: {filter_settings}")

    def _refresh_alerts_cache(self) -> None:
        """Refresh the alerts cache from the event notification system."""
        try:
            # Get recent events
            events = self.event_adapter.get_recent_events(limit=100)

            # Convert events to alert format
            alerts = []
            for event in events:
                alerts.append(self._convert_event_to_alert(event))

            # Update cache
            with self._lock:
                # Check for new alerts
                if self._alerts_cache:
                    existing_ids = {alert["id"] for alert in self._alerts_cache}
                    self._new_alerts_count = sum(
                        1 for alert in alerts if alert["id"] not in existing_ids
                    )
                else:
                    self._new_alerts_count = len(alerts)

                self._alerts_cache = alerts
                self._last_update = datetime.datetime.now()

            self.logger.debug(f"Refreshed alerts cache: {len(alerts)} alerts")

        except Exception as e:
            self.logger.error(f"Error refreshing alerts cache: {str(e)}")

    def _convert_event_to_alert(self, event: Any) -> Dict[str, Any]:
        """
        Convert an event to alert format.

        Args:
            event: Event object

        Returns:
            Dict[str, Any]: Alert data
        """
        # Check if event has required attributes
        if not hasattr(event, "event_id") or not hasattr(event, "category"):
            # Create a basic alert from dictionary if possible
            if isinstance(event, dict):
                return {
                    "id": event.get("event_id", str(time.time())),
                    "timestamp": event.get(
                        "timestamp", datetime.datetime.now().isoformat()
                    ),
                    "category": event.get("category", "unknown"),
                    "severity": event.get("severity", "info"),
                    "message": event.get("message", "Unknown event"),
                    "source": event.get("source", "system"),
                    "acknowledged": event.get("acknowledged", False),
                    "details": event.get("details", {}),
                }

            # Create a fallback alert
            return {
                "id": str(time.time()),
                "timestamp": datetime.datetime.now().isoformat(),
                "category": "unknown",
                "severity": "info",
                "message": "Unknown event format",
                "source": "system",
                "acknowledged": False,
                "details": {},
            }

        # Convert event attributes to alert format
        alert = {
            "id": event.event_id,
            "timestamp": event.timestamp,
            "category": event.category.value
            if hasattr(event.category, "value")
            else str(event.category),
            "severity": event.severity.value
            if hasattr(event.severity, "value")
            else str(event.severity),
            "message": event.message,
            "source": event.source,
            "acknowledged": event.acknowledged,
            "details": event.details,
        }

        return alert

    def _on_event(self, event: Any) -> None:
        """
        Handle incoming events.

        Args:
            event: Event data
        """
        # Invalidate cache to include new event
        self._last_update = datetime.datetime.min

        # Increment new alerts count
        with self._lock:
            self._new_alerts_count += 1

    def reset_new_alerts_count(self) -> None:
        """Reset the new alerts count."""
        with self._lock:
            self._new_alerts_count = 0

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle alert panel commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "get_alerts":
            filter_settings = params.get("filter", {})
            alerts = self.get_filtered_alerts(filter_settings)
            return {
                "handled": True,
                "success": True,
                "alerts": alerts,
                "new_alerts_count": self._new_alerts_count,
            }

        elif command == "acknowledge_alert":
            alert_id = params.get("alert_id")
            if alert_id:
                success = self.acknowledge_alert(alert_id)
                return {"handled": True, "success": success}
            return {
                "handled": True,
                "success": False,
                "error": "Missing alert_id parameter",
            }

        elif command == "acknowledge_all":
            filter_settings = params.get("filter", {})
            count = self.acknowledge_all_visible(filter_settings)
            return {"handled": True, "success": True, "count": count}

        elif command == "update_filter":
            filter_settings = params.get("filter", {})
            if filter_settings:
                self.update_filter_settings(filter_settings)
                return {"handled": True, "success": True}
            return {
                "handled": True,
                "success": False,
                "error": "Missing filter parameter",
            }

        elif command == "refresh_alerts":
            self._refresh_alerts_cache()
            return {"handled": True, "success": True}

        elif command == "reset_new_count":
            self.reset_new_alerts_count()
            return {"handled": True, "success": True}

        # Not an alert panel command
        return {"handled": False}


# Create global instance
alert_panel = AlertPanel()
