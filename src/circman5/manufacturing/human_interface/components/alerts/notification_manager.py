# src/circman5/manufacturing/human_interface/components/alerts/notification_manager.py

"""
Notification manager for CIRCMAN5.0 Human-Machine Interface.

This module implements the notification management system, handling alert
prioritization, display, and user notification for system events.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable
import datetime
import threading
import time
import queue

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.interface_state import interface_state
from ...adapters.event_adapter import EventAdapter
from ....digital_twin.event_notification.event_types import (
    Event,
    EventCategory,
    EventSeverity,
)


class NotificationManager:
    """
    Notification manager for the Human-Machine Interface.

    This class manages user notifications for system events, including
    alert prioritization, display handling, and notification clearing.

    Attributes:
        state: Reference to interface state
        event_adapter: Event adapter for notifications
        notification_queue: Queue of pending notifications
        active_notifications: Dictionary of active notifications
        notification_callbacks: Registered notification callbacks
        logger: Logger instance for this class
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(NotificationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the notification manager."""
        if self._initialized:
            return

        self.logger = setup_logger("notification_manager")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.event_adapter = EventAdapter()  # Get instance

        # Notification tracking
        self.notification_queue = queue.Queue()
        self.active_notifications: Dict[str, Dict[str, Any]] = {}
        self.notification_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Thread management
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._notification_thread: Optional[threading.Thread] = None

        # Register with interface manager
        interface_manager.register_component("notification_manager", self)

        # Register for events
        self.event_adapter.register_callback(
            self._on_event, category=None, severity=None  # Register for all categories
        )

        self._initialized = True
        self.logger.info("Notification Manager initialized")

        # Start notification thread
        self._start_notification_thread()

    def _start_notification_thread(self) -> None:
        """Start the notification processing thread."""
        self._stop_event.clear()
        self._notification_thread = threading.Thread(
            target=self._process_notifications, daemon=True
        )
        self._notification_thread.start()
        self.logger.debug("Notification thread started")

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for notification events.

        Args:
            callback: Callback function to receive notifications
        """
        with self._lock:
            self.notification_callbacks.append(callback)
            self.logger.debug("Notification callback registered")

    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister a callback.

        Args:
            callback: Callback to unregister
        """
        with self._lock:
            if callback in self.notification_callbacks:
                self.notification_callbacks.remove(callback)
                self.logger.debug("Notification callback unregistered")

    def add_notification(self, notification: Dict[str, Any]) -> str:
        """
        Add a notification to the queue.

        Args:
            notification: Notification data

        Returns:
            str: Notification ID
        """
        # Generate ID if not provided
        if "id" not in notification:
            notification["id"] = str(time.time())

        # Add timestamp if not provided
        if "timestamp" not in notification:
            notification["timestamp"] = datetime.datetime.now().isoformat()

        # Default priority
        if "priority" not in notification:
            notification["priority"] = self._get_priority_from_severity(
                notification.get("severity", "info")
            )

        # Add to queue
        self.notification_queue.put(notification)
        self.logger.debug(f"Notification added to queue: {notification['id']}")

        return notification["id"]

    def remove_notification(self, notification_id: str) -> bool:
        """
        Remove a notification.

        Args:
            notification_id: Notification ID to remove

        Returns:
            bool: True if notification was removed
        """
        with self._lock:
            if notification_id in self.active_notifications:
                del self.active_notifications[notification_id]
                self.logger.debug(f"Notification removed: {notification_id}")
                return True

        return False

    def clear_all_notifications(self) -> int:
        """
        Clear all active notifications.

        Returns:
            int: Number of notifications cleared
        """
        with self._lock:
            count = len(self.active_notifications)
            self.active_notifications.clear()
            self.logger.info(f"Cleared {count} notifications")
            return count

    def get_active_notifications(self) -> List[Dict[str, Any]]:
        """
        Get all active notifications.

        Returns:
            List[Dict[str, Any]]: List of active notifications
        """
        with self._lock:
            # Convert to list and sort by priority (higher first)
            notifications = list(self.active_notifications.values())
            notifications.sort(key=lambda n: n.get("priority", 0), reverse=True)
            return notifications

    def _process_notifications(self) -> None:
        """Process notifications from the queue in a background thread."""
        self.logger.debug("Notification processing thread started")

        while not self._stop_event.is_set():
            try:
                # Get notification from queue with timeout to allow checking stop flag
                try:
                    notification = self.notification_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Process the notification
                self._handle_notification(notification)

                # Mark task as done
                self.notification_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing notification: {str(e)}")

        self.logger.debug("Notification processing thread stopped")

    def _handle_notification(self, notification: Dict[str, Any]) -> None:
        """
        Handle a notification.

        Args:
            notification: Notification data
        """
        notification_id = notification["id"]

        # Add to active notifications
        with self._lock:
            self.active_notifications[notification_id] = notification

        # Dispatch to registered callbacks
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Error in notification callback: {str(e)}")

    def _on_event(self, event: Any) -> None:
        """
        Handle incoming events.

        Args:
            event: Event data
        """
        # Check if event is of high enough severity to generate a notification
        if not self._should_notify_for_event(event):
            return

        # Convert event to notification format
        notification = self._convert_event_to_notification(event)

        # Add to queue
        self.add_notification(notification)

    def _should_notify_for_event(self, event: Any) -> bool:
        """
        Determine if an event should generate a notification.

        Args:
            event: Event to check

        Returns:
            bool: True if notification should be created
        """
        # Check if event is of a severe enough type
        if hasattr(event, "severity"):
            severity = event.severity

            # Check severity against notification thresholds
            if severity == EventSeverity.CRITICAL or severity == EventSeverity.ERROR:
                return True

            if severity == EventSeverity.WARNING:
                # Only warn for specific categories
                if hasattr(event, "category"):
                    if event.category in [
                        EventCategory.THRESHOLD,
                        EventCategory.SYSTEM,
                    ]:
                        return True

        return False

    def _convert_event_to_notification(self, event: Any) -> Dict[str, Any]:
        """
        Convert an event to notification format.

        Args:
            event: Event object

        Returns:
            Dict[str, Any]: Notification data
        """
        # Check if event has required attributes
        if not hasattr(event, "event_id") or not hasattr(event, "category"):
            # Create a basic notification from dictionary if possible
            if isinstance(event, dict):
                return {
                    "id": event.get("event_id", str(time.time())),
                    "timestamp": event.get(
                        "timestamp", datetime.datetime.now().isoformat()
                    ),
                    "title": "System Notification",
                    "message": event.get("message", "Unknown event"),
                    "severity": event.get("severity", "info"),
                    "priority": self._get_priority_from_severity(
                        event.get("severity", "info")
                    ),
                    "source": event.get("source", "system"),
                    "details": event.get("details", {}),
                    "acknowledged": False,
                }

            # Create a fallback notification
            return {
                "id": str(time.time()),
                "timestamp": datetime.datetime.now().isoformat(),
                "title": "System Notification",
                "message": "Unknown event format",
                "severity": "info",
                "priority": 0,
                "source": "system",
                "details": {},
                "acknowledged": False,
            }

        # Convert event attributes to notification format
        severity = (
            event.severity.value
            if hasattr(event.severity, "value")
            else str(event.severity)
        )

        return {
            "id": event.event_id,
            "timestamp": event.timestamp,
            "title": self._get_title_for_category(event.category),
            "message": event.message,
            "severity": severity,
            "priority": self._get_priority_from_severity(severity),
            "source": event.source,
            "details": event.details,
            "acknowledged": event.acknowledged,
        }

    def _get_title_for_category(self, category: Any) -> str:
        """
        Get notification title based on event category.

        Args:
            category: Event category

        Returns:
            str: Title string
        """
        category_str = category.value if hasattr(category, "value") else str(category)

        titles = {
            "system": "System Notification",
            "process": "Process Notification",
            "optimization": "Optimization Alert",
            "threshold": "Threshold Alert",
            "user": "User Action",
            "error": "System Error",
        }

        return titles.get(category_str.lower(), "Notification")

    def _get_priority_from_severity(self, severity: str) -> int:
        """
        Get notification priority from severity level.

        Args:
            severity: Severity string

        Returns:
            int: Priority level (higher number = higher priority)
        """
        priorities = {"critical": 100, "error": 75, "warning": 50, "info": 25}

        return priorities.get(severity.lower(), 0)

    def shutdown(self) -> None:
        """Shutdown the notification manager."""
        self._stop_event.set()

        if self._notification_thread and self._notification_thread.is_alive():
            self._notification_thread.join(timeout=2.0)

        self.logger.info("Notification Manager shutdown")

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle notification manager commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "get_notifications":
            notifications = self.get_active_notifications()
            return {"handled": True, "success": True, "notifications": notifications}

        elif command == "add_notification":
            notification_data = params.get("notification")
            if not notification_data:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing notification parameter",
                }

            notification_id = self.add_notification(notification_data)
            return {
                "handled": True,
                "success": True,
                "notification_id": notification_id,
            }

        elif command == "remove_notification":
            notification_id = params.get("notification_id")
            if not notification_id:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing notification_id parameter",
                }

            success = self.remove_notification(notification_id)
            return {"handled": True, "success": success}

        elif command == "clear_notifications":
            count = self.clear_all_notifications()
            return {"handled": True, "success": True, "count": count}

        # Not a notification manager command
        return {"handled": False}


# Create global instance
notification_manager = NotificationManager()
