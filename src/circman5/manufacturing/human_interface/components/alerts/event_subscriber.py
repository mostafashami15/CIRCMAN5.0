# src/circman5/manufacturing/human_interface/components/alerts/event_subscriber.py

"""
Event subscriber component for CIRCMAN5.0 Human-Machine Interface.

This module implements a subscriber that listens to the event notification system
and forwards events to the appropriate interface components.
"""

from typing import Dict, Any, List, Optional, Callable, Set
import datetime
import threading
import queue

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.interface_state import interface_state
from ....digital_twin.event_notification.event_types import (
    Event,
    EventCategory,
    EventSeverity,
)
from ....digital_twin.event_notification.subscribers import Subscriber


class InterfaceEventSubscriber(Subscriber):
    """
    Subscriber for forwarding events to the interface.

    This class subscribes to the event notification system and
    forwards events to registered callbacks based on event category
    and severity.

    Attributes:
        callbacks: Dictionary of registered callbacks by category
        severity_callbacks: Dictionary of registered callbacks by severity
        logger: Logger instance for this class
    """

    def __init__(self, name: str = "interface_subscriber"):
        """
        Initialize interface event subscriber.

        Args:
            name: Subscriber name
        """
        super().__init__(name=name)

        # Initialize callback dictionaries
        self.callbacks: Dict[str, List[Callable[[Event], None]]] = {}
        self.severity_callbacks: Dict[str, List[Callable[[Event], None]]] = {}

        # Initialize event queue for asynchronous processing
        self.event_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._processing_thread = None
        self.is_processing = False

        self.logger.info("Interface Event Subscriber initialized")

    def start_processing(self) -> None:
        """Start asynchronous event processing."""
        if self.is_processing:
            self.logger.warning("Event processing already running")
            return

        # Clear stop event
        self._stop_event.clear()

        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._process_events, daemon=True
        )
        self._processing_thread.start()
        self.is_processing = True

        self.logger.info("Event processing started")

    def stop_processing(self) -> None:
        """Stop asynchronous event processing."""
        if not self.is_processing:
            self.logger.warning("Event processing not running")
            return

        # Set stop event
        self._stop_event.set()

        # Wait for thread to terminate
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        self.is_processing = False
        self.logger.info("Event processing stopped")

    def register_callback(
        self,
        callback: Callable[[Event], None],
        category: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> None:
        """
        Register a callback for events.

        Args:
            callback: Callback function to be called with events
            category: Optional event category to filter for
            severity: Optional event severity to filter for
        """
        if category:
            # Register for specific category
            if category not in self.callbacks:
                self.callbacks[category] = []

            self.callbacks[category].append(callback)
            self.logger.debug(f"Callback registered for category: {category}")
        elif severity:
            # Register for specific severity
            if severity not in self.severity_callbacks:
                self.severity_callbacks[severity] = []

            self.severity_callbacks[severity].append(callback)
            self.logger.debug(f"Callback registered for severity: {severity}")
        else:
            # Register for all categories
            for cat in EventCategory:
                cat_value = cat.value
                if cat_value not in self.callbacks:
                    self.callbacks[cat_value] = []

                self.callbacks[cat_value].append(callback)

            self.logger.debug("Callback registered for all categories")

    def unregister_callback(
        self,
        callback: Callable[[Event], None],
        category: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> None:
        """
        Unregister a callback.

        Args:
            callback: Callback function to unregister
            category: Optional event category to unregister from
            severity: Optional event severity to unregister from
        """
        if category:
            # Unregister from specific category
            if category in self.callbacks and callback in self.callbacks[category]:
                self.callbacks[category].remove(callback)
                self.logger.debug(f"Callback unregistered from category: {category}")
        elif severity:
            # Unregister from specific severity
            if (
                severity in self.severity_callbacks
                and callback in self.severity_callbacks[severity]
            ):
                self.severity_callbacks[severity].remove(callback)
                self.logger.debug(f"Callback unregistered from severity: {severity}")
        else:
            # Unregister from all categories
            for cat_value in self.callbacks:
                if callback in self.callbacks[cat_value]:
                    self.callbacks[cat_value].remove(callback)

            # Also unregister from all severities
            for sev_value in self.severity_callbacks:
                if callback in self.severity_callbacks[sev_value]:
                    self.severity_callbacks[sev_value].remove(callback)

            self.logger.debug(
                "Callback unregistered from all categories and severities"
            )

    def handle_event(self, event: Event) -> None:
        """
        Handle an event from the event notification system.

        Args:
            event: Event to handle
        """
        # Add event to processing queue for asynchronous processing
        self.event_queue.put(event)

    def _process_events(self) -> None:
        """Process events from the queue in a background thread."""
        self.logger.debug("Event processing thread started")

        while not self._stop_event.is_set():
            try:
                # Get event from queue with timeout to allow checking stop flag
                try:
                    event = self.event_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Forward event to appropriate callbacks
                self._dispatch_event(event)

                # Mark task as done
                self.event_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing event: {str(e)}")

        self.logger.debug("Event processing thread stopped")

    def _dispatch_event(self, event: Event) -> None:
        """
        Dispatch an event to registered callbacks.

        Args:
            event: Event to dispatch
        """
        # Get event category and severity
        category = (
            event.category.value
            if hasattr(event.category, "value")
            else str(event.category)
        )
        severity = (
            event.severity.value
            if hasattr(event.severity, "value")
            else str(event.severity)
        )

        # Dispatch to category callbacks
        if category in self.callbacks:
            for callback in self.callbacks[category]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in category callback: {str(e)}")

        # Dispatch to severity callbacks
        if severity in self.severity_callbacks:
            for callback in self.severity_callbacks[severity]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in severity callback: {str(e)}")


# Create global instance
event_subscriber = InterfaceEventSubscriber()
