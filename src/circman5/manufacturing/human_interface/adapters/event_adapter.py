# src/circman5/manufacturing/human_interface/adapters/event_adapter.py

"""
Event adapter for the CIRCMAN5.0 Human-Machine Interface.

This module provides the integration between the human interface system
and the event notification system, enabling event subscription and
alert visualization.
"""

from typing import Dict, Any, List, Optional, Set, Union, Callable
import threading
import datetime
import queue

from ....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...digital_twin.event_notification.event_manager import event_manager
from ...digital_twin.event_notification.event_types import (
    Event,
    EventCategory,
    EventSeverity,
)
from ...digital_twin.event_notification.subscribers import Subscriber


class InterfaceEventSubscriber(Subscriber):
    """
    Event subscriber for the interface.

    This class subscribes to events from the event notification system
    and forwards them to the interface.
    """

    def __init__(self, callback: Optional[Callable[[Event], None]] = None):
        """
        Initialize the interface event subscriber.

        Args:
            callback: Optional callback function for events
        """
        super().__init__(name="interface_subscriber")
        self.callback = callback

    def handle_event(self, event: Event) -> None:
        """
        Handle an event from the event notification system.

        Args:
            event: Event to handle
        """
        self.logger.debug(f"Received event: {event.event_id} - {event.message}")

        # Call callback if provided
        if self.callback:
            try:
                self.callback(event)
            except Exception as e:
                self.logger.error(f"Error in event callback: {str(e)}")


class EventAdapter:
    """
    Adapter for the event notification system.

    This class provides an interface between the HMI and the event notification
    system, handling event subscription, filtering, and dispatch to UI components.

    Attributes:
        event_queue: Queue for events to be processed
        event_callbacks: Registered callbacks for different event categories
        subscriber: Event subscriber for the notification system
        is_running: Whether the event processing thread is running
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the event adapter."""
        self.logger = setup_logger("event_adapter")
        self.constants = ConstantsService()

        # Create event queue
        self.event_queue = queue.Queue()

        # Initialize callback registrations
        self.event_callbacks: Dict[EventCategory, List[Callable[[Event], None]]] = {
            category: [] for category in EventCategory
        }
        self.severity_callbacks: Dict[EventSeverity, List[Callable[[Event], None]]] = {
            severity: [] for severity in EventSeverity
        }

        # Create subscriber
        self.subscriber = InterfaceEventSubscriber(callback=self._on_event)

        # Initialize thread management
        self._stop_event = threading.Event()
        self._event_thread: Optional[threading.Thread] = None
        self.is_running = False

        self.logger.info("Event Adapter initialized")

    def initialize(self) -> None:
        """Initialize the event adapter and start event processing."""
        try:
            # Register subscriber for all event categories
            for category in EventCategory:
                self.subscriber.register(category=category)

            # Start event processing thread
            self._stop_event.clear()
            self._event_thread = threading.Thread(
                target=self._process_events, daemon=True
            )
            self._event_thread.start()
            self.is_running = True

            self.logger.info("Event Adapter started")
        except Exception as e:
            self.logger.error(f"Error initializing event adapter: {str(e)}")
            raise

    def shutdown(self) -> None:
        """Shut down the event adapter."""
        if self.is_running:
            # Signal thread to stop
            self._stop_event.set()

            # Wait for thread to terminate
            if self._event_thread and self._event_thread.is_alive():
                self._event_thread.join(timeout=2.0)

            self.is_running = False
            self.logger.info("Event Adapter shutdown")

    def register_callback(
        self,
        callback: Callable[[Event], None],
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
    ) -> None:
        """
        Register a callback for events.

        Args:
            callback: Callback function
            category: Optional event category to filter for
            severity: Optional event severity to filter for
        """
        if category:
            self.event_callbacks[category].append(callback)
            self.logger.debug(f"Callback registered for category: {category.value}")
        elif severity:
            self.severity_callbacks[severity].append(callback)
            self.logger.debug(f"Callback registered for severity: {severity.value}")
        else:
            # If neither category nor severity specified, register for all categories
            for cat in self.event_callbacks:
                self.event_callbacks[cat].append(callback)
            self.logger.debug("Callback registered for all categories")

    def unregister_callback(
        self,
        callback: Callable[[Event], None],
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
    ) -> None:
        """
        Unregister a callback.

        Args:
            callback: Callback function to unregister
            category: Optional event category to unregister from
            severity: Optional event severity to unregister from
        """
        if category:
            if callback in self.event_callbacks[category]:
                self.event_callbacks[category].remove(callback)
                self.logger.debug(
                    f"Callback unregistered from category: {category.value}"
                )
        elif severity:
            if callback in self.severity_callbacks[severity]:
                self.severity_callbacks[severity].remove(callback)
                self.logger.debug(
                    f"Callback unregistered from severity: {severity.value}"
                )
        else:
            # If neither category nor severity specified, unregister from all
            for cat in self.event_callbacks:
                if callback in self.event_callbacks[cat]:
                    self.event_callbacks[cat].remove(callback)
            for sev in self.severity_callbacks:
                if callback in self.severity_callbacks[sev]:
                    self.severity_callbacks[sev].remove(callback)
            self.logger.debug(
                "Callback unregistered from all categories and severities"
            )

    def get_recent_events(
        self,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get recent events from the event notification system.

        Args:
            category: Optional category filter
            severity: Optional severity filter
            limit: Maximum number of events to retrieve

        Returns:
            List[Event]: List of matching events
        """
        return event_manager.get_events(
            category=category, severity=severity, limit=limit
        )

    def acknowledge_event(self, event_id: str) -> bool:
        """
        Acknowledge an event.

        Args:
            event_id: ID of the event to acknowledge

        Returns:
            bool: True if event was acknowledged
        """
        return event_manager.acknowledge_event(event_id)

    def _on_event(self, event: Event) -> None:
        """
        Handler for events from the event notification system.

        Args:
            event: Event received
        """
        # Add event to processing queue
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

                # Process the event
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
        # Callbacks for the specific category
        for callback in self.event_callbacks[event.category]:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in event callback: {str(e)}")

        # Callbacks for the specific severity
        for callback in self.severity_callbacks[event.severity]:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in severity callback: {str(e)}")
