# src/circman5/manufacturing/digital_twin/event_notification/event_manager.py
from typing import Dict, List, Callable, Any, Optional, Set, Type
import threading
import logging
from dataclasses import dataclass
from enum import Enum

from ....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from .event_types import Event, EventCategory, EventSeverity
from .event_persistence import EventPersistence

# Type definitions for event handlers
EventHandler = Callable[[Event], None]
EventFilter = Callable[[Event], bool]


class EventManager:
    """
    Central event management system for Digital Twin events.

    This class implements a publisher-subscriber pattern for event distribution
    with filtering capabilities and persistence.

    Attributes:
        logger: Logger instance
        constants: Constants service for configuration access
        persistence: Event persistence service for storing events
        subscribers: Dictionary mapping event categories to handlers
        category_filters: Filters for event categories
        severity_subscribers: Dictionary mapping severity levels to handlers
        persistence_enabled: Whether event persistence is enabled
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(EventManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the event manager."""
        if self._initialized:
            return

        self.logger = setup_logger("event_manager")
        self.constants = ConstantsService()

        # Initialize event persistence
        self.persistence = EventPersistence()

        # Initialize subscriber dictionaries
        self.subscribers: Dict[EventCategory, List[EventHandler]] = {
            category: [] for category in EventCategory
        }

        # Initialize category filters
        self.category_filters: Dict[EventCategory, List[EventFilter]] = {
            category: [] for category in EventCategory
        }

        # Initialize severity-based subscribers
        self.severity_subscribers: Dict[EventSeverity, List[EventHandler]] = {
            severity: [] for severity in EventSeverity
        }

        # Lock for thread safety
        self._lock = threading.RLock()

        # Load configuration
        self._load_config()

        self._initialized = True
        self.logger.info("Event Manager initialized")

    def _load_config(self) -> None:
        """Load event system configuration from constants service."""
        try:
            # Get digital twin config
            dt_config = self.constants.get_digital_twin_config()

            # Check if event system configuration exists
            if "EVENT_NOTIFICATION" in dt_config:
                event_config = dt_config["EVENT_NOTIFICATION"]

                # Configure persistence
                self.persistence_enabled = event_config.get("persistence_enabled", True)
                self.persistence.set_max_events(event_config.get("max_events", 1000))

                # Configure other settings as needed
                self.logger.info("Loaded event notification configuration")
            else:
                # Use defaults
                self.persistence_enabled = True
                self.persistence.set_max_events(1000)
                self.logger.warning(
                    "No event notification configuration found, using defaults"
                )

        except Exception as e:
            self.logger.error(
                f"Error loading event notification configuration: {str(e)}"
            )
            # Use defaults
            self.persistence_enabled = True
            self.persistence.set_max_events(1000)

    def subscribe(
        self,
        handler: EventHandler,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
    ) -> None:
        """
        Subscribe a handler to events.

        Args:
            handler: Event handler function
            category: Optional category to subscribe to (all if None)
            severity: Optional severity to subscribe to (all if None)
        """
        with self._lock:
            if category:
                # Subscribe to specific category
                self.subscribers[category].append(handler)
                self.logger.debug(f"Handler subscribed to category: {category.value}")
            elif severity:
                # Subscribe to specific severity
                self.severity_subscribers[severity].append(handler)
                self.logger.debug(f"Handler subscribed to severity: {severity.value}")
            else:
                # Subscribe to all categories
                for cat in EventCategory:
                    self.subscribers[cat].append(handler)
                self.logger.debug("Handler subscribed to all event categories")

    def unsubscribe(
        self,
        handler: EventHandler,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
    ) -> None:
        """
        Unsubscribe a handler from events.

        Args:
            handler: Event handler function to unsubscribe
            category: Optional category to unsubscribe from (all if None)
            severity: Optional severity to unsubscribe from (all if None)
        """
        with self._lock:
            if category:
                # Unsubscribe from specific category
                if handler in self.subscribers[category]:
                    self.subscribers[category].remove(handler)
                    self.logger.debug(
                        f"Handler unsubscribed from category: {category.value}"
                    )
            elif severity:
                # Unsubscribe from specific severity
                if handler in self.severity_subscribers[severity]:
                    self.severity_subscribers[severity].remove(handler)
                    self.logger.debug(
                        f"Handler unsubscribed from severity: {severity.value}"
                    )
            else:
                # Unsubscribe from all categories
                for cat in EventCategory:
                    if handler in self.subscribers[cat]:
                        self.subscribers[cat].remove(handler)
                self.logger.debug("Handler unsubscribed from all event categories")

    def add_filter(self, category: EventCategory, filter_func: EventFilter) -> None:
        """
        Add a filter for a specific event category.

        Args:
            category: Event category to filter
            filter_func: Filter function that returns True for events to be processed
        """
        with self._lock:
            self.category_filters[category].append(filter_func)
            self.logger.debug(f"Filter added for category: {category.value}")

    def remove_filter(self, category: EventCategory, filter_func: EventFilter) -> None:
        """
        Remove a filter for a specific event category.

        Args:
            category: Event category to remove filter from
            filter_func: Filter function to remove
        """
        with self._lock:
            if filter_func in self.category_filters[category]:
                self.category_filters[category].remove(filter_func)
                self.logger.debug(f"Filter removed for category: {category.value}")

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribed handlers.

        Args:
            event: Event to publish
        """
        with self._lock:
            # Apply category filters
            if not self._passes_filters(event):
                return

            # Save event to persistence if enabled
            if self.persistence_enabled:
                self.persistence.save_event(event)

            # Get category handlers
            handlers = self.subscribers[event.category].copy()

            # Get severity handlers
            severity_handlers = self.severity_subscribers[event.severity].copy()

            # Combine handlers
            all_handlers = set(handlers + severity_handlers)

        # Dispatch event to each handler outside the lock
        for handler in all_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {str(e)}")

    def _passes_filters(self, event: Event) -> bool:
        """
        Check if event passes all filters for its category.

        Args:
            event: Event to check

        Returns:
            bool: True if event passes all filters or no filters exist
        """
        filters = self.category_filters[event.category]

        # If no filters, event passes
        if not filters:
            return True

        # Check all filters, event must pass at least one
        for filter_func in filters:
            try:
                if filter_func(event):
                    return True
            except Exception as e:
                self.logger.error(f"Error in event filter: {str(e)}")

        # Event did not pass any filter
        return False

    def get_events(
        self,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get events from persistence with optional filtering.

        Args:
            category: Optional category filter
            severity: Optional severity filter
            limit: Maximum number of events to return

        Returns:
            List[Event]: List of events matching the criteria
        """
        return self.persistence.get_events(
            category=category, severity=severity, limit=limit
        )

    def acknowledge_event(self, event_id: str) -> bool:
        """
        Mark an event as acknowledged.

        Args:
            event_id: ID of the event to acknowledge

        Returns:
            bool: True if event was found and acknowledged
        """
        return self.persistence.acknowledge_event(event_id)

    def clear_events(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear events from persistence.

        Args:
            older_than_days: Optional, clear events older than this many days

        Returns:
            int: Number of events cleared
        """
        return self.persistence.clear_events(older_than_days)


# Create global instance
event_manager = EventManager()
