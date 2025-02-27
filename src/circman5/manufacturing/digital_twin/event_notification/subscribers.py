# src/circman5/manufacturing/digital_twin/event_notification/subscribers.py

from typing import Dict, Any, Optional, List, Callable
import logging

from ....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from .event_manager import event_manager, EventManager
from .event_types import Event, EventCategory, EventSeverity


class Subscriber:
    """Base class for event subscribers."""

    def __init__(self, name: str):
        """
        Initialize subscriber.

        Args:
            name: Subscriber name for identification
        """
        self.logger = setup_logger(f"subscriber_{name}")
        self.constants = ConstantsService()
        self.name = name
        self.registered = False

    def handle_event(self, event: Event) -> None:
        """
        Handle an event.

        Args:
            event: Event to handle
        """
        # Base implementation logs the event
        self.logger.debug(f"Received event: {event.event_id} - {event.message}")

    def register(
        self,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
    ) -> None:
        """
        Register with event manager.

        Args:
            category: Optional category to subscribe to
            severity: Optional severity to subscribe to
        """
        event_manager.subscribe(
            handler=self.handle_event, category=category, severity=severity
        )
        self.registered = True
        self.logger.info(f"Subscriber {self.name} registered")

    def unregister(
        self,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
    ) -> None:
        """
        Unregister from event manager.

        Args:
            category: Optional category to unsubscribe from
            severity: Optional severity to unsubscribe from
        """
        event_manager.unsubscribe(
            handler=self.handle_event, category=category, severity=severity
        )
        self.registered = False
        self.logger.info(f"Subscriber {self.name} unregistered")


class LoggingSubscriber(Subscriber):
    """Subscriber that logs all events."""

    def __init__(self, name: str = "logging"):
        """Initialize logging subscriber."""
        super().__init__(name)

    def handle_event(self, event: Event) -> None:
        """
        Handle event by logging it.

        Args:
            event: Event to handle
        """
        # Log based on severity
        if event.severity == EventSeverity.INFO:
            self.logger.info(f"EVENT: {event.message}")
        elif event.severity == EventSeverity.WARNING:
            self.logger.warning(f"EVENT: {event.message}")
        elif event.severity == EventSeverity.ERROR:
            self.logger.error(f"EVENT: {event.message}")
        elif event.severity == EventSeverity.CRITICAL:
            self.logger.critical(f"EVENT: {event.message}")


class ThresholdAlertSubscriber(Subscriber):
    """Subscriber that handles threshold breach events."""

    def __init__(self, name: str = "threshold_alerts"):
        """Initialize threshold alert subscriber."""
        super().__init__(name)
        self.alert_handlers: List[Callable[[Event], None]] = []

    def handle_event(self, event: Event) -> None:
        """
        Handle threshold breach events.

        Args:
            event: Event to handle
        """
        # Only process threshold events
        if event.category != EventCategory.THRESHOLD:
            return

        # Log the threshold breach
        self.logger.warning(
            f"THRESHOLD BREACH: {event.message} "
            f"(Source: {event.source}, Severity: {event.severity.value})"
        )

        # Call registered alert handlers
        for handler in self.alert_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {str(e)}")

    def add_alert_handler(self, handler: Callable[[Event], None]) -> None:
        """
        Add a handler for threshold alerts.

        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[Event], None]) -> None:
        """
        Remove a handler for threshold alerts.

        Args:
            handler: Handler to remove
        """
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)


class OptimizationSubscriber(Subscriber):
    """Subscriber that handles optimization events."""

    def __init__(self, name: str = "optimization"):
        """Initialize optimization subscriber."""
        super().__init__(name)
        self.optimization_handlers: List[Callable[[Event], None]] = []

    def handle_event(self, event: Event) -> None:
        """
        Handle optimization events.

        Args:
            event: Event to handle
        """
        # Only process optimization events
        if event.category != EventCategory.OPTIMIZATION:
            return

        # Log the optimization opportunity
        self.logger.info(f"OPTIMIZATION: {event.message} " f"(Source: {event.source})")

        # Call registered optimization handlers
        for handler in self.optimization_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in optimization handler: {str(e)}")

    def add_optimization_handler(self, handler: Callable[[Event], None]) -> None:
        """
        Add a handler for optimization events.

        Args:
            handler: Function to handle optimization events
        """
        self.optimization_handlers.append(handler)

    def remove_optimization_handler(self, handler: Callable[[Event], None]) -> None:
        """
        Remove a handler for optimization events.

        Args:
            handler: Handler to remove
        """
        if handler in self.optimization_handlers:
            self.optimization_handlers.remove(handler)
