# src/circman5/manufacturing/human_interface/services/update_service.py

"""
Update service for CIRCMAN5.0 Human-Machine Interface.

This module provides a centralized update coordination system for the
human interface, managing refresh rates, update notifications, and
periodic data polling.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable
import datetime
import threading
import time

from ....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ..core.interface_manager import interface_manager


class UpdateService:
    """
    Update service for the Human-Machine Interface.

    This service provides a centralized update coordination system for the
    human interface, managing refresh rates, update notifications, and
    periodic data polling.

    Attributes:
        update_intervals: Dictionary of update intervals for different components
        update_handlers: Dictionary of update handlers
        is_running: Whether the update service is running
        logger: Logger instance for this class
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(UpdateService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the update service."""
        if self._initialized:
            return

        self.logger = setup_logger("update_service")
        self.constants = ConstantsService()

        # Initialize update intervals (in seconds)
        self.update_intervals = {
            "status": 1.0,
            "kpi": 5.0,
            "process": 2.0,
            "alerts": 10.0,
            "parameters": 30.0,
        }

        # Initialize update handlers
        self.update_handlers: Dict[str, List[Callable]] = {
            component: [] for component in self.update_intervals
        }

        # Initialize thread management
        self.is_running = False
        self._update_threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Register with interface manager
        interface_manager.register_component("update_service", self)

        self._initialized = True
        self.logger.info("Update Service initialized")

    def register_update_handler(self, component: str, handler: Callable) -> None:
        """
        Register an update handler.

        Args:
            component: Component name
            handler: Handler function

        Raises:
            ValueError: If component not recognized
        """
        with self._lock:
            if component not in self.update_intervals:
                raise ValueError(f"Unknown component: {component}")

            self.update_handlers[component].append(handler)
            self.logger.debug(f"Update handler registered for {component}")

    def unregister_update_handler(self, component: str, handler: Callable) -> bool:
        """
        Unregister an update handler.

        Args:
            component: Component name
            handler: Handler function

        Returns:
            bool: True if handler was unregistered
        """
        with self._lock:
            if component not in self.update_handlers:
                return False

            if handler in self.update_handlers[component]:
                self.update_handlers[component].remove(handler)
                self.logger.debug(f"Update handler unregistered from {component}")
                return True

            return False

    def set_update_interval(self, component: str, interval: float) -> None:
        """
        Set update interval for a component.

        Args:
            component: Component name
            interval: Update interval in seconds

        Raises:
            ValueError: If component not recognized or interval invalid
        """
        with self._lock:
            if component not in self.update_intervals:
                raise ValueError(f"Unknown component: {component}")

            if interval <= 0:
                raise ValueError(f"Invalid interval: {interval}")

            self.update_intervals[component] = interval
            self.logger.info(f"Update interval for {component} set to {interval}s")

            # Restart update thread if running
            if self.is_running and component in self._update_threads:
                self._stop_component_updates(component)
                self._start_component_updates(component)

    def start_updates(self) -> None:
        """Start all update threads."""
        with self._lock:
            if self.is_running:
                self.logger.warning("Update service already running")
                return

            self.logger.info("Starting update service")

            # Start update threads for each component
            for component in self.update_intervals:
                self._start_component_updates(component)

            self.is_running = True

    def stop_updates(self) -> None:
        """Stop all update threads."""
        with self._lock:
            if not self.is_running:
                self.logger.warning("Update service not running")
                return

            self.logger.info("Stopping update service")

            # Stop all update threads
            for component in list(self._update_threads.keys()):
                self._stop_component_updates(component)

            self.is_running = False

    def trigger_update(self, component: str) -> None:
        """
        Trigger an immediate update for a component.

        Args:
            component: Component name

        Raises:
            ValueError: If component not recognized
        """
        with self._lock:
            if component not in self.update_handlers:
                raise ValueError(f"Unknown component: {component}")

            # Call all handlers for this component
            self._trigger_component_handlers(component)

    def _start_component_updates(self, component: str) -> None:
        """
        Start update thread for a component.

        Args:
            component: Component name
        """
        # Create stop event
        stop_event = threading.Event()
        self._stop_events[component] = stop_event

        # Create and start thread
        thread = threading.Thread(
            target=self._update_loop, args=(component, stop_event), daemon=True
        )
        self._update_threads[component] = thread
        thread.start()

        self.logger.debug(f"Started update thread for {component}")

    def _stop_component_updates(self, component: str) -> None:
        """
        Stop update thread for a component.

        Args:
            component: Component name
        """
        if component in self._stop_events:
            # Signal thread to stop
            self._stop_events[component].set()

            # Wait for thread to terminate
            if component in self._update_threads:
                thread = self._update_threads[component]
                if thread and thread.is_alive():
                    thread.join(timeout=2.0)

                # Remove thread
                del self._update_threads[component]

            # Remove stop event
            del self._stop_events[component]

            self.logger.debug(f"Stopped update thread for {component}")

    def _update_loop(self, component: str, stop_event: threading.Event) -> None:
        """
        Update loop for a component.

        Args:
            component: Component name
            stop_event: Event to signal thread to stop
        """
        self.logger.debug(f"Update loop started for {component}")

        while not stop_event.is_set():
            try:
                # Get update interval for this component
                interval = self.update_intervals[component]

                # Trigger update handlers
                start_time = time.time()
                self._trigger_component_handlers(component)

                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0.1, interval - elapsed)

                # Sleep until next update
                if stop_event.wait(sleep_time):
                    break

            except Exception as e:
                self.logger.error(f"Error in update loop for {component}: {str(e)}")
                time.sleep(1.0)  # Sleep briefly on error

        self.logger.debug(f"Update loop stopped for {component}")

    def _trigger_component_handlers(self, component: str) -> None:
        """
        Trigger all update handlers for a component.

        Args:
            component: Component name
        """
        with self._lock:
            handlers = self.update_handlers[component].copy()

        # Call all handlers
        for handler in handlers:
            try:
                handler()
            except Exception as e:
                self.logger.error(f"Error in update handler for {component}: {str(e)}")

        # Notify via interface manager event
        interface_manager.trigger_event(
            "component_updated",
            {"component": component, "timestamp": datetime.datetime.now().isoformat()},
        )


# Create global instance
update_service = UpdateService()
