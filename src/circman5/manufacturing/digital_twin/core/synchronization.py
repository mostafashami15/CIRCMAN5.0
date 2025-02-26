# src/circman5/manufacturing/digital_twin/core/synchronization.py

"""
Synchronization module for CIRCMAN5.0 Digital Twin.

This module handles the synchronization between the physical manufacturing system
and its digital representation, ensuring data consistency and temporal alignment.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import datetime
import time
import json
import threading
from enum import Enum

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from .state_manager import StateManager


class SyncMode(Enum):
    """Synchronization modes for the digital twin."""

    REAL_TIME = "real_time"  # Continuous real-time synchronization
    BATCH = "batch"  # Periodic batch synchronization
    MANUAL = "manual"  # Manual synchronization
    EVENT_DRIVEN = "event"  # Event-driven synchronization


class SynchronizationManager:
    """
    Manages synchronization between physical system and digital twin.

    The SynchronizationManager handles data collection from physical systems,
    synchronization of state, and temporal alignment between the physical and
    digital representations.

    Attributes:
        state_manager: StateManager instance for accessing the digital twin state
        sync_mode: Current synchronization mode
        data_sources: Dictionary of data source names to data collection functions
        sync_interval: Interval for periodic synchronization in seconds
        logger: Logger instance for this class
    """

    def __init__(
        self,
        state_manager: StateManager,
        sync_mode: Optional[SyncMode] = None,
        sync_interval: Optional[float] = None,
    ):
        """
        Initialize the SynchronizationManager.

        Args:
            state_manager: StateManager instance for the digital twin
            sync_mode: Optional synchronization mode to use
            sync_interval: Optional interval for periodic synchronization in seconds
        """
        self.state_manager = state_manager

        # Load configuration from constants service
        self.constants = ConstantsService()
        self.dt_config = self.constants.get_digital_twin_config()
        self.sync_config = self.dt_config.get("SYNCHRONIZATION_CONFIG", {})

        # Set sync mode from parameter or config
        if sync_mode:
            self.sync_mode = sync_mode
        else:
            config_mode = self.sync_config.get("default_sync_mode", "real_time").lower()
            if config_mode == "real_time":
                self.sync_mode = SyncMode.REAL_TIME
            elif config_mode == "batch":
                self.sync_mode = SyncMode.BATCH
            elif config_mode == "manual":
                self.sync_mode = SyncMode.MANUAL
            elif config_mode == "event":
                self.sync_mode = SyncMode.EVENT_DRIVEN
            else:
                self.sync_mode = SyncMode.REAL_TIME

        # Set sync interval from parameter or config
        self.sync_interval = sync_interval or self.sync_config.get(
            "default_sync_interval", 1.0
        )

        # Initialize other attributes
        self.data_sources: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.logger = setup_logger("synchronization_manager")
        self.is_running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.logger.info(
            f"SynchronizationManager initialized with mode {self.sync_mode.value}"
        )

    def register_data_source(
        self, name: str, collector_func: Callable[[], Dict[str, Any]]
    ) -> None:
        """
        Register a data source for synchronization.

        Args:
            name: Name of the data source
            collector_func: Function that collects data from the source
        """
        self.data_sources[name] = collector_func
        self.logger.info(f"Registered data source: {name}")

    def start_synchronization(self) -> bool:
        """
        Start the synchronization process.

        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            self.logger.warning("Synchronization is already running")
            return False

        try:
            if self.sync_mode == SyncMode.REAL_TIME or self.sync_mode == SyncMode.BATCH:
                # Start synchronization in a separate thread
                self._stop_event.clear()
                self._sync_thread = threading.Thread(target=self._sync_loop)
                self._sync_thread.daemon = True
                self._sync_thread.start()
                self.is_running = True
                self.logger.info(f"Started {self.sync_mode.value} synchronization")
                return True
            elif self.sync_mode == SyncMode.MANUAL:
                # Manual mode doesn't start a background thread
                self.is_running = True
                self.logger.info("Manual synchronization mode active")
                return True
            else:
                self.logger.error(
                    f"Unsupported synchronization mode: {self.sync_mode.value}"
                )
                return False
        except Exception as e:
            self.logger.error(f"Failed to start synchronization: {str(e)}")
            return False

    def stop_synchronization(self) -> bool:
        """
        Stop the synchronization process.

        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            self.logger.warning("Synchronization is not running")
            return False

        try:
            self._stop_event.set()
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=2.0)
            self.is_running = False
            self.logger.info("Stopped synchronization")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop synchronization: {str(e)}")
            return False

    def synchronize_now(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Perform immediate synchronization.

        Args:
            save_results: Whether to save synchronization results to disk

        Returns:
            Dict[str, Any]: Dictionary of synchronized data
        """
        self.logger.info("Performing immediate synchronization")

        try:
            # Collect data from all registered sources
            sync_data = self._collect_data()

            # Add timestamp
            timestamp = datetime.datetime.now()
            sync_data["timestamp"] = timestamp.isoformat()

            # Update state manager with new data
            self.state_manager.update_state(sync_data)

            # Save synchronization results if requested
            if save_results:
                self._save_sync_results(sync_data, timestamp)

            self.logger.info(
                f"Synchronization completed with {len(sync_data)} data points"
            )
            return sync_data

        except Exception as e:
            self.logger.error(f"Synchronization failed: {str(e)}")
            return {"error": str(e)}

    def set_sync_mode(self, mode: SyncMode) -> None:
        """
        Change the synchronization mode.

        Args:
            mode: New synchronization mode
        """
        if self.is_running:
            self.stop_synchronization()

        self.sync_mode = mode
        self.logger.info(f"Changed synchronization mode to {mode.value}")

        if self.is_running:
            self.start_synchronization()

    def set_sync_interval(self, interval: float) -> None:
        """
        Set the synchronization interval.

        Args:
            interval: New interval in seconds
        """
        self.sync_interval = interval
        self.logger.info(f"Set synchronization interval to {interval} seconds")

    def _sync_loop(self) -> None:
        """
        Main synchronization loop for continuous and batch modes.
        """
        self.logger.info(
            f"Starting synchronization loop with interval {self.sync_interval}s"
        )

        while not self._stop_event.is_set():
            start_time = time.time()

            try:
                # Perform synchronization
                self.synchronize_now()

                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.sync_interval - elapsed)

                # Sleep until next synchronization
                if sleep_time > 0 and not self._stop_event.is_set():
                    time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in synchronization loop: {str(e)}")
                # Sleep before retrying
                time.sleep(1.0)

    def _collect_data(self) -> Dict[str, Any]:
        """
        Collect data from all registered sources.

        Returns:
            Dict[str, Any]: Combined data from all sources
        """
        all_data: Dict[str, Any] = {}

        for source_name, collector_func in self.data_sources.items():
            try:
                source_data = collector_func()
                if source_data:
                    # Add data with source name as key
                    all_data[source_name] = source_data
            except Exception as e:
                self.logger.error(
                    f"Error collecting data from source {source_name}: {str(e)}"
                )

        return all_data

    def register_event_handler(
        self, event_name: str, handler_func: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register an event handler for event-driven synchronization.

        Args:
            event_name: Name of the event to handle
            handler_func: Function to call when the event occurs
        """
        # This is a placeholder - in a real implementation, this would register
        # event handlers for specific synchronization events
        pass

    def notify_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """
        Notify the synchronization manager of an event.

        Args:
            event_name: Name of the event
            event_data: Data associated with the event
        """
        # This is a placeholder - in a real implementation, this would trigger
        # event-driven synchronization when events occur
        pass

    def _save_sync_results(
        self, sync_data: Dict[str, Any], timestamp: datetime.datetime
    ) -> None:
        """
        Save synchronization results using results_manager.

        Args:
            sync_data: Synchronization data to save
            timestamp: Timestamp of the synchronization
        """
        try:
            # Create filename with timestamp
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"sync_data_{timestamp_str}.json"
            temp_path = Path(filename)

            # Save to temporary file
            with open(temp_path, "w") as f:
                json.dump(sync_data, f, indent=2)

            # Save file using results_manager
            results_manager.save_file(temp_path, "digital_twin")

            # Clean up temporary file
            temp_path.unlink()
            self.logger.debug(f"Synchronization data saved to {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save synchronization results: {str(e)}")
