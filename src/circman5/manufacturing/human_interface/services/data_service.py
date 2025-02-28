# src/circman5/manufacturing/human_interface/services/data_service.py

"""
Data service for CIRCMAN5.0 Human-Machine Interface.

This module provides centralized data retrieval for UI components,
handling caching, aggregation, and transformation of data from
various sources.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable, TypeVar, cast
import datetime
import threading
import time

from ....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ..core.interface_manager import interface_manager
from ..adapters.digital_twin_adapter import digital_twin_adapter
from ..adapters.event_adapter import EventAdapter

# Type variable for generic cache function
T = TypeVar("T")


class DataService:
    """
    Data service for the Human-Machine Interface.

    This service provides centralized data retrieval for UI components,
    handling caching, aggregation, and transformation of data from
    various sources.

    Attributes:
        digital_twin_adapter: Reference to digital twin adapter
        event_adapter: Reference to event adapter
        logger: Logger instance for this class
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the data service."""
        if self._initialized:
            return

        self.logger = setup_logger("data_service")
        self.constants = ConstantsService()

        # Get adapters
        self.digital_twin_adapter = digital_twin_adapter
        self.event_adapter = EventAdapter()  # Get instance

        # Initialize caches with TTLs
        self._caches = {
            "current_state": {
                "data": None,
                "last_update": datetime.datetime.min,
                "ttl": datetime.timedelta(seconds=1),
            },
            "kpi_data": {
                "data": None,
                "last_update": datetime.datetime.min,
                "ttl": datetime.timedelta(seconds=5),
            },
            "process_data": {
                "data": None,
                "last_update": datetime.datetime.min,
                "ttl": datetime.timedelta(seconds=2),
            },
            "alerts": {
                "data": None,
                "last_update": datetime.datetime.min,
                "ttl": datetime.timedelta(seconds=10),
            },
            "parameters": {
                "data": None,
                "last_update": datetime.datetime.min,
                "ttl": datetime.timedelta(seconds=30),
            },
        }

        # Thread safety
        self._lock = threading.RLock()

        # Register with interface manager
        interface_manager.register_component("data_service", self)

        # Register for events
        self.event_adapter.register_callback(
            self._on_system_state_change, category=None  # Register for all categories
        )

        self._initialized = True
        self.logger.info("Data Service initialized")

    def get_current_state(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get the current state of the system.

        Args:
            force_refresh: Whether to force a refresh of cached data

        Returns:
            Dict[str, Any]: Current state data
        """
        return self._get_cached_data(
            "current_state", self.digital_twin_adapter.get_current_state, force_refresh
        )

    def get_kpi_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get KPI data for the system.

        Args:
            force_refresh: Whether to force a refresh of cached data

        Returns:
            Dict[str, Any]: KPI data
        """

        # Function to extract KPI data from state
        def extract_kpi_data() -> Dict[str, Any]:
            current_state = self.digital_twin_adapter.get_current_state()
            kpi_data: Dict[str, Any] = {}

            # Extract timestamp
            kpi_data["timestamp"] = current_state.get(
                "timestamp", datetime.datetime.now().isoformat()
            )

            # Extract production metrics
            if "production_line" in current_state:
                prod_line = current_state["production_line"]

                # Production rate
                if "production_rate" in prod_line:
                    production_rate = prod_line["production_rate"]
                    kpi_data["production_rate"] = {
                        "value": production_rate,
                        "unit": "units/hour",
                        "trend": self._calculate_trend(
                            10, "production_line.production_rate"
                        ),
                    }

                # Energy consumption
                if "energy_consumption" in prod_line:
                    energy_consumption = prod_line["energy_consumption"]
                    kpi_data["energy_consumption"] = {
                        "value": energy_consumption,
                        "unit": "kWh",
                        "trend": self._calculate_trend(
                            10, "production_line.energy_consumption"
                        ),
                    }

                    # Energy efficiency
                    if "production_rate" in prod_line and energy_consumption > 0:
                        energy_efficiency = (
                            prod_line["production_rate"] / energy_consumption
                        )
                        kpi_data["energy_efficiency"] = {
                            "value": energy_efficiency,
                            "unit": "units/kWh",
                            "trend": self._calculate_trend(
                                10, "energy_efficiency", derived=True
                            ),
                        }

                # Defect rate
                if "defect_rate" in prod_line:
                    defect_rate = prod_line["defect_rate"]
                    kpi_data["defect_rate"] = {
                        "value": defect_rate,
                        "unit": "%",
                        "trend": self._calculate_trend(
                            10, "production_line.defect_rate"
                        ),
                    }

            return kpi_data

        return self._get_cached_data("kpi_data", extract_kpi_data, force_refresh)

    def get_process_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get process data for the system.

        Args:
            force_refresh: Whether to force a refresh of cached data

        Returns:
            Dict[str, Any]: Process data
        """

        # Function to extract process data from state
        def extract_process_data() -> Dict[str, Any]:
            current_state = self.digital_twin_adapter.get_current_state()
            process_data: Dict[str, Any] = {}

            # Extract timestamp
            process_data["timestamp"] = current_state.get(
                "timestamp", datetime.datetime.now().isoformat()
            )

            # Extract system status
            process_data["system_status"] = current_state.get(
                "system_status", "unknown"
            )

            # Extract production line data
            if "production_line" in current_state:
                process_data["production_line"] = current_state["production_line"]

            # Extract manufacturing processes data
            if "manufacturing_processes" in current_state:
                process_data["processes"] = current_state["manufacturing_processes"]

            return process_data

        return self._get_cached_data(
            "process_data", extract_process_data, force_refresh
        )

    def get_alerts(
        self,
        filter_settings: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get system alerts.

        Args:
            filter_settings: Optional filter settings
            force_refresh: Whether to force a refresh of cached data

        Returns:
            List[Dict[str, Any]]: Alerts data
        """
        # Ensure filter_settings is not None
        filter_settings = filter_settings or {}

        # Function to get alerts from event adapter
        def get_alerts() -> List[Dict[str, Any]]:
            # Get alert panel component through interface manager
            try:
                alert_component = interface_manager.get_component("alert_panel")
                return alert_component.get_filtered_alerts(filter_settings)
            except Exception as e:
                self.logger.error(f"Error getting alerts: {str(e)}")
                return []

        return self._get_cached_data("alerts", get_alerts, force_refresh)

    def get_parameters(
        self, group_name: Optional[str] = None, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get system parameters.

        Args:
            group_name: Optional parameter group name to filter by
            force_refresh: Whether to force a refresh of cached data

        Returns:
            List[Dict[str, Any]]: Parameters data
        """

        # Function to get parameters
        def get_parameters() -> List[Dict[str, Any]]:
            # Get parameter control component through interface manager
            try:
                param_component = interface_manager.get_component("parameter_control")
                return param_component.get_parameters(group_name)
            except Exception as e:
                self.logger.error(f"Error getting parameters: {str(e)}")
                return []

        return self._get_cached_data("parameters", get_parameters, force_refresh)

    def get_aggregated_dashboard_data(self) -> Dict[str, Any]:
        """
        Get aggregated data for dashboard rendering.

        Returns:
            Dict[str, Any]: Aggregated dashboard data
        """
        # Get all data types
        current_state = self.get_current_state()
        kpi_data = self.get_kpi_data()
        process_data = self.get_process_data()
        alerts = self.get_alerts()

        # Build aggregated data
        aggregated_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "system_status": current_state.get("system_status", "unknown"),
            "kpi_data": kpi_data,
            "process_data": process_data,
            "alerts": {
                "items": alerts[:10] if alerts else [],  # Only include top 10 alerts
                "total_count": len(alerts),
                "critical_count": sum(
                    1
                    for alert in alerts
                    if alert.get("severity", "").lower() == "critical"
                ),
            },
        }

        return aggregated_data

    def _get_cached_data(
        self, cache_key: str, data_func: Callable[[], T], force_refresh: bool = False
    ) -> T:
        """
        Get cached data, refreshing if needed.

        Args:
            cache_key: Cache identifier
            data_func: Function to call to refresh data
            force_refresh: Whether to force a refresh

        Returns:
            T: Cached data
        """
        with self._lock:
            cache = self._caches.get(cache_key)

            if not cache:
                self.logger.warning(f"Invalid cache key: {cache_key}")
                return data_func()

            now = datetime.datetime.now()

            # Check if we need to refresh
            if (
                force_refresh
                or cache["data"] is None
                or now - cache["last_update"] > cache["ttl"]
            ):
                # Call data function to refresh cache
                try:
                    cache["data"] = data_func()
                    cache["last_update"] = now
                except Exception as e:
                    self.logger.error(f"Error refreshing cache {cache_key}: {str(e)}")
                    # If cache is empty, return empty result based on cache key
                    if cache["data"] is None:
                        if cache_key in ["alerts", "parameters"]:
                            return cast(T, [])
                        else:
                            return cast(T, {})

            # Return cached data
            return cast(T, cache["data"])

    def _calculate_trend(
        self, history_length: int, metric_path: str, derived: bool = False
    ) -> str:
        """
        Calculate trend for a metric based on historical data.

        Args:
            history_length: Number of historical states to retrieve
            metric_path: Path to the metric in the state
            derived: Whether this is a derived metric

        Returns:
            str: Trend indicator ("up", "down", "stable", or "unknown")
        """
        try:
            # Get history from adapter
            history = self.digital_twin_adapter.get_state_history(history_length)

            if not history or len(history) < 2:
                return "unknown"

            # Extract values
            values = []

            # Extract based on metric path
            for state in history:
                if derived:
                    # Handle derived metrics
                    if metric_path == "energy_efficiency":
                        if "production_line" in state:
                            prod_line = state["production_line"]
                            if (
                                "production_rate" in prod_line
                                and "energy_consumption" in prod_line
                            ):
                                if prod_line["energy_consumption"] > 0:
                                    efficiency = (
                                        prod_line["production_rate"]
                                        / prod_line["energy_consumption"]
                                    )
                                    values.append(efficiency)
                else:
                    # Regular metrics
                    parts = metric_path.split(".")
                    value = state
                    try:
                        for part in parts:
                            value = value[part]
                        values.append(value)
                    except (KeyError, TypeError):
                        pass

            # Calculate trend direction
            if len(values) >= 2:
                first_half = values[: len(values) // 2]
                second_half = values[len(values) // 2 :]

                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)

                difference = second_avg - first_avg

                if abs(difference) < 0.01 * first_avg:
                    return "stable"
                elif difference > 0:
                    return "up"
                else:
                    return "down"

            return "unknown"

        except Exception as e:
            self.logger.error(f"Error calculating trend: {str(e)}")
            return "unknown"

    def _on_system_state_change(self, event: Any) -> None:
        """
        Handle system state change events.

        Args:
            event: Event data
        """
        # Invalidate relevant caches based on event
        with self._lock:
            # Always invalidate current state cache
            self._caches["current_state"]["last_update"] = datetime.datetime.min

            # Invalidate other caches based on event type
            if hasattr(event, "category"):
                category = (
                    event.category.value
                    if hasattr(event.category, "value")
                    else str(event.category)
                )

                if category.lower() == "system":
                    # System events invalidate process and KPI caches
                    self._caches["process_data"]["last_update"] = datetime.datetime.min
                    self._caches["kpi_data"]["last_update"] = datetime.datetime.min

                elif category.lower() == "threshold":
                    # Threshold events invalidate KPI cache
                    self._caches["kpi_data"]["last_update"] = datetime.datetime.min

                elif category.lower() in ["error", "warning"]:
                    # Error/warning events invalidate alerts cache
                    self._caches["alerts"]["last_update"] = datetime.datetime.min


# Create global instance
data_service = DataService()
