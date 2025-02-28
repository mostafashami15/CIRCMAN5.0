# src/circman5/manufacturing/human_interface/components/dashboard/kpi_panel.py

"""
KPI panel component for CIRCMAN5.0 Human-Machine Interface.

This module implements the KPI panel that displays key performance indicators
for the manufacturing system, including production metrics, energy efficiency,
quality metrics, and material utilization.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
import datetime
import threading
import statistics

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.dashboard_manager import dashboard_manager
from ...core.interface_state import interface_state
from ...adapters.event_adapter import EventAdapter
from ....digital_twin.core.twin_core import DigitalTwin
from ....digital_twin.core.state_manager import StateManager


class KPIPanel:
    """
    KPI panel component for the Human-Machine Interface.

    This panel displays key performance indicators for the manufacturing
    system, including production metrics, energy efficiency, quality
    metrics, and material utilization.

    Attributes:
        state: Reference to interface state
        digital_twin: Reference to digital twin
        state_manager: Reference to state manager
        event_adapter: Event adapter for notifications
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the KPI panel."""
        self.logger = setup_logger("kpi_panel")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.digital_twin = DigitalTwin()  # Get instance
        self.state_manager = StateManager()  # Get instance
        self.event_adapter = EventAdapter()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        # Cache for KPI data
        self._kpi_cache: Dict[str, Any] = {}
        self._last_update = datetime.datetime.min
        self._cache_ttl = datetime.timedelta(seconds=2)  # Cache TTL of 2 seconds

        # Register with interface manager and dashboard manager
        interface_manager.register_component("kpi_panel", self)
        dashboard_manager.register_component("kpi_panel", self)

        # Register event handlers for system state changes
        self.event_adapter.register_callback(
            self._on_system_state_change,
            category=None,  # Will be registered for all categories
            severity=None,
        )

        self.logger.info("KPI Panel initialized")

    def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the KPI panel.

        Args:
            config: Panel configuration

        Returns:
            Dict[str, Any]: Panel data
        """
        # Get KPI data
        kpi_data = self._get_kpi_data(config.get("metrics"))

        # Prepare panel data
        panel_data = {
            "type": "kpi_panel",
            "title": config.get("title", "Key Performance Indicators"),
            "timestamp": datetime.datetime.now().isoformat(),
            "expanded": self.state.is_panel_expanded(config.get("id", "kpi")),
            "kpi_data": kpi_data,
            "config": config,
        }

        return panel_data

    def _get_kpi_data(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get current KPI data.

        Args:
            metrics: Optional list of specific metrics to retrieve

        Returns:
            Dict[str, Any]: KPI data
        """
        # Check if cached data is still valid
        now = datetime.datetime.now()
        if now - self._last_update < self._cache_ttl and self._kpi_cache:
            # If specific metrics requested, filter the cached data
            if metrics and self._kpi_cache:
                return {k: v for k, v in self._kpi_cache.items() if k in metrics}
            return self._kpi_cache.copy()

        # Get current state and history from digital twin
        try:
            current_state = self.digital_twin.get_current_state()
            history = self.state_manager.get_history(limit=10)

            # Initialize KPI data
            kpi_data = {"timestamp": current_state.get("timestamp", now.isoformat())}

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
                            history, "production_line.production_rate"
                        ),
                    }

                # Energy consumption
                if "energy_consumption" in prod_line:
                    energy_consumption = prod_line["energy_consumption"]
                    kpi_data["energy_consumption"] = {
                        "value": energy_consumption,
                        "unit": "kWh",
                        "trend": self._calculate_trend(
                            history, "production_line.energy_consumption"
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
                                history, "energy_efficiency", derived=True
                            ),
                        }

                # Defect rate
                if "defect_rate" in prod_line:
                    defect_rate = prod_line["defect_rate"]
                    kpi_data["defect_rate"] = {
                        "value": defect_rate,
                        "unit": "%",
                        "trend": self._calculate_trend(
                            history, "production_line.defect_rate"
                        ),
                    }

                # Yield rate
                if "yield_rate" in prod_line:
                    yield_rate = prod_line["yield_rate"]
                    kpi_data["yield_rate"] = {
                        "value": yield_rate,
                        "unit": "%",
                        "trend": self._calculate_trend(
                            history, "production_line.yield_rate"
                        ),
                    }

                # Efficiency
                if "efficiency" in prod_line:
                    efficiency = prod_line["efficiency"]
                    kpi_data["efficiency"] = {
                        "value": efficiency,
                        "unit": "",
                        "trend": self._calculate_trend(
                            history, "production_line.efficiency"
                        ),
                    }

            # Extract material utilization
            if "materials" in current_state:
                materials = current_state["materials"]
                total_inventory = 0
                material_count = 0

                for material, properties in materials.items():
                    if isinstance(properties, dict) and "inventory" in properties:
                        total_inventory += properties["inventory"]
                        material_count += 1

                if material_count > 0:
                    kpi_data["material_inventory"] = {
                        "value": total_inventory,
                        "unit": "units",
                        "trend": self._calculate_trend(
                            history, "material_inventory", derived=True
                        ),
                    }

            # Update cache
            self._kpi_cache = kpi_data
            self._last_update = now

            # If specific metrics requested, filter the data
            if metrics:
                return {k: v for k, v in kpi_data.items() if k in metrics}

            return kpi_data

        except Exception as e:
            self.logger.error(f"Error getting KPI data: {str(e)}")

            # Return basic error data
            return {"timestamp": now.isoformat(), "error": str(e)}

    def _calculate_trend(
        self, history: List[Dict[str, Any]], metric_path: str, derived: bool = False
    ) -> str:
        """
        Calculate trend for a metric based on historical data.

        Args:
            history: List of historical states
            metric_path: Path to the metric in the state (e.g., "production_line.temperature")
            derived: Whether this is a derived metric not directly in the state

        Returns:
            str: Trend indicator ("up", "down", "stable", or "unknown")
        """
        if not history or len(history) < 2:
            return "unknown"

        try:
            # Extract values from history
            values = []

            if derived:
                # Handle derived metrics
                if metric_path == "energy_efficiency":
                    for state in history:
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

                elif metric_path == "material_inventory":
                    for state in history:
                        if "materials" in state:
                            materials = state["materials"]
                            total_inventory = sum(
                                material.get("inventory", 0)
                                for material in materials.values()
                                if isinstance(material, dict)
                            )
                            values.append(total_inventory)
            else:
                # Handle regular metrics
                parts = metric_path.split(".")

                for state in history:
                    current = state
                    try:
                        for part in parts:
                            current = current[part]
                        values.append(current)
                    except (KeyError, TypeError):
                        pass

            # Calculate trend
            if len(values) >= 2:
                # Linear regression slope (simple approximation)
                n = len(values)
                x_values = list(range(n))
                x_mean = statistics.mean(x_values)
                y_mean = statistics.mean(values)

                numerator = sum(
                    (x - x_mean) * (y - y_mean) for x, y in zip(x_values, values)
                )
                denominator = sum((x - x_mean) ** 2 for x in x_values)

                if denominator != 0:
                    slope = numerator / denominator

                    # Determine trend
                    if slope > 0.01:
                        return "up"
                    elif slope < -0.01:
                        return "down"
                    else:
                        return "stable"

            return "unknown"

        except Exception as e:
            self.logger.warning(f"Error calculating trend for {metric_path}: {str(e)}")
            return "unknown"

    def _on_system_state_change(self, event: Any) -> None:
        """
        Handle system state change events.

        Args:
            event: Event data
        """
        # Invalidate cache
        self._last_update = datetime.datetime.min

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle KPI panel commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "get_kpi_data":
            metrics = params.get("metrics")
            kpi_data = self._get_kpi_data(metrics)
            return {"handled": True, "success": True, "kpi_data": kpi_data}

        elif command == "refresh_kpis":
            # Invalidate cache
            self._last_update = datetime.datetime.min
            metrics = params.get("metrics")
            kpi_data = self._get_kpi_data(metrics)
            return {"handled": True, "success": True, "kpi_data": kpi_data}

        # Not a KPI panel command
        return {"handled": False}


# Create global instance
kpi_panel = KPIPanel()
