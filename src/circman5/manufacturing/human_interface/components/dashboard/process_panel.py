# src/circman5/manufacturing/human_interface/components/dashboard/process_panel.py

"""
Process panel component for CIRCMAN5.0 Human-Machine Interface.

This module implements the process panel that visualizes manufacturing processes,
showing process flow, current state, and operational parameters.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
import datetime
import threading

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.dashboard_manager import dashboard_manager
from ...core.interface_state import interface_state
from ...adapters.event_adapter import EventAdapter
from ....digital_twin.core.twin_core import DigitalTwin
from ....digital_twin.visualization.process_visualizer import ProcessVisualizer


class ProcessPanel:
    """
    Process panel component for the Human-Machine Interface.

    This panel visualizes manufacturing processes, showing process flow,
    current state, and operational parameters.

    Attributes:
        state: Reference to interface state
        digital_twin: Reference to digital twin
        process_visualizer: Reference to process visualizer
        event_adapter: Event adapter for notifications
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the process panel."""
        self.logger = setup_logger("process_panel")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.digital_twin = DigitalTwin()  # Get instance
        self.event_adapter = EventAdapter()  # Get instance

        # Get manufacturing configuration
        self.manufacturing_config = self.constants.get_manufacturing_constants()
        self.process_stages = self.manufacturing_config.get("MANUFACTURING_STAGES", {})

        # Initialize process visualizer with state manager
        self.process_visualizer = ProcessVisualizer(self.digital_twin.state_manager)

        # Thread safety
        self._lock = threading.RLock()

        # Cache for process data
        self._process_cache: Dict[str, Any] = {}
        self._last_update = datetime.datetime.min
        self._cache_ttl = datetime.timedelta(seconds=2)  # Cache TTL of 2 seconds

        # Register with interface manager and dashboard manager
        interface_manager.register_component("process_panel", self)
        dashboard_manager.register_component("process_panel", self)

        # Register event handlers for system state changes
        self.event_adapter.register_callback(
            self._on_system_state_change,
            category=None,  # Will be registered for all categories
            severity=None,
        )

        self.logger.info("Process Panel initialized")

    def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the process panel.

        Args:
            config: Panel configuration

        Returns:
            Dict[str, Any]: Panel data
        """
        # Get process data
        process_data = self._get_process_data()
        selected_process = self.state.get_selected_process()

        # Prepare panel data
        panel_data = {
            "type": "process_panel",
            "title": config.get("title", "Manufacturing Process"),
            "timestamp": datetime.datetime.now().isoformat(),
            "expanded": self.state.is_panel_expanded(config.get("id", "process")),
            "process_data": process_data,
            "selected_process": selected_process or "main_process",
            "config": config,
        }

        return panel_data

    def _get_process_data(self) -> Dict[str, Any]:
        """
        Get current process data.

        Returns:
            Dict[str, Any]: Process data
        """
        # Check if cached data is still valid
        now = datetime.datetime.now()
        if now - self._last_update < self._cache_ttl and self._process_cache:
            return self._process_cache.copy()

        # Get current state from digital twin
        try:
            current_state = self.digital_twin.get_current_state()

            # Extract process information
            process_data = {
                "timestamp": current_state.get("timestamp", now.isoformat()),
                "main_process": self._extract_main_process_data(current_state),
                "stages": self._extract_stage_data(current_state),
            }

            # Update cache
            self._process_cache = process_data
            self._last_update = now

            return process_data

        except Exception as e:
            self.logger.error(f"Error getting process data: {str(e)}")

            # Return basic error data
            return {
                "timestamp": now.isoformat(),
                "error": str(e),
                "main_process": {"status": "error"},
                "stages": {},
            }

    def _extract_main_process_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract main process data from state.

        Args:
            state: Current state

        Returns:
            Dict[str, Any]: Main process data
        """
        # Initialize with defaults
        main_process = {"status": "unknown", "active_stages": [], "throughput": 0.0}

        # Extract production line information
        if "production_line" in state:
            prod_line = state["production_line"]
            main_process["status"] = prod_line.get("status", "unknown")

            if "production_rate" in prod_line:
                main_process["throughput"] = prod_line["production_rate"]

        # Determine active stages
        active_stages = []

        # Check if manufacturing_processes key exists
        if "manufacturing_processes" in state:
            for stage_name, stage_data in state["manufacturing_processes"].items():
                if (
                    isinstance(stage_data, dict)
                    and stage_data.get("status") == "active"
                ):
                    active_stages.append(stage_name)

        # If no explicit stage data, infer from configuration
        if not active_stages and self.process_stages:
            if (
                "production_line" in state
                and state["production_line"].get("status") == "running"
            ):
                # If production line is running, assume all stages are active
                active_stages = list(self.process_stages.keys())

        main_process["active_stages"] = active_stages

        return main_process

    def _extract_stage_data(self, state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract stage data from state.

        Args:
            state: Current state

        Returns:
            Dict[str, Dict[str, Any]]: Process stage data
        """
        # Initialize stages dictionary
        stages = {}

        # Check if manufacturing_processes key exists
        if "manufacturing_processes" in state:
            # Extract stage data from state
            for stage_name, stage_data in state["manufacturing_processes"].items():
                if isinstance(stage_data, dict):
                    stages[stage_name] = stage_data.copy()

        # If no explicit stage data, create synthetic data from configuration
        if not stages and self.process_stages:
            # Create synthetic data based on configuration
            prod_line_status = "idle"
            if "production_line" in state:
                prod_line_status = state["production_line"].get("status", "idle")

            # Create stage data for each configured stage
            for stage_name, stage_config in self.process_stages.items():
                # Default status based on production line
                status = "active" if prod_line_status == "running" else "idle"

                # Create synthetic metrics
                stages[stage_name] = {
                    "status": status,
                    "name": stage_config.get("name", stage_name),
                    "description": stage_config.get("description", ""),
                    "position": stage_config.get("position", 0),
                    "metrics": {
                        "yield_rate": 0.0,
                        "throughput": 0.0,
                        "cycle_time": 0.0,
                    },
                }

                # If production line is running, add some synthetic metrics
                if prod_line_status == "running" and "production_line" in state:
                    prod_line = state["production_line"]

                    # Scale metrics based on position in process
                    position_factor = (stages[stage_name]["position"] + 1) / len(
                        self.process_stages
                    )

                    # Add scaled metrics
                    stages[stage_name]["metrics"]["yield_rate"] = (
                        90.0 + position_factor * 5.0
                    )
                    stages[stage_name]["metrics"]["throughput"] = (
                        prod_line.get("production_rate", 0.0) * position_factor
                    )
                    stages[stage_name]["metrics"]["cycle_time"] = 30.0 * (
                        1.0 - position_factor * 0.2
                    )

        return stages

    def get_process_flow(self) -> List[Dict[str, Any]]:
        """
        Get process flow information.

        Returns:
            List[Dict[str, Any]]: Process flow data
        """
        # Initialize process flow
        process_flow = []

        # Use process stages from configuration
        if self.process_stages:
            # Sort stages by position
            sorted_stages = sorted(
                self.process_stages.items(), key=lambda x: x[1].get("position", 0)
            )

            # Create process flow
            for stage_name, stage_config in sorted_stages:
                process_flow.append(
                    {
                        "id": stage_name,
                        "name": stage_config.get("name", stage_name),
                        "description": stage_config.get("description", ""),
                        "position": stage_config.get("position", 0),
                        "inputs": stage_config.get("inputs", []),
                        "outputs": stage_config.get("outputs", []),
                    }
                )

        # If no configuration, create generic process flow
        if not process_flow:
            process_flow = [
                {
                    "id": "raw_materials",
                    "name": "Raw Materials",
                    "description": "Raw material preparation",
                    "position": 0,
                    "inputs": [],
                    "outputs": ["prepared_materials"],
                },
                {
                    "id": "manufacturing",
                    "name": "Manufacturing",
                    "description": "Core manufacturing process",
                    "position": 1,
                    "inputs": ["prepared_materials"],
                    "outputs": ["products"],
                },
                {
                    "id": "quality_control",
                    "name": "Quality Control",
                    "description": "Quality inspection and testing",
                    "position": 2,
                    "inputs": ["products"],
                    "outputs": ["verified_products"],
                },
                {
                    "id": "packaging",
                    "name": "Packaging",
                    "description": "Product packaging",
                    "position": 3,
                    "inputs": ["verified_products"],
                    "outputs": ["packaged_products"],
                },
            ]

        return process_flow

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
        Handle process panel commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "get_process_data":
            process_data = self._get_process_data()
            return {"handled": True, "success": True, "process_data": process_data}

        elif command == "get_process_flow":
            process_flow = self.get_process_flow()
            return {"handled": True, "success": True, "process_flow": process_flow}

        elif command == "select_process":
            process_id = params.get("process_id")
            if process_id:
                self.state.set_selected_process(process_id)
                return {"handled": True, "success": True}
            return {
                "handled": True,
                "success": False,
                "error": "Missing process_id parameter",
            }

        elif command == "refresh_process":
            # Invalidate cache
            self._last_update = datetime.datetime.min
            process_data = self._get_process_data()
            return {"handled": True, "success": True, "process_data": process_data}

        # Not a process panel command
        return {"handled": False}


# Create global instance
process_panel = ProcessPanel()
