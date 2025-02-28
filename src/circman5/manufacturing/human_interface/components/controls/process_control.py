# src/circman5/manufacturing/human_interface/components/controls/process_control.py

"""
Process control component for CIRCMAN5.0 Human-Machine Interface.

This module implements the process control interface, allowing users to monitor,
start/stop, and adjust manufacturing processes through the digital twin system.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
import datetime
import threading
import time

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.dashboard_manager import dashboard_manager
from ...core.interface_state import interface_state
from ...adapters.event_adapter import EventAdapter
from ....digital_twin.core.twin_core import DigitalTwin


class ProcessControl:
    """
    Process control component for the Human-Machine Interface.

    This component provides an interface for monitoring, starting/stopping,
    and adjusting manufacturing processes through the digital twin system.

    Attributes:
        state: Reference to interface state
        digital_twin: Reference to digital twin
        event_adapter: Event adapter for notifications
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the process control."""
        self.logger = setup_logger("process_control")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.digital_twin = DigitalTwin()  # Get instance
        self.event_adapter = EventAdapter()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        # Get manufacturing configuration
        self.manufacturing_config = self.constants.get_manufacturing_constants()
        self.process_config = self.manufacturing_config.get(
            "MANUFACTURING_PROCESSES", {}
        )

        # Cache for process data
        self._process_cache: Dict[str, Any] = {}
        self._last_update = datetime.datetime.min
        self._cache_ttl = datetime.timedelta(seconds=1)  # Cache TTL of 1 second

        # Register with interface manager
        interface_manager.register_component("process_control", self)

        # Register for events
        self.event_adapter.register_callback(
            self._on_system_state_change, category=None  # Register for all categories
        )

        self.logger.info("Process Control initialized")

    def get_process_status(self) -> Dict[str, Any]:
        """
        Get current status of all processes.

        Returns:
            Dict[str, Any]: Process status data
        """
        # Check if cached data is still valid
        now = datetime.datetime.now()
        if now - self._last_update < self._cache_ttl and self._process_cache:
            return self._process_cache.copy()

        # Get current state from digital twin
        try:
            current_state = self.digital_twin.get_current_state()

            # Extract process status
            process_status = {
                "timestamp": current_state.get("timestamp", now.isoformat()),
                "system_status": current_state.get("system_status", "unknown"),
                "main_process": {},
                "sub_processes": {},
            }

            # Extract production line status (main process)
            if "production_line" in current_state:
                prod_line = current_state["production_line"]
                process_status["main_process"] = {
                    "status": prod_line.get("status", "unknown"),
                    "temperature": prod_line.get("temperature", 0.0),
                    "energy_consumption": prod_line.get("energy_consumption", 0.0),
                    "production_rate": prod_line.get("production_rate", 0.0),
                    "defect_rate": prod_line.get("defect_rate", 0.0),
                    "cycle_time": prod_line.get("cycle_time", 0.0),
                }

            # Extract sub-process status
            if "manufacturing_processes" in current_state:
                process_status["sub_processes"] = current_state[
                    "manufacturing_processes"
                ]
            elif self.process_config:
                # If no explicit process data but we have configuration, create synthetic data
                process_status["sub_processes"] = {}

                # Use production line status to infer sub-process status
                prod_line_status = "idle"
                if "production_line" in current_state:
                    prod_line_status = current_state["production_line"].get(
                        "status", "idle"
                    )

                for process_name, process_config in self.process_config.items():
                    # Infer status based on production line
                    status = "active" if prod_line_status == "running" else "idle"

                    # Create synthetic process status
                    process_status["sub_processes"][process_name] = {
                        "name": process_config.get("name", process_name),
                        "status": status,
                        "metrics": {
                            "temperature": 0.0,
                            "pressure": 0.0,
                            "flow_rate": 0.0,
                        },
                    }

            # Get current control mode from interface state
            process_status["control_mode"] = self.state.get_process_control_mode()

            # Update cache
            self._process_cache = process_status
            self._last_update = now

            return process_status

        except Exception as e:
            self.logger.error(f"Error getting process status: {str(e)}")

            # Return basic error status
            return {
                "timestamp": now.isoformat(),
                "system_status": "error",
                "error": str(e),
                "main_process": {"status": "unknown"},
                "sub_processes": {},
                "control_mode": self.state.get_process_control_mode(),
            }

    def start_process(self, process_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a manufacturing process.

        Args:
            process_id: Optional process ID (uses main process if None)

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            # Get current state
            current_state = self.digital_twin.get_current_state()

            # Create update state
            update_state: Dict[str, Any] = {}

            if process_id is None or process_id == "main_process":
                # Start main production line
                update_state = {
                    "production_line": {"status": "running"},
                    "system_status": "running",
                }
            else:
                # Start specific sub-process
                # Create the nested structure properly
                update_state["manufacturing_processes"] = {}
                update_state["manufacturing_processes"][process_id] = {
                    "status": "active"
                }

                # If all sub-processes will be active, also start main process
                if "manufacturing_processes" in current_state:
                    all_active = True
                    for name, process in current_state[
                        "manufacturing_processes"
                    ].items():
                        if name != process_id and process.get("status") != "active":
                            all_active = False
                            break

                    if all_active:
                        update_state["production_line"] = {"status": "running"}
                        update_state["system_status"] = "running"

            # Update digital twin state
            self.digital_twin.update(update_state)

            # Invalidate cache
            self._last_update = datetime.datetime.min

            self.logger.info(f"Started process: {process_id or 'main_process'}")
            return {"success": True}

        except Exception as e:
            self.logger.error(f"Error starting process: {str(e)}")
            return {"success": False, "error": str(e)}

    def stop_process(self, process_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop a manufacturing process.

        Args:
            process_id: Optional process ID (uses main process if None)

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            # Create update state
            update_state: Dict[str, Any] = {}

            if process_id is None or process_id == "main_process":
                # Stop main production line
                update_state = {
                    "production_line": {"status": "idle"},
                    "system_status": "idle",
                }
            else:
                # Stop specific sub-process
                # Create the nested structure properly
                update_state["manufacturing_processes"] = {}
                update_state["manufacturing_processes"][process_id] = {"status": "idle"}

            # Update digital twin state
            self.digital_twin.update(update_state)

            # Invalidate cache
            self._last_update = datetime.datetime.min

            self.logger.info(f"Stopped process: {process_id or 'main_process'}")
            return {"success": True}

        except Exception as e:
            self.logger.error(f"Error stopping process: {str(e)}")
            return {"success": False, "error": str(e)}

    def set_control_mode(self, mode: str) -> Dict[str, Any]:
        """
        Set process control mode.

        Args:
            mode: Control mode (monitor, manual, automatic)

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            # Set control mode in interface state
            self.state.set_process_control_mode(mode)

            # Invalidate cache
            self._last_update = datetime.datetime.min

            self.logger.info(f"Set control mode to: {mode}")
            return {"success": True}

        except ValueError as ve:
            self.logger.warning(f"Invalid control mode: {mode}")
            return {"success": False, "error": str(ve)}
        except Exception as e:
            self.logger.error(f"Error setting control mode: {str(e)}")
            return {"success": False, "error": str(e)}

    def adjust_process_parameter(
        self, process_id: str, parameter: str, value: Any
    ) -> Dict[str, Any]:
        """
        Adjust a process parameter.

        Args:
            process_id: Process ID
            parameter: Parameter name
            value: New parameter value

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            # Convert string values to appropriate types
            if isinstance(value, str):
                try:
                    # Try to convert to number if possible
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Leave as string if conversion fails
                    pass

            # Create update state
            update_state: Dict[str, Any] = {}

            if process_id == "main_process":
                # Update main production line parameter
                update_state["production_line"] = {parameter: value}
            else:
                # Update sub-process parameter
                update_state["manufacturing_processes"] = {}
                update_state["manufacturing_processes"][process_id] = {parameter: value}

            # Update digital twin state
            self.digital_twin.update(update_state)

            # Invalidate cache
            self._last_update = datetime.datetime.min

            self.logger.info(f"Adjusted parameter: {process_id}.{parameter} = {value}")
            return {"success": True}

        except Exception as e:
            self.logger.error(f"Error adjusting process parameter: {str(e)}")
            return {"success": False, "error": str(e)}

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
        Handle process control commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        # Ensure params is not None
        params = params or {}

        if command == "get_process_status":
            process_status = self.get_process_status()
            return {"handled": True, "success": True, "process_status": process_status}

        elif command == "start_process":
            process_id = params.get("process_id")
            result = self.start_process(process_id)
            return {"handled": True, **result}

        elif command == "stop_process":
            process_id = params.get("process_id")
            result = self.stop_process(process_id)
            return {"handled": True, **result}

        elif command == "set_control_mode":
            mode = params.get("mode")
            if not mode:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing mode parameter",
                }

            result = self.set_control_mode(mode)
            return {"handled": True, **result}

        elif command == "adjust_parameter":
            process_id = params.get("process_id")
            parameter = params.get("parameter")
            value = params.get("value")

            if not process_id or not parameter or value is None:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing process_id, parameter, or value parameter",
                }

            result = self.adjust_process_parameter(process_id, parameter, value)
            return {"handled": True, **result}

        # Not a process control command
        return {"handled": False}


# Create global instance
process_control = ProcessControl()
