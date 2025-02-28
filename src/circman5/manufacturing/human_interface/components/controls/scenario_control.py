# src/circman5/manufacturing/human_interface/components/controls/scenario_control.py

"""
Scenario control component for CIRCMAN5.0 Human-Machine Interface.

This module implements the scenario control interface, allowing users to create,
manage, and run what-if scenarios through the digital twin simulation system.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
import datetime
import threading
import json
from pathlib import Path

from .....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...core.interface_manager import interface_manager
from ...core.dashboard_manager import dashboard_manager
from ...core.interface_state import interface_state
from ...adapters.event_adapter import EventAdapter
from ...adapters.digital_twin_adapter import digital_twin_adapter
from ....digital_twin.simulation.scenario_manager import ScenarioManager


class ScenarioControl:
    """
    Scenario control component for the Human-Machine Interface.

    This component provides an interface for creating, managing, and running
    what-if scenarios using the digital twin simulation capabilities.

    Attributes:
        state: Reference to interface state
        scenario_manager: Reference to scenario manager
        digital_twin_adapter: Reference to digital twin adapter
        event_adapter: Event adapter for notifications
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the scenario control."""
        self.logger = setup_logger("scenario_control")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.scenario_manager = ScenarioManager()  # Get instance
        self.digital_twin_adapter = digital_twin_adapter
        self.event_adapter = EventAdapter()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        # Cache for scenario data
        self._scenarios_cache: Dict[str, Any] = {}
        self._last_update = datetime.datetime.min
        self._cache_ttl = datetime.timedelta(seconds=30)  # Cache TTL of 30 seconds

        # Register with interface manager
        interface_manager.register_component("scenario_control", self)

        # Register for events
        self.event_adapter.register_callback(
            self._on_system_state_change, category=None  # Register for all categories
        )

        self.logger.info("Scenario Control initialized")

    def get_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available scenarios.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of scenario data
        """
        # Check if cached data is still valid
        now = datetime.datetime.now()
        if now - self._last_update < self._cache_ttl and self._scenarios_cache:
            return self._scenarios_cache.copy()

        # Get scenarios from adapter
        try:
            scenarios = self.digital_twin_adapter.get_all_scenarios()

            # Update cache
            self._scenarios_cache = scenarios
            self._last_update = now

            return scenarios

        except Exception as e:
            self.logger.error(f"Error getting scenarios: {str(e)}")
            return {}

    def create_scenario(
        self, name: str, parameters: Dict[str, Any], description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new simulation scenario.

        Args:
            name: Scenario name
            parameters: Scenario parameters
            description: Optional scenario description

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            # Save scenario using adapter
            success = self.digital_twin_adapter.save_scenario(
                name=name, parameters=parameters, description=description
            )

            if success:
                # Invalidate cache
                self._last_update = datetime.datetime.min

                self.logger.info(f"Created scenario: {name}")
                return {"success": True}
            else:
                self.logger.warning(f"Failed to create scenario: {name}")
                return {"success": False, "error": "Failed to create scenario"}

        except Exception as e:
            self.logger.error(f"Error creating scenario: {str(e)}")
            return {"success": False, "error": str(e)}

    def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Run a simulation scenario.

        Args:
            scenario_name: Name of the scenario to run

        Returns:
            Dict[str, Any]: Result with simulation results and error message if applicable
        """
        try:
            # Run scenario using adapter
            simulation_results = self.digital_twin_adapter.run_scenario(scenario_name)

            if simulation_results:
                self.logger.info(f"Ran scenario: {scenario_name}")
                return {"success": True, "results": simulation_results}
            else:
                self.logger.warning(f"Failed to run scenario: {scenario_name}")
                return {
                    "success": False,
                    "error": "Failed to run scenario",
                    "results": [],
                }

        except Exception as e:
            self.logger.error(f"Error running scenario: {str(e)}")
            return {"success": False, "error": str(e), "results": []}

    def compare_scenarios(self, scenario_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple scenarios.

        Args:
            scenario_names: List of scenario names to compare

        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            # Compare scenarios using adapter
            comparison = self.digital_twin_adapter.compare_scenarios(scenario_names)

            if comparison:
                self.logger.info(f"Compared scenarios: {scenario_names}")
                return {"success": True, "comparison": comparison}
            else:
                self.logger.warning(f"Failed to compare scenarios: {scenario_names}")
                return {
                    "success": False,
                    "error": "Failed to compare scenarios",
                    "comparison": {},
                }

        except Exception as e:
            self.logger.error(f"Error comparing scenarios: {str(e)}")
            return {"success": False, "error": str(e), "comparison": {}}

    def export_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Export a scenario to JSON.

        Args:
            scenario_name: Name of the scenario to export

        Returns:
            Dict[str, Any]: Result with scenario JSON and error message if applicable
        """
        try:
            # Get scenario from cache or adapter
            scenarios = self.get_all_scenarios()

            if scenario_name in scenarios:
                # Get the scenario
                scenario = scenarios[scenario_name]

                # Convert to JSON
                scenario_json = json.dumps(scenario, indent=2)

                self.logger.info(f"Exported scenario: {scenario_name}")
                return {"success": True, "scenario_json": scenario_json}
            else:
                self.logger.warning(f"Scenario not found: {scenario_name}")
                return {
                    "success": False,
                    "error": f"Scenario not found: {scenario_name}",
                    "scenario_json": "",
                }

        except Exception as e:
            self.logger.error(f"Error exporting scenario: {str(e)}")
            return {"success": False, "error": str(e), "scenario_json": ""}

    def import_scenario(self, scenario_json: str) -> Dict[str, Any]:
        """
        Import a scenario from JSON.

        Args:
            scenario_json: JSON string with scenario data

        Returns:
            Dict[str, Any]: Result with success and error message if applicable
        """
        try:
            # Parse JSON
            scenario_data = json.loads(scenario_json)

            # Extract required fields
            name = scenario_data.get("name")
            parameters = scenario_data.get("parameters")
            description = scenario_data.get("description", "")

            if not name or not parameters:
                self.logger.warning("Missing required fields in scenario JSON")
                return {
                    "success": False,
                    "error": "Missing required fields in scenario JSON",
                }

            # Create scenario
            result = self.create_scenario(name, parameters, description)

            if result["success"]:
                self.logger.info(f"Imported scenario: {name}")
                return {"success": True}
            else:
                return result

        except json.JSONDecodeError as jde:
            self.logger.error(f"Error parsing scenario JSON: {str(jde)}")
            return {
                "success": False,
                "error": f"Error parsing scenario JSON: {str(jde)}",
            }
        except Exception as e:
            self.logger.error(f"Error importing scenario: {str(e)}")
            return {"success": False, "error": str(e)}

    def _on_system_state_change(self, event: Any) -> None:
        """
        Handle system state change events.

        Args:
            event: Event data
        """
        # Only invalidate cache for certain events
        # (avoid frequent cache invalidation for all events)
        if hasattr(event, "category") and hasattr(event, "message"):
            if "scenario" in event.message.lower():
                # Invalidate cache
                self._last_update = datetime.datetime.min

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle scenario control commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "get_all_scenarios":
            scenarios = self.get_all_scenarios()
            return {"handled": True, "success": True, "scenarios": scenarios}

        elif command == "create_scenario":
            name = params.get("name")
            parameters = params.get("parameters")
            description = params.get("description", "")

            if not name or not parameters:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing name or parameters",
                }

            result = self.create_scenario(name, parameters, description)
            return {"handled": True, **result}

        elif command == "run_scenario":
            scenario_name = params.get("scenario_name")

            if not scenario_name:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing scenario_name parameter",
                }

            result = self.run_scenario(scenario_name)
            return {"handled": True, **result}

        elif command == "compare_scenarios":
            scenario_names = params.get("scenario_names")

            if not scenario_names or not isinstance(scenario_names, list):
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing or invalid scenario_names parameter",
                }

            result = self.compare_scenarios(scenario_names)
            return {"handled": True, **result}

        elif command == "export_scenario":
            scenario_name = params.get("scenario_name")

            if not scenario_name:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing scenario_name parameter",
                }

            result = self.export_scenario(scenario_name)
            return {"handled": True, **result}

        elif command == "import_scenario":
            scenario_json = params.get("scenario_json")

            if not scenario_json:
                return {
                    "handled": True,
                    "success": False,
                    "error": "Missing scenario_json parameter",
                }

            result = self.import_scenario(scenario_json)
            return {"handled": True, **result}

        # Not a scenario control command
        return {"handled": False}


# Create global instance
scenario_control = ScenarioControl()
