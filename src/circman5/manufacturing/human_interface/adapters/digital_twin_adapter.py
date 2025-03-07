# src/circman5/manufacturing/human_interface/adapters/digital_twin_adapter.py

"""
Digital Twin adapter for CIRCMAN5.0 Human-Machine Interface.

This module provides a standardized interface for the human interface system
to interact with the digital twin, managing state retrieval, updates, and
simulation capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
import datetime
import threading

from ....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ...digital_twin.core.twin_core import DigitalTwin
from ...digital_twin.core.state_manager import StateManager
from ...digital_twin.simulation.simulation_engine import SimulationEngine
from ...digital_twin.simulation.scenario_manager import ScenarioManager


class DigitalTwinAdapter:
    """
    Adapter for the Digital Twin system.

    This class provides a simplified and standardized interface for the
    human interface system to interact with the digital twin, managing
    state retrieval, updates, and simulation capabilities.

    Attributes:
        digital_twin: Reference to digital twin
        state_manager: Reference to state manager
        simulation_engine: Reference to simulation engine
        scenario_manager: Reference to scenario manager
        logger: Logger instance for this class
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(DigitalTwinAdapter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the digital twin adapter."""
        if self._initialized:
            return

        self.logger = setup_logger("digital_twin_adapter")
        self.constants = ConstantsService()

        # Get references to digital twin components
        self.digital_twin = DigitalTwin()  # Get instance
        self.state_manager = StateManager()  # Get instance
        self.simulation_engine = SimulationEngine(self.state_manager)  # Create instance
        self.scenario_manager = ScenarioManager()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        self._initialized = True
        self.logger.info("Digital Twin Adapter initialized")

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the digital twin.

        Returns:
            Dict[str, Any]: Current state
        """
        try:
            return self.digital_twin.get_current_state()
        except Exception as e:
            self.logger.error(f"Error getting current state: {str(e)}")

            # Return basic error state
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "system_status": "error",
                "error": str(e),
            }

    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical states of the digital twin.

        Args:
            limit: Optional limit on the number of historical states to retrieve

        Returns:
            List[Dict[str, Any]]: List of historical states
        """
        try:
            return self.digital_twin.get_state_history(limit)
        except Exception as e:
            self.logger.error(f"Error getting state history: {str(e)}")
            return []

    def update_state(self, updates: Dict[str, Any]) -> bool:
        """
        Update the digital twin state.
        """
        try:
            # Make sure digital twin is initialized
            if (
                not hasattr(self.digital_twin, "is_running")
                or not self.digital_twin.is_running
            ):
                self.digital_twin.initialize()

            # Then update state
            result = self.digital_twin.update(updates)

            # Verify update was applied (debug)
            current = self.digital_twin.get_current_state()
            self.logger.debug(f"Updated state: {current}")

            return result
        except Exception as e:
            self.logger.error(f"Error updating state: {str(e)}")
            return False

    def run_simulation(
        self, steps: int = 10, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a simulation using the digital twin.

        Args:
            steps: Number of simulation steps to run
            parameters: Optional parameter modifications for the simulation

        Returns:
            List[Dict[str, Any]]: List of simulated states
        """
        try:
            return self.digital_twin.simulate(steps=steps, parameters=parameters)
        except Exception as e:
            self.logger.error(f"Error running simulation: {str(e)}")

            # Return empty list on error
            return []

    def save_state(self, filename: Optional[str] = None) -> bool:
        """
        Save the current state to a file.

        Args:
            filename: Optional filename to save the state

        Returns:
            bool: True if save was successful
        """
        try:
            return self.digital_twin.save_state(file_path=filename)
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            return False

    def load_state(self, filename: str) -> bool:
        """
        Load a state from a file.

        Args:
            filename: Filename to load the state from

        Returns:
            bool: True if load was successful
        """
        try:
            return self.digital_twin.load_state(file_path=filename)
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            return False

    def save_scenario(
        self, name: str, parameters: Dict[str, Any], description: str = ""
    ) -> bool:
        """
        Save a simulation scenario.

        Args:
            name: Name for the scenario
            parameters: Scenario parameters
            description: Optional scenario description

        Returns:
            bool: True if scenario was saved
        """
        try:
            scenario = self.scenario_manager.create_scenario(
                name=name, parameters=parameters, description=description
            )

            return scenario is not None

        except Exception as e:
            self.logger.error(f"Error saving scenario: {str(e)}")
            return False

    def run_scenario(self, scenario_name: str) -> List[Dict[str, Any]]:
        """
        Run a saved simulation scenario.

        Args:
            scenario_name: Name of the scenario to run

        Returns:
            List[Dict[str, Any]]: List of simulated states
        """
        try:
            # Get scenario
            scenario = self.scenario_manager.get_scenario(scenario_name)
            if not scenario:
                self.logger.warning(f"Scenario not found: {scenario_name}")
                return []

            # Run simulation with scenario parameters
            simulation_results = self.digital_twin.simulate(
                steps=10, parameters=scenario.parameters  # Default steps
            )

            # Update scenario with results
            if simulation_results:
                scenario.set_results(simulation_results)
                scenario.calculate_metrics()

            return simulation_results

        except Exception as e:
            self.logger.error(f"Error running scenario: {str(e)}")
            return []

    def get_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all saved scenarios.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of scenario data
        """
        try:
            scenarios = {}

            for name, scenario in self.scenario_manager.scenarios.items():
                scenarios[name] = scenario.to_dict()

            return scenarios

        except Exception as e:
            self.logger.error(f"Error getting scenarios: {str(e)}")
            return {}

    def compare_scenarios(
        self, scenario_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple scenarios.

        Args:
            scenario_names: List of scenario names to compare

        Returns:
            Dict[str, Dict[str, float]]: Comparison results
        """
        try:
            return self.scenario_manager.compare_scenarios(scenario_names)
        except Exception as e:
            self.logger.error(f"Error comparing scenarios: {str(e)}")
            return {}


# Create global instance
digital_twin_adapter = DigitalTwinAdapter()
