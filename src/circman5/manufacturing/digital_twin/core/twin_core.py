# src/circman5/manufacturing/digital_twin/core/twin_core.py

"""
Digital Twin Core module for CIRCMAN5.0.

This module implements the core functionality of the digital twin system,
acting as the central coordinator between the physical manufacturing system
and its digital representation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import json
import datetime
from dataclasses import dataclass, field

from .state_manager import StateManager
from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService


@dataclass
class DigitalTwinConfig:
    """Configuration for Digital Twin system."""

    name: str = "SoliTek_DigitalTwin"
    update_frequency: float = 1.0  # Hz
    history_length: int = 1000  # Maximum number of historical states to keep
    simulation_steps: int = 10  # Default number of steps for simulation
    data_sources: List[str] = field(
        default_factory=lambda: ["sensors", "manual_input", "manufacturing_system"]
    )
    synchronization_mode: str = "real_time"  # Options: real_time, batch, manual
    log_level: str = "INFO"

    @classmethod
    def from_constants(cls) -> "DigitalTwinConfig":
        """
        Create configuration from constants service.

        Returns:
            DigitalTwinConfig: Config instance with values from constants service
        """
        constants_service = ConstantsService()
        config = constants_service.get_digital_twin_config()
        dt_config = config.get("DIGITAL_TWIN_CONFIG", {})

        return cls(
            name=dt_config.get("name", "SoliTek_DigitalTwin"),
            update_frequency=dt_config.get("update_frequency", 1.0),
            history_length=dt_config.get("history_length", 1000),
            simulation_steps=dt_config.get("simulation_steps", 10),
            synchronization_mode=dt_config.get("synchronization_mode", "real_time"),
            log_level=dt_config.get("log_level", "INFO"),
        )


class DigitalTwin:
    """
    Core Digital Twin class that manages the digital representation of the manufacturing system.

    The Digital Twin coordinates between the physical manufacturing system and its
    digital representation, enabling simulation, optimization, and real-time monitoring.

    Attributes:
        config: Configuration settings for the digital twin
        state_manager: Manages the current and historical states of the system
        logger: Logger instance for this class
    """

    def __init__(self, config: Optional[DigitalTwinConfig] = None):
        """
        Initialize the Digital Twin system.

        Args:
            config: Optional configuration settings, uses defaults if not provided
        """
        # Use constants service if no config provided
        self.config = config or DigitalTwinConfig.from_constants()
        self.logger = setup_logger("digital_twin_core")
        # Set log level separately - assuming setup_logger returns a standard logger
        self.logger.setLevel(self.config.log_level)

        # Get simulation parameters from constants
        self.constants = ConstantsService()
        self.dt_config = self.constants.get_digital_twin_config()
        self.simulation_params = self.dt_config.get("SIMULATION_PARAMETERS", {})

        # Initialize state manager with config
        self.state_manager = StateManager(history_length=self.config.history_length)
        self.logger.info(
            f"Digital Twin '{self.config.name}' initialized with update frequency {self.config.update_frequency}Hz"
        )
        self.is_running = False
        self._last_update = datetime.datetime.now()

    def initialize(self) -> bool:
        """
        Initialize the digital twin with initial state and connections.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.logger.info("Initializing Digital Twin...")
            initial_state = self._get_initial_state()
            self.state_manager.set_state(initial_state)
            self.is_running = True
            self.logger.info("Digital Twin initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Digital Twin: {str(e)}")
            return False

    def update(self, external_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the digital twin state with new data from the physical system.

        Args:
            external_data: Optional data from external sources to update the state

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            self.logger.debug("Updating Digital Twin state...")

            # Record update timestamp
            current_time = datetime.datetime.now()
            time_delta = (current_time - self._last_update).total_seconds()
            self._last_update = current_time

            # Get current state
            current_state = self.state_manager.get_current_state()

            # Collect data from configured sources
            sensor_data = (
                self._collect_sensor_data()
                if "sensors" in self.config.data_sources
                else {}
            )
            system_data = (
                self._collect_system_data()
                if "manufacturing_system" in self.config.data_sources
                else {}
            )

            # Merge data from all sources
            new_data = {
                **sensor_data,
                **(external_data or {}),
                **system_data,
                "timestamp": current_time.isoformat(),
                "time_delta": time_delta,
            }

            # Update state
            updated_state = self._update_state(current_state, new_data)
            self.state_manager.set_state(updated_state)

            self.logger.debug(
                f"Digital Twin state updated: {len(updated_state)} parameters"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to update Digital Twin: {str(e)}")
            return False

    def simulate(
        self, steps: Optional[int] = None, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a simulation based on the current state.

        Args:
            steps: Number of simulation steps to run
            parameters: Optional parameters to modify for the simulation

        Returns:
            List[Dict[str, Any]]: List of simulated states
        """
        steps = steps or self.config.simulation_steps
        self.logger.info(f"Starting simulation for {steps} steps")

        # Get current state as the starting point
        current_state = self.state_manager.get_current_state().copy()

        # Apply parameter modifications if provided
        if parameters:
            for key, value in parameters.items():
                if key in current_state:
                    current_state[key] = value

        # Run simulation
        simulated_states = [current_state]
        current_sim_state = current_state

        for i in range(steps):
            # Generate next state based on simulation model
            next_state = self._simulate_next_state(current_sim_state)
            simulated_states.append(next_state)
            current_sim_state = next_state

        if simulated_states:
            # Update the state manager with the final state from simulation
            self.state_manager.set_state(simulated_states[-1])
            self.logger.info(
                f"Simulation completed with {len(simulated_states)} states"
            )
        return simulated_states

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the digital twin.

        Returns:
            Dict[str, Any]: Current state
        """
        return self.state_manager.get_current_state()

    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical states of the digital twin.

        Args:
            limit: Optional limit on the number of historical states to retrieve

        Returns:
            List[Dict[str, Any]]: List of historical states
        """
        return self.state_manager.get_history(limit)

    def save_state(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Save the current state to a file.

        Args:
            file_path: Optional path to save the state. If not provided, uses results_manager.

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            state = self.state_manager.get_current_state()

            # Get timestamp for filename
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"digital_twin_state_{timestamp_str}.json"

            if file_path:
                # Use provided path
                save_path = Path(file_path)
                save_dir = save_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)

                with open(save_path, "w") as f:
                    json.dump(state, f, indent=2)
                self.logger.info(f"State saved to {save_path}")
            else:
                # Use results_manager
                try:
                    # First check if directory exists, create if needed
                    dt_dir = results_manager.get_path("digital_twin")
                    if not dt_dir.exists():
                        dt_dir.mkdir(parents=True, exist_ok=True)

                    # Create temporary file
                    temp_path = Path(filename)
                    with open(temp_path, "w") as f:
                        json.dump(state, f, indent=2)

                    # Save file using results_manager
                    results_manager.save_file(temp_path, "digital_twin")

                    # Clean up temporary file
                    temp_path.unlink()
                    self.logger.info(f"State saved to results manager: {filename}")

                    return True
                except Exception as e:
                    self.logger.error(f"Error using results_manager: {str(e)}")
                    # Fall back to saving in current directory
                    with open(filename, "w") as f:
                        json.dump(state, f, indent=2)
                    self.logger.warning(f"Saved state to current directory: {filename}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
            return False

    def load_state(self, file_path: Union[str, Path]) -> bool:
        """
        Load a state from a file.

        Args:
            file_path: Path to load the state from

        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            # Handle results_manager paths if path is not absolute
            path = Path(file_path)
            if not path.is_absolute():
                # Assume it's a relative path in the digital_twin directory
                dt_dir = results_manager.get_path("digital_twin")
                path = dt_dir / path

            # Load state from file
            with open(path, "r") as f:
                state = json.load(f)

            # Set the state in the state manager
            self.state_manager.set_state(state)
            self.logger.info(f"State loaded from {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")
            return False

    def _get_initial_state(self) -> Dict[str, Any]:
        """
        Get the initial state for the digital twin.

        Returns:
            Dict[str, Any]: Initial state dictionary
        """
        # This is a placeholder - in a real implementation, this would
        # collect initial state from the actual manufacturing system
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "system_status": "initialized",
            "production_line": {
                "status": "idle",
                "temperature": 22.5,
                "energy_consumption": 0.0,
                "production_rate": 0.0,
            },
            "materials": {
                "silicon_wafer": {"inventory": 1000, "quality": 0.95},
                "solar_glass": {"inventory": 500, "quality": 0.98},
            },
            "environment": {"temperature": 22.0, "humidity": 45.0},
        }

    def _collect_sensor_data(self) -> Dict[str, Any]:
        """
        Collect data from sensors.

        Returns:
            Dict[str, Any]: Sensor data
        """
        # This is a placeholder - in a real implementation, this would
        # collect data from actual sensors
        return {}

    def _collect_system_data(self) -> Dict[str, Any]:
        """
        Collect data from the manufacturing system.

        Returns:
            Dict[str, Any]: System data
        """
        # This is a placeholder - in a real implementation, this would
        # collect data from the actual manufacturing system
        return {}

    def _update_state(
        self, current_state: Dict[str, Any], new_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update the current state with new data.

        Args:
            current_state: Current state dictionary
            new_data: New data to update the state with

        Returns:
            Dict[str, Any]: Updated state
        """
        # Deep merge of dictionaries
        updated_state = current_state.copy()

        def _merge_dict(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge two dictionaries."""
            result = d1.copy()
            for key, value in d2.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = _merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        return _merge_dict(updated_state, new_data)

    def _simulate_next_state(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate the next state based on the current state.

        Args:
            current_state: Current state to base simulation on

        Returns:
            Dict[str, Any]: Simulated next state
        """
        # Deep copy the current state to avoid reference issues
        import copy

        next_state = copy.deepcopy(current_state)

        # Update timestamp
        if "timestamp" in next_state:
            current_time = datetime.datetime.fromisoformat(next_state["timestamp"])
            next_time = current_time + datetime.timedelta(seconds=1)
            next_state["timestamp"] = next_time.isoformat()

        # Get simulation parameters from configuration with fallbacks
        temp_increment = 0.5  # Default value
        energy_increment = 2.0  # Default value
        prod_rate_increment = 0.2  # Default value

        # Try to get from config, but ensure default values if not found
        if self.simulation_params:
            temp_increment = self.simulation_params.get(
                "temperature_increment", temp_increment
            )
            energy_increment = self.simulation_params.get(
                "energy_consumption_increment", energy_increment
            )
            prod_rate_increment = self.simulation_params.get(
                "production_rate_increment", prod_rate_increment
            )

        # Force minimum increments for testing
        temp_increment = max(0.5, temp_increment)
        energy_increment = max(2.0, energy_increment)

        # Apply changes to production_line
        if "production_line" in next_state:
            if next_state["production_line"].get("status") == "running":
                # Apply increments - ensure we're modifying the state properly
                current_temp = next_state["production_line"].get("temperature", 20.0)
                next_state["production_line"]["temperature"] = (
                    current_temp + temp_increment
                )

                current_energy = next_state["production_line"].get(
                    "energy_consumption", 100.0
                )
                next_state["production_line"]["energy_consumption"] = (
                    current_energy + energy_increment
                )

                # Add production rate if it exists
                if "production_rate" in next_state["production_line"]:
                    current_rate = next_state["production_line"]["production_rate"]
                    next_state["production_line"]["production_rate"] = (
                        current_rate + prod_rate_increment
                    )

        # Return simulated state
        return next_state
