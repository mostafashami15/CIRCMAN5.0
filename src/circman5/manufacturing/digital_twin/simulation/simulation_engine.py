# src/circman5/manufacturing/digital_twin/simulation/simulation_engine.py
"""
Simulation Engine for CIRCMAN5.0 Digital Twin.

This module implements the core simulation capabilities for the digital twin system,
providing physics-based modeling of manufacturing processes.
"""

import random
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import datetime
from pathlib import Path
import json
import copy

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from ..core.state_manager import StateManager


class SimulationEngine:
    """
    Core simulation engine for the digital twin system.

    The SimulationEngine provides physics-based modeling of manufacturing processes,
    enabling predictive simulation and what-if analysis capabilities.

    Attributes:
        state_manager: Reference to the StateManager for accessing system state
        logger: Logger instance for this class
    """

    def __init__(self, state_manager: StateManager):
        """
        Initialize the simulation engine.

        Args:
            state_manager: StateManager instance to access system state
        """
        self.state_manager = state_manager
        self.logger = setup_logger("simulation_engine")

        # Load simulation parameters from constants service
        self.constants = ConstantsService()
        self.simulation_config = self.constants.get_digital_twin_config().get(
            "SIMULATION_PARAMETERS", {}
        )

        self.logger.info("Simulation Engine initialized")

    def run_simulation(
        self,
        steps: int = 10,
        initial_state: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a simulation for the specified number of steps.

        Args:
            steps: Number of simulation steps to run
            initial_state: Optional custom initial state (uses current state if None)
            parameters: Optional parameter modifications for the simulation

        Returns:
            List[Dict[str, Any]]: List of simulated states
        """
        # Get initial state (current state if not provided)
        if initial_state is None:
            initial_state = self.state_manager.get_current_state()

        # Apply parameter modifications if provided
        current_state = initial_state.copy()
        if parameters:
            # Apply parameters to the current state
            self._apply_parameters(current_state, parameters)

        # Record states
        simulated_states = [current_state]

        # Run simulation steps
        for i in range(steps):
            self.logger.debug(f"Running simulation step {i+1}/{steps}")

            # Generate next state using physics-based models
            next_state = self._simulate_next_state(current_state)

            # Add to results
            simulated_states.append(next_state)
            current_state = next_state

        self.logger.info(f"Completed simulation with {len(simulated_states)} states")
        return simulated_states

    def _apply_parameters(
        self, state: Dict[str, Any], parameters: Dict[str, Any]
    ) -> None:
        """
        Apply parameter modifications to the state.

        Args:
            state: State to modify
            parameters: Parameters to apply
        """
        # Simple direct parameter application for now
        for key, value in parameters.items():
            # Handle nested parameters with dot notation
            if "." in key:
                path = key.split(".")
                current = state
                # Navigate to the nested location
                for p in path[:-1]:
                    if p not in current:
                        current[p] = {}
                    current = current[p]
                # Set the value
                current[path[-1]] = value
            else:
                # Set top-level parameter
                state[key] = value

    def _simulate_next_state(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the next state based on physics-based models."""
        # Create a deep copy of the current state
        next_state = copy.deepcopy(current_state)

        # Update timestamp
        if "timestamp" in next_state:
            current_time = datetime.datetime.fromisoformat(next_state["timestamp"])
            next_time = current_time + datetime.timedelta(seconds=1)
            next_state["timestamp"] = next_time.isoformat()

        # Initialize production line if not present
        if "production_line" not in next_state:
            next_state["production_line"] = {
                "status": "idle",
                "temperature": 22.5,
                "energy_consumption": 0.0,
                "production_rate": 0.0,
            }

        # Randomly change status occasionally (10% chance)
        if random.random() < 0.1:
            current_status = next_state["production_line"].get("status", "idle")
            next_state["production_line"]["status"] = (
                "running" if current_status == "idle" else "idle"
            )

        # Add realistic behavior based on status
        if next_state["production_line"].get("status") == "running":
            # When running, show significant activity
            next_state["production_line"]["production_rate"] = random.uniform(80, 150)
            next_state["production_line"]["energy_consumption"] = random.uniform(
                50, 120
            )
            next_state["production_line"]["temperature"] += random.uniform(-0.5, 1.0)

            # Consume materials when running
            if "materials" in next_state:
                for material_name, material_data in next_state["materials"].items():
                    if isinstance(material_data, dict) and "inventory" in material_data:
                        # Consume material based on production rate
                        consumption = (
                            next_state["production_line"]["production_rate"] * 0.01
                        )
                        material_data["inventory"] = max(
                            0, material_data["inventory"] - consumption
                        )
        else:
            # When idle, show minimal activity
            next_state["production_line"]["production_rate"] = 0.0
            next_state["production_line"]["energy_consumption"] = random.uniform(0, 5)
            # Temperature gradually returns to ambient
            ambient = 22.0
            if "temperature" in next_state["production_line"]:
                next_state["production_line"]["temperature"] += (
                    ambient - next_state["production_line"]["temperature"]
                ) * 0.1

        # Add environmental variations
        if "environment" in next_state:
            next_state["environment"]["temperature"] += random.uniform(-0.2, 0.2)
            next_state["environment"]["humidity"] += random.uniform(-1.0, 1.0)
            # Keep humidity in reasonable range
            if "humidity" in next_state["environment"]:
                next_state["environment"]["humidity"] = max(
                    0, min(100, next_state["environment"]["humidity"])
                )

        return next_state

    def _simulate_production_line(self, state: Dict[str, Any]) -> None:
        if "production_line" not in state:
            return

        prod_line = state["production_line"]

        # Apply changes based on production line status
        if prod_line.get("status") == "running":
            # Temperature gradually approaches target with some fluctuation
            target_temp = self.simulation_config.get("target_temperature", 22.5)
            if "temperature" in prod_line:
                regulation = self.simulation_config.get("temperature_regulation", 0.1)
                random_fluctuation = (random.random() - 0.5) * 0.5
                prod_line["temperature"] += (
                    target_temp - prod_line["temperature"]
                ) * regulation + random_fluctuation

            # Energy consumption increases during operation with some variation
            if "energy_consumption" in prod_line:
                base_increment = self.simulation_config.get(
                    "energy_consumption_increment", 2.0
                )
                variation = random.normalvariate(1.0, 0.2)  # Mean 1.0, stddev 0.2
                prod_line["energy_consumption"] += base_increment * variation

            # Production rate varies based on system conditions
            if "production_rate" in prod_line:
                if "temperature" in prod_line:
                    # Production rate depends on how close temperature is to optimal
                    temp_impact = (
                        1.0 - abs(prod_line["temperature"] - target_temp) / 5.0
                    )
                    temp_impact = max(0.5, min(1.2, temp_impact))

                    base_change = self.simulation_config.get(
                        "production_rate_increment", 0.2
                    )
                    prod_line["production_rate"] *= 1.0 + (
                        base_change * temp_impact - 0.1
                    ) * random.normalvariate(1.0, 0.2)
                    # Ensure production rate stays positive and reasonable
                    prod_line["production_rate"] = max(
                        0.1, prod_line["production_rate"]
                    )
        else:
            # System is idle - energy consumption decreases, temperature normalizes
            if "energy_consumption" in prod_line:
                # Gradual energy decrease during idle time
                prod_line["energy_consumption"] *= 0.95
                prod_line["energy_consumption"] = max(
                    1.0, prod_line["energy_consumption"]
                )

            # Production rate drops to zero during idle time
            if "production_rate" in prod_line:
                prod_line["production_rate"] *= 0.8
                if prod_line["production_rate"] < 0.1:
                    prod_line["production_rate"] = 0.0

    def _simulate_materials(self, state: Dict[str, Any]) -> None:
        """
        Simulate material behavior.

        Args:
            state: State to update with simulation results
        """
        if "materials" not in state:
            return

        materials = state["materials"]

        # Simulate material consumption based on production rate
        if "production_line" in state and "production_rate" in state["production_line"]:
            prod_rate = state["production_line"]["production_rate"]

            for material, properties in materials.items():
                # Decrease inventory based on production rate
                consumption_rate = self.simulation_config.get(
                    f"{material}_consumption_rate", 0.5
                )

                if "inventory" in properties:
                    properties["inventory"] = max(
                        0, properties["inventory"] - prod_rate * consumption_rate
                    )

    def _simulate_environment(self, state: Dict[str, Any]) -> None:
        """
        Simulate environmental conditions.

        Args:
            state: State to update with simulation results
        """
        if "environment" not in state:
            return

        env = state["environment"]

        # Add small random changes to environmental parameters
        if "temperature" in env:
            env["temperature"] += np.random.normal(0, 0.1)  # Small random fluctuation

        if "humidity" in env:
            env["humidity"] += np.random.normal(0, 0.5)  # Small random fluctuation
            # Keep humidity in reasonable range
            env["humidity"] = max(0, min(100, env["humidity"]))
