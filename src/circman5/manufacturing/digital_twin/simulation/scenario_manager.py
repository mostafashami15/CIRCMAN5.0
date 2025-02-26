# src/circman5/manufacturing/digital_twin/simulation/scenario_manager.py
"""
Scenario Manager for CIRCMAN5.0 Digital Twin.

This module provides functionality for creating, managing, and comparing
simulation scenarios for what-if analysis.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import datetime
from pathlib import Path
import json
import copy

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService


class SimulationScenario:
    """
    Represents a simulation scenario with specific parameters and results.

    Attributes:
        name: Name of the scenario
        description: Description of the scenario
        parameters: Dictionary of scenario parameters
        results: List of simulation states (results)
        created: Timestamp when scenario was created
    """

    def __init__(self, name: str, parameters: Dict[str, Any], description: str = ""):
        """
        Initialize a simulation scenario.

        Args:
            name: Name of the scenario
            parameters: Dictionary of scenario parameters
            description: Optional description of the scenario
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.results: List[Dict[str, Any]] = []
        self.created = datetime.datetime.now().isoformat()
        self.metrics: Dict[str, float] = {}

    def set_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Set the simulation results for this scenario.

        Args:
            results: List of simulation states
        """
        self.results = results

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for this scenario.

        Returns:
            Dict[str, float]: Dictionary of calculated metrics
        """
        if not self.results:
            return {}

        # Extract final state for metric calculation
        final_state = self.results[-1]

        # Initialize metrics dictionary
        metrics = {}

        # Production metrics
        if "production_line" in final_state:
            prod_line = final_state["production_line"]

            if "production_rate" in prod_line:
                metrics["production_rate"] = prod_line["production_rate"]

            if (
                "energy_consumption" in prod_line
                and prod_line["energy_consumption"] > 0
            ):
                if "production_rate" in prod_line:
                    metrics["energy_efficiency"] = (
                        prod_line["production_rate"] / prod_line["energy_consumption"]
                    )

        # Material metrics
        if "materials" in final_state:
            materials = final_state["materials"]

            # Aggregate material inventory
            total_inventory = 0
            for material_name, material_data in materials.items():
                if "inventory" in material_data:
                    total_inventory += material_data["inventory"]

            metrics["total_inventory"] = total_inventory

            # Calculate material quality if available
            quality_sum = 0
            quality_count = 0
            for material_name, material_data in materials.items():
                if "quality" in material_data:
                    quality_sum += material_data["quality"]
                    quality_count += 1

            if quality_count > 0:
                metrics["average_material_quality"] = quality_sum / quality_count

        # Store metrics
        self.metrics = metrics
        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert scenario to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the scenario
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "created": self.created,
            "metrics": self.metrics,
            "results_length": len(self.results),
        }


class ScenarioManager:
    """
    Manages simulation scenarios for what-if analysis.

    Attributes:
        scenarios: Dictionary of scenarios by name
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the scenario manager."""
        self.scenarios: Dict[str, SimulationScenario] = {}
        self.logger = setup_logger("scenario_manager")

        # Load configuration from constants service
        self.constants = ConstantsService()
        self.scenario_config = self.constants.get_digital_twin_config().get(
            "SCENARIO_MANAGEMENT", {}
        )

    def create_scenario(
        self, name: str, parameters: Dict[str, Any], description: str = ""
    ) -> SimulationScenario:
        """
        Create a new simulation scenario.

        Args:
            name: Name for the scenario
            parameters: Dictionary of scenario parameters
            description: Optional description of the scenario

        Returns:
            SimulationScenario: Created scenario

        Raises:
            ValueError: If a scenario with the given name already exists
        """
        if name in self.scenarios:
            raise ValueError(f"Scenario with name '{name}' already exists")

        scenario = SimulationScenario(name, parameters, description)
        self.scenarios[name] = scenario

        self.logger.info(f"Created scenario: {name}")
        return scenario

    def get_scenario(self, name: str) -> SimulationScenario:
        """
        Get a scenario by name.

        Args:
            name: Name of the scenario

        Returns:
            SimulationScenario: The requested scenario

        Raises:
            KeyError: If no scenario with the given name exists
        """
        if name not in self.scenarios:
            raise KeyError(f"No scenario with name '{name}' exists")

        return self.scenarios[name]

    def compare_scenarios(
        self, scenario_names: List[str], metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple scenarios based on selected metrics.

        Args:
            scenario_names: List of scenario names to compare
            metrics: Optional list of metric names to compare (all if None)

        Returns:
            Dict[str, Dict[str, float]]: Comparison results by scenario and metric
        """
        comparison = {}

        for name in scenario_names:
            try:
                scenario = self.get_scenario(name)

                # Calculate metrics if not already done
                if not scenario.metrics:
                    scenario.calculate_metrics()

                # Filter metrics if specified
                if metrics:
                    comparison[name] = {
                        metric: scenario.metrics.get(metric, 0.0)
                        for metric in metrics
                        if metric in scenario.metrics
                    }
                else:
                    comparison[name] = scenario.metrics

            except KeyError:
                self.logger.warning(
                    f"Scenario '{name}' not found, skipping in comparison"
                )

        return comparison

    def save_scenarios(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Save all scenarios to a file.

        Args:
            file_path: Optional path to save scenarios. If not provided, uses results_manager.

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Convert scenarios to dictionaries
            serializable_scenarios = {
                name: scenario.to_dict() for name, scenario in self.scenarios.items()
            }

            if file_path:
                # Use provided path
                save_path = Path(file_path)
                save_dir = save_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)

                with open(save_path, "w") as f:
                    json.dump(serializable_scenarios, f, indent=2)

                self.logger.info(f"Scenarios saved to {save_path}")
            else:
                # Use results_manager
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_scenarios_{timestamp_str}.json"
                temp_path = Path(filename)

                with open(temp_path, "w") as f:
                    json.dump(serializable_scenarios, f, indent=2)

                results_manager.save_file(temp_path, "digital_twin")
                temp_path.unlink()  # Clean up temporary file

                self.logger.info(f"Scenarios saved to results manager: {filename}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save scenarios: {str(e)}")
            return False

    def load_scenarios(self, file_path: Union[str, Path]) -> bool:
        """
        Load scenarios from a file.

        Args:
            file_path: Path to load scenarios from

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

            # Load scenarios from file
            with open(path, "r") as f:
                loaded_scenarios = json.load(f)

            # Convert dictionaries to SimulationScenario objects
            for name, scenario_dict in loaded_scenarios.items():
                scenario = SimulationScenario(
                    name=scenario_dict["name"],
                    parameters=scenario_dict["parameters"],
                    description=scenario_dict.get("description", ""),
                )
                scenario.created = scenario_dict["created"]
                scenario.metrics = scenario_dict.get("metrics", {})

                # Note: Results are not saved/loaded due to potentially large size
                self.scenarios[name] = scenario

            self.logger.info(f"Loaded {len(loaded_scenarios)} scenarios from {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load scenarios: {str(e)}")
            return False
