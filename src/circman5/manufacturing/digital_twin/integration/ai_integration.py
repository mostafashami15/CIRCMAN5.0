# src/circman5/manufacturing/digital_twin/integration/ai_integration.py

"""
AI Integration module for CIRCMAN5.0 Digital Twin.

This module handles the integration between the digital twin and AI optimization components,
enabling data exchange, optimization, and application of AI results to the digital twin.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import datetime
import json
from pathlib import Path

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from ..core.twin_core import DigitalTwin
from ..core.state_manager import StateManager
from ...optimization.model import ManufacturingModel
from ...optimization.optimizer import ProcessOptimizer
from ...optimization.types import OptimizationResults, PredictionDict


class AIIntegration:
    """
    Integrates the digital twin with AI optimization components.

    This class handles data extraction from the digital twin for AI processing,
    sends data to optimization models, and applies optimization results back
    to the digital twin.

    Attributes:
        digital_twin: Reference to the DigitalTwin instance
        model: ManufacturingModel instance for predictions
        optimizer: ProcessOptimizer instance for optimization
        constants: ConstantsService for accessing configurations
        logger: Logger instance for this class
    """

    def __init__(
        self,
        digital_twin: "DigitalTwin",  # Use string literal for forward reference
        model: Optional[ManufacturingModel] = None,
        optimizer: Optional[ProcessOptimizer] = None,
    ):
        """
        Initialize the AI integration.

        Args:
            digital_twin: Digital Twin instance to integrate with
            model: Optional ManufacturingModel instance (created if not provided)
            optimizer: Optional ProcessOptimizer instance (created if not provided)
        """
        self.digital_twin = digital_twin
        self.constants = ConstantsService()
        self.logger = setup_logger("ai_integration")

        # Get configuration from constants service
        self.config = self.constants.get_digital_twin_config().get("AI_INTEGRATION", {})
        self.optimization_config = self.constants.get_optimization_config()

        # Initialize model and optimizer
        self.model = model or ManufacturingModel()
        self.optimizer = optimizer or ProcessOptimizer(self.model)

        # Initialize optimization history
        self.optimization_history: List[Dict[str, Any]] = []

        # Set configuration defaults if not provided
        self.default_params = self.config.get(
            "DEFAULT_PARAMETERS",
            {
                "input_amount": 100.0,
                "energy_used": 50.0,
                "cycle_time": 30.0,
                "efficiency": 0.9,
                "defect_rate": 0.05,
                "thickness_uniformity": 95.0,
            },
        )

        self.logger.info("AI Integration initialized")

    def extract_parameters_from_state(
        self, state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Extract relevant parameters from digital twin state for AI optimization.

        Args:
            state: Optional state dictionary (uses current state if None)

        Returns:
            Dict[str, float]: Parameters dictionary for optimization
        """
        # Get current state if not provided
        if state is None:
            state = self.digital_twin.get_current_state()

        # Initialize parameters dictionary
        params: Dict[str, float] = {}

        # Extract production line parameters
        if "production_line" in state:
            prod_line = state["production_line"]

            # Map production line parameters from configuration
            parameter_mapping = self.config.get(
                "PARAMETER_MAPPING",
                {
                    "production_rate": "output_amount",
                    "energy_consumption": "energy_used",
                    "temperature": "temperature",
                    "cycle_time": "cycle_time",
                },
            )

            for state_key, param_key in parameter_mapping.items():
                if state_key in prod_line and isinstance(
                    prod_line[state_key], (int, float)
                ):
                    params[param_key] = float(prod_line[state_key])

        # Extract material parameters
        if "materials" in state:
            materials = state["materials"]

            # Calculate total material input
            total_input = sum(
                float(material_data.get("inventory", 0))
                for material_name, material_data in materials.items()
                if isinstance(material_data, dict)
            )
            params["input_amount"] = total_input

            # Calculate average material quality
            quality_values = [
                float(material_data.get("quality", 0))
                for material_name, material_data in materials.items()
                if isinstance(material_data, dict) and "quality" in material_data
            ]

            if quality_values:
                params["efficiency"] = sum(quality_values) / len(quality_values)
            else:
                params["efficiency"] = 0.9  # Default efficiency

        # Set default parameters if missing, using values from constants service
        for key, default_value in self.default_params.items():
            if key not in params:
                params[key] = default_value

        self.logger.debug(f"Extracted parameters from state: {params}")
        return params

    def predict_outcomes(
        self, parameters: Optional[Dict[str, float]] = None
    ) -> PredictionDict:
        """
        Predict manufacturing outcomes for the current or provided parameters.

        Args:
            parameters: Optional parameters dictionary (extracted from current state if None)

        Returns:
            PredictionDict: Prediction results
        """
        try:
            # Extract parameters from state if not provided
            if parameters is None:
                parameters = self.extract_parameters_from_state()

            # Ensure model is trained
            if not self.model.is_trained:
                self.logger.warning(
                    "Model is not trained, predictions may not be accurate"
                )

            # Make prediction
            prediction = self.model.predict_batch_outcomes(parameters)

            self.logger.info(
                f"Predicted outcome: {prediction['predicted_output']:.2f} with confidence {prediction['confidence_score']:.2f}"
            )
            return prediction

        except Exception as e:
            self.logger.error(f"Error predicting outcomes: {str(e)}")
            raise

    def optimize_parameters(
        self,
        current_params: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    ) -> Dict[str, float]:
        """
        Optimize process parameters for the digital twin.

        Args:
            current_params: Optional current parameters (extracted from current state if None)
            constraints: Optional parameter constraints (can be single values or min/max tuples)

        Returns:
            Dict[str, float]: Optimized parameters
        """
        try:
            # Extract parameters from state if not provided
            if current_params is None:
                current_params = self.extract_parameters_from_state()

            # Make sure model is trained
            if not self.model.is_trained:
                self.logger.warning(
                    "Model is not trained, optimization may not be accurate"
                )

            # Run optimization - use type casting to satisfy the type checker
            from typing import cast

            typed_constraints = cast(
                Optional[Dict[str, Union[float, Tuple[float, float]]]], constraints
            )
            optimized_params = self.optimizer.optimize_process_parameters(
                current_params, typed_constraints
            )

            # Record optimization result
            optimization_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "original_params": current_params,
                "optimized_params": optimized_params,
                "improvements": {
                    k: (
                        (optimized_params.get(k, 0) - current_params.get(k, 0))
                        / current_params.get(k, 1)
                        * 100
                    )
                    for k in current_params
                    if k in optimized_params and current_params.get(k, 0) != 0
                },
            }
            self.optimization_history.append(optimization_record)

            # Save optimization history
            self._save_optimization_history()

            self.logger.info(f"Parameters optimized successfully")
            return optimized_params

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            raise

    def apply_optimized_parameters(
        self, optimized_params: Dict[str, float], simulation_steps: int = 10
    ) -> bool:
        """
        Apply optimized parameters to the digital twin through simulation.

        Args:
            optimized_params: Optimized parameters to apply
            simulation_steps: Number of simulation steps to run

        Returns:
            bool: True if parameters were applied successfully
        """
        try:
            # Convert optimized parameters to state parameters
            state_updates = self._convert_parameters_to_state(optimized_params)

            # Run simulation with updated parameters
            simulation_results = self.digital_twin.simulate(
                steps=simulation_steps, parameters=state_updates
            )

            # Log application results
            if simulation_results:
                final_state = simulation_results[-1]
                self.logger.info(
                    f"Applied optimized parameters to digital twin. "
                    f"Simulated output: {final_state.get('production_line', {}).get('production_rate', 0):.2f}"
                )
                return True
            else:
                self.logger.warning(
                    "No simulation results returned when applying optimized parameters"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error applying optimized parameters: {str(e)}")
            return False

    def _convert_parameters_to_state(
        self, parameters: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Convert optimization parameters to digital twin state format with improved mapping.

        Args:
            parameters: Dictionary of optimization parameters

        Returns:
            Dict[str, Any]: Parameters in digital twin state format
        """
        state_updates: Dict[str, Any] = {"production_line": {}, "materials": {}}

        # Get parameter mapping from configuration
        parameter_mapping = self.config.get(
            "PARAMETER_MAPPING",
            {
                "production_rate": "output_amount",
                "energy_consumption": "energy_used",
                "temperature": "temperature",
                "cycle_time": "cycle_time",
            },
        )

        # Reverse the mapping for setting state values
        reverse_mapping = {v: k for k, v in parameter_mapping.items()}

        # Map optimization parameters to production line parameters
        for param_key, value in parameters.items():
            # Skip engineered features
            if param_key in [
                "efficiency_rate",
                "energy_efficiency",
                "efficiency_quality",
            ]:
                continue

            if param_key in reverse_mapping:
                state_key = reverse_mapping[param_key]
                state_updates["production_line"][state_key] = value

        # Ensure output_amount is properly mapped to production_rate
        if "output_amount" in parameters:
            state_updates["production_line"]["production_rate"] = parameters[
                "output_amount"
            ]

        # Map efficiency and defect rate directly to production line
        if "efficiency" in parameters:
            state_updates["production_line"]["efficiency"] = parameters["efficiency"]

        if "defect_rate" in parameters:
            state_updates["production_line"]["defect_rate"] = parameters["defect_rate"]

        # Set production line status to running
        state_updates["production_line"]["status"] = "running"

        # Update system status
        state_updates["system_status"] = "optimized"

        # Update materials if input_amount was changed
        if "input_amount" in parameters and parameters["input_amount"] > 0:
            # Divide input materials between silicon_wafer and solar_glass
            total_input = parameters["input_amount"]
            state_updates["materials"] = {
                "silicon_wafer": {
                    "inventory": total_input * 0.6,  # 60% silicon wafer
                    "quality": parameters.get("efficiency", 0.9),
                },
                "solar_glass": {
                    "inventory": total_input * 0.4,  # 40% solar glass
                    "quality": parameters.get("efficiency", 0.9)
                    + 0.03,  # Slightly better quality
                },
            }

        return state_updates

    def _save_optimization_history(self) -> None:
        """Save optimization history to file using results_manager."""
        try:
            # Create filename with timestamp
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_history_{timestamp_str}.json"

            # Save to file
            with open(filename, "w") as f:
                json.dump(self.optimization_history, f, indent=2)

            # Save using results_manager
            results_manager.save_file(Path(filename), "lca_results")

            # Clean up temporary file
            Path(filename).unlink()

            self.logger.debug(f"Saved optimization history to results manager")

        except Exception as e:
            self.logger.error(f"Error saving optimization history: {str(e)}")

    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization report.

        Returns:
            Dict[str, Any]: Report data
        """
        if not self.optimization_history:
            return {"error": "No optimization history available"}

        try:
            # Get latest optimization
            latest = self.optimization_history[-1]

            # Calculate average improvements across all optimizations
            all_improvements = []
            for record in self.optimization_history:
                improvements = record.get("improvements", {})
                all_improvements.extend(improvements.values())

            avg_improvement = (
                sum(all_improvements) / len(all_improvements) if all_improvements else 0
            )

            # Generate report
            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_optimizations": len(self.optimization_history),
                "average_improvement": avg_improvement,
                "latest_optimization": {
                    "timestamp": latest.get("timestamp"),
                    "improvements": latest.get("improvements"),
                },
                "parameter_trends": self._calculate_parameter_trends(),
                "current_state": self.digital_twin.get_current_state(),
            }

            # Save report
            report_file = "optimization_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            results_manager.save_file(Path(report_file), "reports")
            Path(report_file).unlink()

            return report

        except Exception as e:
            self.logger.error(f"Error generating optimization report: {str(e)}")
            return {"error": str(e)}

    def _calculate_parameter_trends(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate parameter trends across optimization history.

        Returns:
            Dict[str, Dict[str, float]]: Parameter trends
        """
        if not self.optimization_history or len(self.optimization_history) < 2:
            return {}

        # Get parameters from first and last optimization
        first = self.optimization_history[0]
        last = self.optimization_history[-1]

        first_params = first.get("optimized_params", {})
        last_params = last.get("optimized_params", {})

        # Calculate trends
        trends = {}
        for param in set(first_params.keys()).union(last_params.keys()):
            first_value = first_params.get(param, 0)
            last_value = last_params.get(param, 0)

            if first_value != 0:
                percent_change = (last_value - first_value) / first_value * 100
            else:
                percent_change = 0

            trends[param] = {
                "first_value": first_value,
                "last_value": last_value,
                "absolute_change": last_value - first_value,
                "percent_change": percent_change,
            }

        return trends

    def train_model_from_digital_twin(
        self, history_limit: Optional[int] = None
    ) -> bool:
        """
        Train the AI model using data from digital twin history.

        Args:
            history_limit: Optional limit on number of historical states to use

        Returns:
            bool: True if training was successful
        """
        try:
            # Get historical states
            history = self.digital_twin.get_state_history(history_limit)

            if not history:
                self.logger.warning("No historical states available for training")
                return False

            # Convert to DataFrames
            production_data = []
            quality_data = []

            for state in history:
                batch_id = f"batch_{history.index(state)}"

                # Extract production data
                if "production_line" in state:
                    prod_line = state["production_line"]
                    prod_record = {
                        "batch_id": batch_id,
                        "timestamp": state.get("timestamp", ""),
                        "input_amount": 0,
                        "output_amount": prod_line.get("production_rate", 0),
                        "energy_used": prod_line.get("energy_consumption", 0),
                        "cycle_time": prod_line.get("cycle_time", 30),
                    }

                    # Get material input amount
                    if "materials" in state:
                        materials = state["materials"]
                        input_amount = sum(
                            float(material_data.get("inventory", 0))
                            for material_name, material_data in materials.items()
                            if isinstance(material_data, dict)
                        )
                        prod_record["input_amount"] = input_amount

                    production_data.append(prod_record)

                # Extract quality data
                quality_record = {
                    "batch_id": batch_id,
                    "timestamp": state.get("timestamp", ""),
                    "efficiency": 0.9,  # Default
                    "defect_rate": 0.05,  # Default
                    "thickness_uniformity": 95.0,  # Default
                }

                # Override with actual values if available
                if "production_line" in state:
                    prod_line = state["production_line"]
                    if "efficiency" in prod_line:
                        quality_record["efficiency"] = prod_line["efficiency"]
                    if "defect_rate" in prod_line:
                        quality_record["defect_rate"] = prod_line["defect_rate"]

                # If materials have quality data
                if "materials" in state:
                    materials = state["materials"]
                    quality_values = [
                        float(material_data.get("quality", 0))
                        for material_name, material_data in materials.items()
                        if isinstance(material_data, dict)
                        and "quality" in material_data
                    ]

                    if quality_values:
                        quality_record["efficiency"] = sum(quality_values) / len(
                            quality_values
                        )

                quality_data.append(quality_record)

            # Convert to DataFrames
            production_df = pd.DataFrame(production_data)
            quality_df = pd.DataFrame(quality_data)

            # Train model if enough data is available
            if len(production_df) < 5 or len(quality_df) < 5:
                self.logger.warning(
                    f"Insufficient data for training: {len(production_df)} production records, "
                    f"{len(quality_df)} quality records"
                )
                return False

            # Train model
            metrics = self.model.train_optimization_model(production_df, quality_df)

            self.logger.info(
                f"Model trained successfully from digital twin history. "
                f"R2 score: {metrics['r2']:.4f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error training model from digital twin: {str(e)}")
            return False
