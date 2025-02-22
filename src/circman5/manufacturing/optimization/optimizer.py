# src/circman5/manufacturing/optimization/optimizer.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Union, TYPE_CHECKING, cast
from scipy.optimize import minimize

from .types import OptimizationResults
from ...utils.logging_config import setup_logger
from ...utils.results_manager import results_manager

if TYPE_CHECKING:
    from .model import ManufacturingModel


class ProcessOptimizer:
    def __init__(self, model: Optional["ManufacturingModel"] = None) -> None:
        """Initialize the process optimizer."""
        self.logger = setup_logger("process_optimizer")

        if model is None:
            from .model import ManufacturingModel

            self.model = ManufacturingModel()
        else:
            self.model = model

    def optimize_process_parameters(
        self,
        current_params: Dict[str, float],
        constraints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Optimize manufacturing process parameters."""
        try:
            # Validate parameters before adding engineered features
            if not current_params:
                raise ValueError("Parameters dictionary cannot be empty")

            if any(v is None for v in current_params.values()):
                raise ValueError("Parameters cannot contain None values")

            # Add default output_amount if not provided
            if "output_amount" not in current_params:
                current_params = current_params.copy()
                input_amount = current_params.get("input_amount")
                if input_amount is None:
                    raise ValueError("input_amount must be provided")
                current_params["output_amount"] = 0.9 * input_amount

            # Add engineered features before validation
            enhanced_params = self.model._add_engineered_features(current_params)
            current_params = cast(Dict[str, float], enhanced_params)

            # Convert parameters to array format
            param_names = list(current_params.keys())
            initial_params = np.array([current_params[k] for k in param_names])

            # Create bounds list
            bounds: List[tuple[float, float]] = []
            for param in param_names:
                if constraints and param in constraints:
                    target = constraints[param]
                    # Create bounds as ±10% of target value
                    bound_min = max(0.0, target * 0.9)
                    bound_max = target * 1.1
                    bounds.append((bound_min, bound_max))
                else:
                    # Default bounds if no constraint provided
                    bounds.append((0.0, float("inf")))

            # Validate inputs
            self._validate_inputs(current_params)

            # Define objective function for scipy.optimize
            def objective(x):
                params = dict(zip(param_names, x))
                prediction = self.model.predict_batch_outcomes(params)
                return -prediction["predicted_output"]  # Negative because we maximize

            # Run optimization
            result = minimize(
                objective,
                initial_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000},
            )

            # Convert result back to dictionary
            optimized_params = dict(zip(param_names, result.x))

            # Calculate improvements
            improvements = {
                k: ((optimized_params[k] - current_params[k]) / current_params[k] * 100)
                for k in current_params
            }

            # Save detailed results
            results = {
                "original_params": current_params,
                "optimized_params": optimized_params,
                "improvement": improvements,
                "optimization_success": result.success,
                "optimization_message": result.message,
                "iterations": result.nit,
                "objective_value": float(-result.fun),  # Convert back to positive
            }

            results_file = "optimization_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            results_manager.save_file(results_file, "lca_results")
            Path(results_file).unlink()

            self.logger.info(
                f"Parameter optimization completed successfully. "
                f"Objective value: {-result.fun:.2f}"
            )
            return optimized_params

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            raise

    def _validate_inputs(
        self,
        current_params: Dict[str, float],
    ) -> None:
        """Validate optimization inputs."""
        if not self.model.is_trained:
            raise ValueError("Model must be trained before optimization")

        if not current_params:
            raise ValueError("Parameters dictionary cannot be empty")

        if any(v is None for v in current_params.values()):
            raise ValueError("Parameters cannot contain None values")

        required_params = set(self.model.config["feature_columns"])
        if not required_params.issubset(current_params.keys()):
            missing = required_params - set(current_params.keys())
            raise ValueError(f"Missing required parameters: {missing}")

    def analyze_optimization_potential(
        self, production_data: pd.DataFrame, quality_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze potential optimizations with enhanced analysis."""
        try:
            # Train model if needed
            if not self.model.is_trained:
                self.model.train_optimization_model(production_data, quality_data)

            # Calculate statistical summaries
            current_params = {
                "input_amount": production_data["input_amount"].mean(),
                "energy_used": production_data["energy_used"].mean(),
                "cycle_time": production_data["cycle_time"].mean(),
                "efficiency": quality_data["efficiency"].mean(),
                "defect_rate": quality_data["defect_rate"].mean(),
                "thickness_uniformity": quality_data["thickness_uniformity"].mean(),
            }

            # Use means as target values for constraints
            constraints = {
                param: float(value) for param, value in current_params.items()
            }

            # Get optimized parameters
            optimized_params = self.optimize_process_parameters(
                current_params, constraints=constraints
            )

            # Calculate detailed improvements
            improvements = {}
            for param in current_params:
                current = current_params[param]
                optimized = optimized_params[param]
                pct_change = (optimized - current) / current * 100

                improvements[param] = {
                    "current_value": current,
                    "optimized_value": optimized,
                    "percent_change": pct_change,
                    "absolute_change": optimized - current,
                }

            # Calculate optimization summary
            summary = {
                "average_improvement": float(
                    np.mean([imp["percent_change"] for imp in improvements.values()])
                ),
                "parameters_improved": sum(
                    1 for imp in improvements.values() if imp["percent_change"] > 0
                ),
                "total_parameters": len(improvements),
            }

            # Save analysis results
            analysis_results = {
                "current_params": {k: float(v) for k, v in current_params.items()},
                "optimized_params": {k: float(v) for k, v in optimized_params.items()},
                "improvements": improvements,
                "summary": summary,
            }

            analysis_file = "optimization_potential.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis_results, f, indent=2)
            results_manager.save_file(analysis_file, "metrics")
            Path(analysis_file).unlink()

            # Log summary
            self.logger.info(
                "Optimization potential analysis completed:\n"
                f"Average improvement: {summary['average_improvement']:.2f}%\n"
                f"Parameters improved: {summary['parameters_improved']}/{summary['total_parameters']}"
            )

            return {
                param: details["percent_change"]
                for param, details in improvements.items()
            }

        except Exception as e:
            self.logger.error(f"Error in optimization potential analysis: {str(e)}")
            raise
