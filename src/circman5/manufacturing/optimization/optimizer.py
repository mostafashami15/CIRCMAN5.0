# src/circman5/manufacturing/optimization/optimizer.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING, cast
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
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
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

            # Add constraints for engineered and output features
            if constraints is not None:
                constraints = constraints.copy()
                # Add constraints for output_amount
                input_constraint = constraints.get("input_amount", (0.0, float("inf")))
                constraints["output_amount"] = (0.0, input_constraint[1])
                # Add constraints for engineered features
                constraints.update(
                    {
                        "efficiency_rate": (0.0, 1.0),
                        "energy_efficiency": (0.0, 1.0),
                        "efficiency_quality": (0.0, 1.0),
                    }
                )

            # Validate inputs after all features are added
            self._validate_inputs(current_params, constraints)

            # Convert parameters to array format
            param_names = list(current_params.keys())
            initial_params = np.array([current_params[k] for k in param_names])

            # Define optimization bounds
            bounds = self._get_optimization_bounds(param_names, constraints)

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
        constraints: Optional[Dict[str, Tuple[float, float]]],
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

        if constraints:
            for param, (min_val, max_val) in constraints.items():
                if min_val > max_val:
                    raise ValueError(
                        f"Invalid constraint for {param}: "
                        f"min value {min_val} is greater than max value {max_val}"
                    )

    def _get_optimization_bounds(
        self, param_names: list, constraints: Optional[Dict[str, Tuple[float, float]]]
    ) -> list:
        """Get optimization bounds from constraints."""
        default_bounds = (None, None)
        if constraints is None:
            return [default_bounds] * len(param_names)

        return [constraints.get(param, default_bounds) for param in param_names]

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

            # Add statistical bounds as constraints
            constraints = {
                param: (
                    max(data[param].mean() - 2 * data[param].std(), data[param].min()),
                    min(data[param].mean() + 2 * data[param].std(), data[param].max()),
                )
                for param, data in [
                    ("input_amount", production_data),
                    ("energy_used", production_data),
                    ("cycle_time", production_data),
                    ("efficiency", quality_data),
                    ("defect_rate", quality_data),
                    ("thickness_uniformity", quality_data),
                ]
            }

            # Get optimized parameters with statistical constraints
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
                    "within_bounds": (
                        constraints[param][0] <= optimized <= constraints[param][1]
                    ),
                }

            # Calculate optimization summary
            summary = {
                "average_improvement": np.mean(
                    [imp["percent_change"] for imp in improvements.values()]
                ),
                "max_improvement": max(
                    imp["percent_change"] for imp in improvements.values()
                ),
                "min_improvement": min(
                    imp["percent_change"] for imp in improvements.values()
                ),
                "parameters_improved": sum(
                    1 for imp in improvements.values() if imp["percent_change"] > 0
                ),
                "total_parameters": len(improvements),
            }

            # Save enhanced analysis results
            analysis_results = {
                "current_params": {k: float(v) for k, v in current_params.items()},
                "optimized_params": {k: float(v) for k, v in optimized_params.items()},
                "improvements": {
                    k: {
                        "current_value": float(v["current_value"]),
                        "optimized_value": float(v["optimized_value"]),
                        "percent_change": float(v["percent_change"]),
                        "absolute_change": float(v["absolute_change"]),
                        "within_bounds": bool(v["within_bounds"]),
                    }
                    for k, v in improvements.items()
                },
                "summary": {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in summary.items()
                },
                "constraints_used": {
                    param: {"min": float(min_val), "max": float(max_val)}
                    for param, (min_val, max_val) in constraints.items()
                },
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

    def _calculate_parameter_sensitivity(
        self, base_params: Dict[str, float], param_name: str, variation: float = 0.1
    ) -> Dict[str, float]:
        """Calculate sensitivity of output to parameter variations."""
        try:
            base_prediction = self.model.predict_batch_outcomes(base_params)[
                "predicted_output"
            ]

            # Test parameter variations
            test_params = base_params.copy()
            test_params[param_name] *= 1 + variation
            upper_prediction = self.model.predict_batch_outcomes(test_params)[
                "predicted_output"
            ]

            test_params = base_params.copy()
            test_params[param_name] *= 1 - variation
            lower_prediction = self.model.predict_batch_outcomes(test_params)[
                "predicted_output"
            ]

            # Calculate sensitivity metrics
            sensitivity = {
                "parameter": param_name,
                "base_value": base_params[param_name],
                "sensitivity_score": abs(upper_prediction - lower_prediction)
                / (2 * variation * base_prediction),
                "upper_impact": (upper_prediction - base_prediction)
                / base_prediction
                * 100,
                "lower_impact": (lower_prediction - base_prediction)
                / base_prediction
                * 100,
            }

            return sensitivity

        except Exception as e:
            self.logger.error(f"Error calculating parameter sensitivity: {str(e)}")
            raise
