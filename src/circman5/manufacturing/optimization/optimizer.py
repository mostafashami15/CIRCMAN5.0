# src/circman5/manufacturing/optimization/optimizer.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Union, TYPE_CHECKING, cast
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
        constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    ) -> Dict[str, float]:
        """
        Optimize manufacturing process parameters.

        Args:
            current_params: Current parameters dictionary
            constraints: Optional constraints, either as single values or (min, max) tuples

        Returns:
            Dict[str, float]: Optimized parameters
        """
        try:
            # Validate parameters before adding engineered features
            if not current_params:
                raise ValueError("Parameters dictionary cannot be empty")

            if any(v is None for v in current_params.values()):
                raise ValueError("Parameters cannot contain None values")

            # Validate constraints upfront if provided
            if constraints:
                for param, constraint in constraints.items():
                    if isinstance(constraint, tuple) and len(constraint) == 2:
                        min_val, max_val = constraint
                        if min_val > max_val:
                            raise ValueError(
                                f"Invalid constraint for {param}: min ({min_val}) > max ({max_val})"
                            )

            # Store original parameter keys (before adding engineered features)
            original_param_keys = set(current_params.keys())

            # Add default output_amount if not provided
            if "output_amount" not in current_params:
                current_params = current_params.copy()
                input_amount = current_params.get("input_amount")
                if input_amount is None:
                    raise ValueError("input_amount must be provided")
                current_params["output_amount"] = 0.9 * input_amount

            # Add engineered features
            enhanced_params = self.model._add_engineered_features(current_params)
            current_params = cast(Dict[str, float], enhanced_params)

            # Only optimize the original parameters, not the derived/engineered features
            # Focus on the parameters in the model's feature_columns
            param_names = [
                param
                for param in self.model.config["feature_columns"]
                if param in original_param_keys
            ]

            # Add output_amount if it's in original parameters
            if (
                "output_amount" in original_param_keys
                and "output_amount" not in param_names
            ):
                param_names.append("output_amount")

            initial_params = np.array([current_params[k] for k in param_names])

            # Create bounds list with improved range to encourage optimization
            bounds: List[tuple[float, float]] = []
            for param in param_names:
                if constraints and param in constraints:
                    target = constraints[param]
                    # Check if target is a tuple (min, max)
                    if isinstance(target, tuple) and len(target) == 2:
                        bound_min, bound_max = target
                        # Min/max was already validated upfront
                    else:
                        # Create wider bounds as ±30% of target value (increased from 20%)
                        bound_min = max(0.0, float(target) * 0.7)
                        bound_max = float(target) * 1.3
                    bounds.append((bound_min, bound_max))
                else:
                    # Significantly wider default bounds to promote exploration
                    if param == "input_amount":
                        bounds.append(
                            (
                                max(1.0, current_params[param] * 0.6),  # Wider range
                                current_params[param] * 1.4,
                            )
                        )
                    elif param == "energy_used":
                        bounds.append(
                            (
                                max(0.1, current_params[param] * 0.6),  # Wider range
                                current_params[param] * 1.4,
                            )
                        )
                    elif param == "cycle_time":
                        bounds.append(
                            (
                                max(5.0, current_params[param] * 0.6),  # Wider range
                                current_params[param] * 1.4,
                            )
                        )
                    elif param == "defect_rate":
                        bounds.append(
                            (0.01, min(0.2, current_params[param] * 2.0))
                        )  # More room for improvement
                    else:
                        # Default bounds if no specific rule - wider range
                        bounds.append(
                            (
                                max(0.1, current_params[param] * 0.6),
                                current_params[param] * 1.4,
                            )
                        )

            # Validate inputs
            self._validate_inputs(
                {
                    key: current_params[key]
                    for key in self.model.config["feature_columns"]
                }
            )

            # Create a perturbed starting point to avoid local minima
            # Add small random noise to initial parameters (within bounds)
            perturbed_params = initial_params.copy()
            for i, (param, (lower, upper)) in enumerate(zip(param_names, bounds)):
                # Add noise but ensure we stay within bounds
                noise_scale = 0.05 * (upper - lower)  # 5% of the parameter range
                noise = np.random.normal(0, noise_scale)
                perturbed_value = initial_params[i] + noise
                # Ensure the perturbed value stays within bounds
                perturbed_params[i] = max(lower, min(upper, perturbed_value))

            # Define objective function with improved gradient handling
            def objective(x):
                # Create parameters dictionary with only the optimized parameters
                opt_params = dict(zip(param_names, x))

                # Ensure we have all required parameters for the model
                model_params = {}
                for key in self.model.config["feature_columns"]:
                    if key in opt_params:
                        model_params[key] = opt_params[key]
                    elif key in current_params:
                        model_params[key] = current_params[key]

                # Add output_amount if it's part of optimization
                if "output_amount" in opt_params:
                    model_params["output_amount"] = opt_params["output_amount"]

                # Add safeguard for zero energy_used
                if model_params.get("energy_used", 0) <= 0.1:
                    model_params["energy_used"] = 0.1

                # Add penalties for extreme values to shape the objective function
                penalty = 0
                for param_name, value in opt_params.items():
                    if param_name == "defect_rate" and value > 1.5:
                        # Penalize high defect rates
                        penalty += 0.1 * (value - 1.5) ** 2
                    elif param_name == "energy_used":
                        # Favor energy efficiency
                        penalty += 0.01 * value / model_params.get("output_amount", 1.0)

                # Make prediction
                prediction = self.model.predict_batch_outcomes(model_params)
                # Return negative of (output minus penalty) because we maximize
                return -(prediction["predicted_output"] - penalty)

            # Try multiple optimization runs with different starting points
            best_result = None
            best_value = float("inf")

            # Try standard optimization first
            result = minimize(
                objective,
                initial_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-6, "gtol": 1e-5},
            )

            if result.fun < best_value:
                best_result = result
                best_value = result.fun

            # Now try with perturbed initial point
            result = minimize(
                objective,
                perturbed_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-6, "gtol": 1e-5},
            )

            if result.fun < best_value:
                best_result = result
                best_value = result.fun

            # Use best result found
            result = best_result if best_result is not None else result

            # Convert result back to dictionary
            optimized_params = dict(zip(param_names, result.x))

            # Start with a copy of current parameters and update with optimized values
            final_params = current_params.copy()
            final_params.update(optimized_params)

            # Recalculate the engineered features
            final_params_enhanced = self.model._add_engineered_features(final_params)

            # Ensure we return a dictionary, not a DataFrame
            if isinstance(final_params_enhanced, pd.DataFrame):
                final_params = {
                    col: float(final_params_enhanced.iloc[0][col])
                    for col in final_params_enhanced.columns
                }
            else:
                final_params = cast(Dict[str, float], final_params_enhanced)

            # Calculate improvements
            improvements = {}
            for k in param_names:  # Only calculate for optimized parameters
                if k in final_params and k in current_params:
                    current_val = current_params[k]
                    new_val = final_params[k]
                    if abs(current_val) > 1e-10:  # Protect against division by zero
                        pct_change = ((new_val - current_val) / current_val) * 100
                        improvements[k] = pct_change
                    else:
                        # For near-zero values, use absolute change
                        improvements[k] = new_val - current_val

            # Save detailed results
            results = {
                "original_params": {
                    k: float(v) for k, v in current_params.items() if k in param_names
                },
                "optimized_params": {
                    k: float(v) for k, v in final_params.items() if k in param_names
                },
                "improvement": improvements,
                "optimization_success": result.success,
                "optimization_message": str(result.message),
                "iterations": int(result.nit),
                "objective_value": float(-result.fun),  # Convert back to positive
            }

            results_file = "optimization_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            results_manager.save_file(results_file, "lca_results")
            Path(results_file).unlink()

            self.logger.info(
                f"Parameter optimization completed successfully. "
                f"Objective value: {-result.fun:.2f}, "
                f"Iterations: {result.nit}"
            )
            return final_params

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
                "input_amount": float(production_data["input_amount"].mean()),
                "energy_used": float(production_data["energy_used"].mean()),
                "cycle_time": float(production_data["cycle_time"].mean()),
                "efficiency": float(quality_data["efficiency"].mean()),
                "defect_rate": float(quality_data["defect_rate"].mean()),
                "thickness_uniformity": float(
                    quality_data["thickness_uniformity"].mean()
                ),
            }

            # Add output_amount
            if "output_amount" in production_data.columns:
                current_params["output_amount"] = float(
                    production_data["output_amount"].mean()
                )
            else:
                # Estimate output amount
                current_params["output_amount"] = 0.9 * current_params["input_amount"]

            # Create proper min/max constraints based on data statistics
            constraints = {}
            for param in current_params:
                if param in production_data.columns:
                    min_val = max(
                        0.8 * current_params[param], float(production_data[param].min())
                    )
                    max_val = min(
                        1.2 * current_params[param], float(production_data[param].max())
                    )
                    constraints[param] = (float(min_val), float(max_val))
                elif param in quality_data.columns:
                    min_val = max(
                        0.8 * current_params[param], float(quality_data[param].min())
                    )
                    max_val = min(
                        1.2 * current_params[param], float(quality_data[param].max())
                    )
                    constraints[param] = (float(min_val), float(max_val))
                else:
                    # For parameters not in the data (like output_amount when estimated)
                    constraints[param] = (
                        float(0.8 * current_params[param]),
                        float(1.2 * current_params[param]),
                    )

                # Ensure min < max
                if constraints[param][0] > constraints[param][1]:
                    raise ValueError(
                        f"Invalid constraint for {param}: min ({constraints[param][0]}) > max ({constraints[param][1]})"
                    )

            # Get optimized parameters
            optimized_params = self.optimize_process_parameters(
                current_params, constraints=constraints
            )

            # Calculate detailed improvements
            improvements = {}
            for param in current_params:
                if param in optimized_params:
                    current = current_params[param]
                    optimized = optimized_params[param]
                    if abs(current) > 1e-10:  # Avoid division by zero
                        pct_change = (optimized - current) / current * 100
                    else:
                        pct_change = 0.0

                    improvements[param] = {
                        "current_value": float(current),
                        "optimized_value": float(optimized),
                        "percent_change": float(pct_change),
                        "absolute_change": float(optimized - current),
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

            # Return simplified improvements for API compatibility
            return {
                param: details["percent_change"]
                for param, details in improvements.items()
            }

        except Exception as e:
            self.logger.error(f"Error in optimization potential analysis: {str(e)}")
            raise
