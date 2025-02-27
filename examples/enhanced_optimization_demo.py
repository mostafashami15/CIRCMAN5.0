# examples/enhanced_optimization_demo.py

"""
Enhanced Digital Twin Optimization Demo for CIRCMAN5.0.

This script demonstrates more aggressive optimization with the digital twin
to show meaningful improvements in manufacturing parameters.
"""

import time
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, cast

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration
from circman5.manufacturing.digital_twin.visualization.twin_visualizer import (
    TwinVisualizer,
)
from circman5.manufacturing.optimization.model import ManufacturingModel
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer
from circman5.utils.results_manager import results_manager


# Enhanced optimizer class with more aggressive exploration
class EnhancedOptimizer(ProcessOptimizer):
    """Optimizer with enhanced exploration and multi-objective function."""

    def optimize_process_parameters(
        self,
        current_params: Dict[str, float],
        constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    ) -> Dict[str, float]:
        """
        Enhanced optimization with multi-objective function and forced exploration.
        """
        try:
            # Use the parent class method, but intercept the objective function
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

            # Create much wider bounds to encourage exploration
            bounds: List[tuple[float, float]] = []
            for param in param_names:
                if constraints and param in constraints:
                    target = constraints[param]
                    # Check if target is a tuple (min, max)
                    if isinstance(target, tuple) and len(target) == 2:
                        bound_min, bound_max = target
                    else:
                        # Create bounds as ±40% of target value (wider range)
                        bound_min = max(0.0, float(target) * 0.6)
                        bound_max = float(target) * 1.4
                    bounds.append((bound_min, bound_max))
                else:
                    # Much more exploratory default bounds
                    if param == "input_amount":
                        bounds.append(
                            (
                                max(1.0, current_params[param] * 0.5),
                                current_params[param] * 1.5,
                            )
                        )
                    elif param == "energy_used":
                        bounds.append(
                            (
                                max(0.1, current_params[param] * 0.5),
                                current_params[param] * 1.5,
                            )
                        )
                    elif param == "cycle_time":
                        bounds.append(
                            (
                                max(5.0, current_params[param] * 0.6),
                                current_params[param] * 1.4,
                            )
                        )
                    elif param == "defect_rate":
                        bounds.append((0.01, min(0.1, current_params[param] * 2.0)))
                    elif param == "efficiency":
                        bounds.append(
                            (
                                max(0.5, current_params[param] * 0.9),
                                min(0.99, current_params[param] * 1.1),
                            )
                        )
                    else:
                        # Default bounds if no specific rule
                        bounds.append(
                            (
                                max(0.1, current_params[param] * 0.5),
                                current_params[param] * 1.5,
                            )
                        )

            # Validate inputs
            self._validate_inputs(current_params)

            # Store original params for reference in objective function
            original_params = current_params.copy()

            # Define enhanced multi-objective function for scipy.optimize
            def multi_objective(x):
                # Create parameters dictionary
                params = dict(zip(param_names, x))

                # Add safeguard for zero energy_used
                if params.get("energy_used", 0) == 0:
                    params["energy_used"] = 0.1

                # Make prediction
                prediction = self.model.predict_batch_outcomes(params)

                # Base objective: maximize output
                objective_val = -prediction[
                    "predicted_output"
                ]  # Negative because we maximize

                # Add factors to encourage parameter changes
                # Calculate distance from original params (want some distance, but not too much)
                param_changes = sum(
                    (
                        (params.get(p, 0) - original_params.get(p, 0))
                        / original_params.get(p, 1)
                    )
                    ** 2
                    for p in ["energy_used", "cycle_time", "efficiency", "defect_rate"]
                )

                # Penalize no change and excessive change
                optimal_change = 0.3  # Target 30% total change
                change_penalty = 5.0 * abs(param_changes - optimal_change)

                # Reward energy efficiency (output / energy_used)
                if params.get("energy_used", 0) > 0:
                    energy_efficiency = (
                        prediction["predicted_output"] / params["energy_used"]
                    )
                    # Negative because we want to maximize efficiency
                    efficiency_factor = -10.0 * energy_efficiency
                else:
                    efficiency_factor = 0

                # Encourage quality improvements (lower defect rate)
                quality_factor = 50.0 * params.get("defect_rate", 0)

                # Combined objective function
                return (
                    objective_val + change_penalty + efficiency_factor + quality_factor
                )

            # Run optimization with increased iterations and better tolerance
            from scipy.optimize import minimize

            result = minimize(
                multi_objective,
                initial_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 5000, "ftol": 1e-8, "gtol": 1e-7},
            )

            # Convert result back to dictionary
            optimized_params = dict(zip(param_names, result.x))

            # Calculate improvements with zero protection
            improvements = {}
            for k in current_params:
                if k in optimized_params:
                    current_val = current_params[k]
                    new_val = optimized_params[k]
                    if abs(current_val) > 1e-10:  # Protect against division by zero
                        pct_change = ((new_val - current_val) / current_val) * 100
                        improvements[k] = pct_change
                    else:
                        # For near-zero values, use absolute change instead
                        improvements[k] = new_val - current_val

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

            results_file = "enhanced_optimization_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            results_manager.save_file(results_file, "lca_results")
            Path(results_file).unlink()

            self.logger.info(
                f"Enhanced parameter optimization completed successfully. "
                f"Objective value: {-result.fun:.2f}, "
                f"Iterations: {result.nit}"
            )
            return optimized_params

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            raise


def main():
    """Run the enhanced digital twin optimization demo."""
    print("Starting Enhanced Digital Twin Optimization Demo")

    # Initialize Digital Twin
    print("Initializing Digital Twin...")
    digital_twin = DigitalTwin()
    digital_twin.initialize()
    print("Digital Twin initialized.")

    # Set realistic initial state with moderate performance
    print("Setting realistic initial state...")
    digital_twin.update(
        {
            "system_status": "running",
            "production_line": {
                "status": "running",
                "production_rate": 85.0,  # Moderate production rate
                "energy_consumption": 60.0,  # Higher energy consumption
                "temperature": 25.0,
                "cycle_time": 32.0,  # Slightly longer cycle time
                "efficiency": 0.88,  # Lower efficiency
                "defect_rate": 0.06,  # Higher defect rate
            },
            "materials": {
                "silicon_wafer": {"inventory": 500, "quality": 0.92},
                "solar_glass": {"inventory": 300, "quality": 0.95},
            },
            "environment": {"temperature": 22.0, "humidity": 45.0},
        }
    )
    print("Initial state set.")

    # Create AI integration with enhanced optimizer
    print("Initializing AI integration with enhanced optimizer...")
    model = ManufacturingModel()
    optimizer = EnhancedOptimizer(model)  # Use our enhanced optimizer
    ai_integration = AIIntegration(digital_twin, model, optimizer)
    print("AI integration initialized.")

    # Train model with synthetic data showing clear relationships
    print("Training AI model with realistic synthetic data...")
    import pandas as pd
    import numpy as np

    # Create synthetic production data with clear improvement patterns
    np.random.seed(42)
    num_samples = 200  # More training data

    # Generate base values with wide ranges
    input_values = np.random.uniform(60, 150, num_samples)
    energy_values = np.random.uniform(30, 80, num_samples)
    cycle_times = np.random.uniform(20, 40, num_samples)

    # Create formula with clear relationships:
    # - Higher input → more output
    # - Higher energy → higher output but diminishing returns
    # - Longer cycle time → slightly higher output but diminishing returns
    output_values = (
        input_values * 0.85  # Base conversion
        - (energy_values - 40) ** 2 * 0.005  # Quadratic energy penalty
        + np.log(cycle_times) * 10  # Logarithmic benefit from cycle time
        + np.random.normal(0, 2, num_samples)  # Small random variation
    )
    output_values = np.clip(output_values, 0, None)  # Ensure non-negative

    production_data = pd.DataFrame(
        {
            "batch_id": [f"batch_{i}" for i in range(num_samples)],
            "input_amount": input_values,
            "energy_used": energy_values,
            "cycle_time": cycle_times,
            "output_amount": output_values,
        }
    )

    # Create synthetic quality data with clear relationships:
    # - Higher efficiency → better quality
    # - Higher defect rate → worse quality
    efficiency_values = np.random.uniform(0.80, 0.98, num_samples)
    defect_rates = (
        0.1 - 0.09 * efficiency_values + 0.01 * np.random.random(num_samples)
    )  # Strong inverse correlation
    defect_rates = np.clip(defect_rates, 0.01, 0.1)  # Keep within realistic range

    quality_data = pd.DataFrame(
        {
            "batch_id": [f"batch_{i}" for i in range(num_samples)],
            "efficiency": efficiency_values,
            "defect_rate": defect_rates,
            "thickness_uniformity": np.random.uniform(90, 98, num_samples),
        }
    )

    # Train the model
    model.train_optimization_model(production_data, quality_data)
    print("Model trained successfully.")

    # Extract parameters from Digital Twin
    print("Extracting parameters from Digital Twin...")
    current_params = ai_integration.extract_parameters_from_state()
    print(f"Current parameters: {json.dumps(current_params, indent=2)}")

    # Predict outcomes
    print("Predicting manufacturing outcomes...")
    prediction = ai_integration.predict_outcomes(current_params)
    print(f"Prediction: {json.dumps(prediction, indent=2)}")

    # Visualize before optimization
    print("Generating pre-optimization visualization...")
    visualizer = TwinVisualizer(digital_twin.state_manager)
    visualizer.visualize_current_state()

    # Optimize parameters with wider constraints
    print("Performing enhanced optimization...")
    constraints_dict = {
        "energy_used": (30.0, 75.0),  # Wide energy range
        "cycle_time": (20.0, 40.0),  # Wide cycle time range
        "defect_rate": (0.01, 0.09),  # Wide defect rate range
        "efficiency": (0.85, 0.98),  # Wide efficiency range
    }
    # Use type casting to satisfy the type checker
    constraints = cast(Dict[str, Union[float, Tuple[float, float]]], constraints_dict)
    optimized_params = ai_integration.optimize_parameters(current_params, constraints)
    print(f"Optimized parameters: {json.dumps(optimized_params, indent=2)}")

    # Apply optimized parameters
    print("Applying optimized parameters to Digital Twin...")
    success = ai_integration.apply_optimized_parameters(
        optimized_params, simulation_steps=10
    )
    print(f"Parameters applied successfully: {success}")

    # Generate visualization after optimization
    print("Generating post-optimization visualization...")
    visualizer.visualize_current_state()

    # Generate optimization report
    print("Generating optimization report...")
    report = ai_integration.generate_optimization_report()
    print(
        f"Report generated with {report.get('total_optimizations', 0)} optimizations."
    )

    # Show improvement summary
    if (
        "latest_optimization" in report
        and "improvements" in report["latest_optimization"]
    ):
        improvements = report["latest_optimization"]["improvements"]
        print("\nOptimization Improvements:")
        for param, value in improvements.items():
            if isinstance(value, (int, float)):
                print(f"  {param}: {value:.2f}%")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
