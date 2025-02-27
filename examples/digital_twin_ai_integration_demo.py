# examples/improved_digital_twin_ai_demo.py

"""
Improved Digital Twin AI Integration Demo for CIRCMAN5.0.

This script demonstrates how to connect the digital twin with AI optimization components,
run optimizations, and apply the results back to the digital twin with more realistic values.
"""

import time
import json
from pathlib import Path

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration
from circman5.manufacturing.digital_twin.visualization.twin_visualizer import (
    TwinVisualizer,
)
from circman5.manufacturing.optimization.model import ManufacturingModel
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer
from circman5.utils.results_manager import results_manager


def main():
    """Run the improved Digital Twin AI integration demo."""
    print("Starting Improved Digital Twin AI Integration Demo")

    # Initialize Digital Twin
    print("Initializing Digital Twin...")
    digital_twin = DigitalTwin()
    digital_twin.initialize()
    print("Digital Twin initialized.")

    # Set more realistic initial state
    print("Setting realistic initial state...")
    digital_twin.update(
        {
            "system_status": "running",
            "production_line": {
                "status": "running",
                "production_rate": 90.0,  # Realistic non-zero value
                "energy_consumption": 50.0,
                "temperature": 24.0,
                "cycle_time": 30.0,
                "efficiency": 0.92,
                "defect_rate": 0.03,
            },
            "materials": {
                "silicon_wafer": {"inventory": 500, "quality": 0.95},
                "solar_glass": {"inventory": 300, "quality": 0.98},
            },
            "environment": {"temperature": 22.0, "humidity": 45.0},
        }
    )
    print("Initial state set.")

    # Create AI integration
    print("Initializing AI integration...")
    model = ManufacturingModel()
    optimizer = ProcessOptimizer(model)
    ai_integration = AIIntegration(digital_twin, model, optimizer)
    print("AI integration initialized.")

    # Train model with synthetic data
    print("Training AI model with synthetic data...")
    import pandas as pd
    import numpy as np

    # Create synthetic production data with more realistic correlations
    np.random.seed(42)
    num_samples = 100

    # Generate base values
    input_values = np.random.uniform(80, 120, num_samples)
    energy_values = np.random.uniform(40, 60, num_samples)
    cycle_times = np.random.uniform(25, 35, num_samples)

    # Create correlated output values (more energy → less output, more input → more output)
    output_values = (
        input_values * 0.9 - energy_values * 0.1 + np.random.normal(0, 5, num_samples)
    )
    output_values = np.maximum(output_values, 0)  # Ensure non-negative

    production_data = pd.DataFrame(
        {
            "batch_id": [f"batch_{i}" for i in range(num_samples)],
            "input_amount": input_values,
            "energy_used": energy_values,
            "cycle_time": cycle_times,
            "output_amount": output_values,
        }
    )

    # Create synthetic quality data with realistic correlations
    efficiency_values = 0.85 + 0.1 * np.random.random(num_samples)
    defect_rates = 0.1 - 0.05 * efficiency_values + 0.02 * np.random.random(num_samples)
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

    # Optimize parameters with explicit constraints - use type casting to satisfy the type checker
    print("Optimizing parameters with constraints...")
    from typing import Dict, Union, Tuple, cast

    constraints_dict = {
        "energy_used": (30.0, 70.0),  # Allow exploring energy range
        "cycle_time": (20.0, 40.0),  # Allow exploring cycle time range
        "defect_rate": (0.01, 0.09),  # Allow exploring defect rate range
    }
    # Use type casting to satisfy the type checker
    constraints = cast(Dict[str, Union[float, Tuple[float, float]]], constraints_dict)
    optimized_params = ai_integration.optimize_parameters(current_params, constraints)
    print(f"Optimized parameters: {json.dumps(optimized_params, indent=2)}")

    # Visualize before optimization
    print("Generating pre-optimization visualization...")
    visualizer = TwinVisualizer(digital_twin.state_manager)
    visualizer.visualize_current_state()

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
            print(f"  {param}: {value:.2f}%")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
