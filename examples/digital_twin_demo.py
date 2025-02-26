#!/usr/bin/env python3
# examples/digital_twin_demo.py

"""
Demo script for CIRCMAN5.0 Digital Twin visualization.

This script demonstrates the Digital Twin visualization capabilities
by loading test data, running simulations, and generating visualizations.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Union, Tuple

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circman5.manufacturing.core import SoliTekManufacturingAnalysis
from src.circman5.test_data_generator import ManufacturingDataGenerator
from src.circman5.utils.results_manager import results_manager


def run_demo():
    """Run the Digital Twin visualization demo."""
    print("===== CIRCMAN5.0 Digital Twin Visualization Demo =====")

    # Get the standard run directory from results_manager
    output_dir = results_manager.get_run_dir()

    # Define standard paths from results_manager
    data_dir = results_manager.get_path("input_data")
    viz_dir = results_manager.get_path("visualizations")
    temp_dir = results_manager.get_path(
        "temp"
    )  # Temporary directory for intermediate files

    print(f"Output will be saved to: {output_dir.absolute()}")

    # Generate test data
    print("\nGenerating test data...")
    generator = ManufacturingDataGenerator(days=5)
    production_data = generator.generate_production_data()
    energy_data = generator.generate_energy_data()
    quality_data = generator.generate_quality_data()
    material_data = generator.generate_material_flow_data()

    # Save data files using results_manager
    print("Saving test data files...")
    for filename, dataframe in {
        "production_data.csv": production_data,
        "energy_data.csv": energy_data,
        "quality_data.csv": quality_data,
        "material_data.csv": material_data,
    }.items():
        temp_path = Path(temp_dir) / filename
        dataframe.to_csv(temp_path, index=False)
        results_manager.save_file(temp_path, "input_data")
        temp_path.unlink()  # Clean up temp file

    # Initialize manufacturing analysis
    print("\nInitializing manufacturing analysis system...")
    analyzer = SoliTekManufacturingAnalysis()

    # Load data using paths from results_manager
    print("Loading test data...")
    analyzer.load_data(
        production_path=str(
            results_manager.get_path("input_data") / "production_data.csv"
        ),
        energy_path=str(results_manager.get_path("input_data") / "energy_data.csv"),
        quality_path=str(results_manager.get_path("input_data") / "quality_data.csv"),
        material_path=str(results_manager.get_path("input_data") / "material_data.csv"),
    )

    # Generate initial visualizations using results_manager
    print("\nGenerating initial Digital Twin visualizations...")
    for filename, generate_func in {
        "initial_state.png": analyzer.generate_digital_twin_visualization,
        "initial_dashboard.png": analyzer.generate_digital_twin_dashboard,
    }.items():
        temp_path = Path(temp_dir) / filename
        generate_func(save_path=temp_path)
        results_manager.save_file(temp_path, "visualizations")
        temp_path.unlink()  # Clean up temp file

    print("Visualizations saved to results_manager's visualizations directory.")

    # Run a simulation
    print("\nRunning manufacturing simulation...")
    parameters = {
        "production_line": {
            "temperature": 24.0,
            "production_rate": 80.0,  # Start with non-zero value
            "energy_consumption": 50.0,  # Start with non-zero value
            "status": "running",  # Explicitly set to running, not idle
        },
        "materials": {
            "silicon_wafer": {"inventory": 1000, "quality": 0.95},
            "solar_glass": {"inventory": 500, "quality": 0.98},
        },
        "environment": {"temperature": 22.0, "humidity": 45.0},
    }

    # Increase simulation steps to see more dynamic behavior
    simulation_result = analyzer.simulate_manufacturing_scenario(
        steps=50, parameters=parameters
    )
    print(f"Simulation completed with {len(simulation_result)} states")

    # Generate post-simulation visualizations using results_manager
    print("\nGenerating post-simulation visualizations...")
    for filename, generate_func in {
        "simulation_state.png": analyzer.generate_digital_twin_visualization,
        "simulation_dashboard.png": analyzer.generate_digital_twin_dashboard,
    }.items():
        temp_path = Path(temp_dir) / filename
        generate_func(save_path=temp_path)
        results_manager.save_file(temp_path, "visualizations")
        temp_path.unlink()  # Clean up temp file

    # Generate historical visualization using results_manager
    print("Generating historical visualizations...")
    temp_path = Path(temp_dir) / "historical_metrics.png"
    analyzer.visualize_digital_twin_history(
        metrics=[
            "production_line.production_rate",
            "production_line.energy_consumption",
            "production_line.temperature",
        ],
        limit=20,
        save_path=temp_path,
    )
    results_manager.save_file(temp_path, "visualizations")
    temp_path.unlink()  # Clean up temp file

    # Try optimizing using the digital twin
    print("\nOptimizing manufacturing parameters using Digital Twin...")
    current_params = {
        "input_amount": 100.0,
        "output_amount": 90.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 20.0,
        "defect_rate": 3.0,
        "thickness_uniformity": 95.0,
    }

    constraints: Dict[str, Union[float, Tuple[float, float]]] = {
        "energy_used": (120.0, 180.0),
        "cycle_time": (45.0, 55.0),
        "efficiency": (19.0, 22.0),
    }

    optimized_params = analyzer.optimize_using_digital_twin(current_params, constraints)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
