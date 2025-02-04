# demo_script.py

"""
CIRCMAN5.0 System Demonstration
This script showcases the AI-driven manufacturing optimization system's capabilities.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator


def prepare_data(data_generator):
    """Prepare data with consistent column names."""
    production_data = data_generator.generate_production_data()
    quality_data = data_generator.generate_quality_data()
    energy_data = data_generator.generate_energy_data()
    material_flow = data_generator.generate_material_flow_data()

    # No column renaming - use output_amount consistently
    return production_data, quality_data, energy_data, material_flow


def main():
    # Initialize system
    print("Initializing CIRCMAN5.0 Manufacturing Analysis System...")
    analyzer = SoliTekManufacturingAnalysis()

    # Generate synthetic data
    print("\nGenerating synthetic manufacturing data...")
    data_generator = ManufacturingDataGenerator(start_date="2024-01-01", days=30)

    # Load data with consistent column names
    production_data, quality_data, energy_data, material_flow = prepare_data(
        data_generator
    )

    analyzer.production_data = production_data
    analyzer.quality_data = quality_data
    analyzer.energy_data = energy_data
    analyzer.material_flow = material_flow

    print(f"Generated {len(analyzer.production_data)} production records")

    # Train AI optimization model
    print("\nTraining AI optimization model...")
    metrics = analyzer.train_optimization_model()
    print(f"Model training metrics: {metrics}")

    # Optimize process parameters
    print("\nOptimizing manufacturing parameters...")
    current_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    optimized_params = analyzer.optimize_process_parameters(current_params)
    print("\nOptimization Results:")
    print("Current Parameters:")
    for param, value in current_params.items():
        print(f"  {param}: {value:.2f}")
    print("\nOptimized Parameters:")
    for param, value in optimized_params.items():
        print(f"  {param}: {value:.2f}")

    # Generate visualizations
    print("\nGenerating analysis visualizations...")
    metrics = ["production", "energy", "quality", "sustainability"]
    results_dir = Path("tests/results/latest")
    results_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        save_path = results_dir / f"{metric}_analysis.png"
        analyzer.generate_visualization(metric, str(save_path))
        print(f"  Saved {metric} visualization to {save_path}")

    # Generate comprehensive report
    print("\nGenerating comprehensive analysis report...")
    report_path = results_dir / "analysis_report.xlsx"
    analyzer.generate_comprehensive_report(str(report_path))
    print(f"Analysis report saved to {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        sys.exit(1)
