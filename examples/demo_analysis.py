"""
CIRCMAN5.0 Demonstration Script
"""

import pandas as pd
from pathlib import Path
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.results_manager import results_manager


def run_complete_demonstration():
    print("Starting CIRCMAN5.0 Demonstration...")

    # 1. Generate Test Data
    print("\n1. Generating Synthetic Manufacturing Data...")
    data_generator = ManufacturingDataGenerator(days=30)

    # Generate all required datasets
    production_data = data_generator.generate_production_data()
    quality_data = data_generator.generate_quality_data()
    energy_data = data_generator.generate_energy_data()
    material_data = data_generator.generate_material_flow_data()

    # Save generated data
    data_generator.save_generated_data()

    # 2. Initialize Analysis System
    print("\n2. Initializing Manufacturing Analysis System...")
    analyzer = SoliTekManufacturingAnalysis()

    # 3. Load Data with Verification
    print("\n3. Loading Data...")
    production_path = (
        results_manager.get_path("SYNTHETIC_DATA") / "test_production_data.csv"
    )
    quality_path = results_manager.get_path("SYNTHETIC_DATA") / "test_quality_data.csv"
    energy_path = results_manager.get_path("SYNTHETIC_DATA") / "test_energy_data.csv"
    material_path = (
        results_manager.get_path("SYNTHETIC_DATA") / "test_material_data.csv"
    )

    # Verify files exist
    print(f"Verifying data files...")
    for path in [production_path, quality_path, energy_path, material_path]:
        if not path.exists():
            print(f"Missing data file: {path}")
            return
        print(f"Found: {path}")

    # Load data with explicit paths
    analyzer.load_data(
        production_path=str(production_path),
        quality_path=str(quality_path),
        energy_path=str(energy_path),
        material_path=str(material_path),
    )

    # 4. Manufacturing Performance Analysis
    print("\n4. Analyzing Manufacturing Performance...")
    try:
        performance_metrics = analyzer.analyze_manufacturing_performance()
        print("\nEfficiency Metrics:", performance_metrics["efficiency"])
        print("\nQuality Metrics:", performance_metrics["quality"])
        print("\nSustainability Metrics:", performance_metrics["sustainability"])
    except Exception as e:
        print(f"Error in performance analysis: {str(e)}")

    # 5. Process Optimization Analysis
    print("\n5. Analyzing Process Optimization Potential...")
    try:
        optimization_results = analyzer.analyze_optimization_potential()
        print("\nOptimization Potential:", optimization_results)
    except Exception as e:
        print(f"Error in optimization analysis: {str(e)}")

    # 6. Lifecycle Assessment
    print("\n6. Performing Lifecycle Assessment...")
    try:
        lca_results = analyzer.perform_lifecycle_assessment()
        print("\nLCA Results:", lca_results.to_dict())
    except Exception as e:
        print(f"Error in LCA analysis: {str(e)}")

    # 7. Generate Comprehensive Report
    print("\n7. Generating Comprehensive Analysis Report...")
    try:
        analyzer.generate_comprehensive_report()
    except Exception as e:
        print(f"Error generating report: {str(e)}")

    print("\nResults saved in:")
    print(f"Reports: {results_manager.get_path('reports')}")
    print(f"Visualizations: {results_manager.get_path('visualizations')}")
    print(f"LCA Results: {results_manager.get_path('lca_results')}")


if __name__ == "__main__":
    run_complete_demonstration()
