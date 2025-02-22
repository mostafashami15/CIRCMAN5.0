# examples/completed_demo.py
"""
CIRCMAN5.0 Complete Demonstration Script
Demonstrates full system capabilities including AI optimization and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.results_manager import results_manager
from circman5.manufacturing.reporting.optimization_visualizer import (
    OptimizationVisualizer,
)
from circman5.manufacturing.optimization.types import OptimizationResults, MetricsDict
from circman5.manufacturing.reporting.visualizations import ManufacturingVisualizer
from circman5.monitoring import ManufacturingMonitor


def save_synthetic_data(data_generator: ManufacturingDataGenerator) -> None:
    """Generate and save all required synthetic data files."""
    print("\nGenerating synthetic data files...")

    # Generate all required datasets
    synthetic_data = {
        "test_production_data.csv": data_generator.generate_production_data(),
        "test_quality_data.csv": data_generator.generate_quality_data(),
        "test_energy_data.csv": data_generator.generate_energy_data(),
        "test_material_data.csv": data_generator.generate_material_flow_data(),
    }

    # Save each dataset
    synthetic_dir = results_manager.get_path("SYNTHETIC_DATA")
    for filename, data in synthetic_data.items():
        filepath = synthetic_dir / filename
        data.to_csv(filepath, index=False)
        print(f"Created: {filepath}")


def print_metrics(metrics: Dict) -> None:
    """Print metrics in a formatted way."""
    for category, values in metrics.items():
        print(f"\n{category.upper()} METRICS:")
        if isinstance(values, dict):
            for metric, value in values.items():
                if isinstance(value, (int, float, np.number)):
                    print(f"  {metric}: {float(value):.2f}")
                else:
                    print(f"  {metric}: {value}")
        else:
            print(f"  {values}")


def run_basic_analysis(analyzer: SoliTekManufacturingAnalysis) -> None:
    """Run and display basic manufacturing analysis."""
    print("\n4. Running Manufacturing Analysis...")
    try:
        # Manufacturing Performance
        performance_metrics = analyzer.analyze_manufacturing_performance()
        print("\nPerformance Metrics:")
        print_metrics(performance_metrics)

        # Generate visualizations
        analyzer.generate_visualization("production", "production_trends.png")
        analyzer.generate_visualization("quality", "quality_metrics.png")
        analyzer.generate_visualization("sustainability", "sustainability_metrics.png")
        analyzer.generate_visualization("energy", "energy_patterns.png")

    except Exception as e:
        print(f"Error in basic analysis: {str(e)}")
        raise


def run_lca_analysis(analyzer: SoliTekManufacturingAnalysis) -> None:
    """Run and display lifecycle assessment."""
    print("\n5. Running Lifecycle Assessment...")
    try:
        lca_results = analyzer.perform_lifecycle_assessment()
        print("\nLCA Results:")
        print(lca_results.to_dict())
    except Exception as e:
        print(f"Error in LCA analysis: {str(e)}")
        raise


def run_monitoring_analysis(analyzer: SoliTekManufacturingAnalysis) -> None:
    """Run monitoring system analysis."""
    print("\n6. Running Monitoring Analysis...")
    monitor = ManufacturingMonitor()

    try:
        # Start batch monitoring
        monitor.start_batch_monitoring("BATCH_001")

        # Record metrics
        monitor.record_efficiency_metrics(
            output_quantity=analyzer.production_data["output_amount"].mean(),
            cycle_time=analyzer.production_data["cycle_time"].mean(),
            energy_consumption=analyzer.energy_data["energy_consumption"].mean(),
        )

        monitor.record_quality_metrics(
            defect_rate=analyzer.quality_data["defect_rate"].mean(),
            yield_rate=analyzer.production_data["yield_rate"].mean(),
            uniformity_score=analyzer.quality_data["thickness_uniformity"].mean(),
        )

        monitor.record_resource_metrics(
            material_consumption=analyzer.material_flow["quantity_used"].mean(),
            water_usage=100.0,  # Example value
            waste_generated=analyzer.material_flow["waste_generated"].mean(),
        )

        # Generate batch summary
        batch_summary = monitor.get_batch_summary("BATCH_001")
        print("\nBatch Performance Summary:")
        print_metrics(batch_summary)

        # Save metrics
        monitor.save_metrics("efficiency")
        monitor.save_metrics("quality")
        monitor.save_metrics("resources")

    except Exception as e:
        print(f"Error in monitoring analysis: {str(e)}")
        raise


def run_visualization_analysis(analyzer: SoliTekManufacturingAnalysis) -> None:
    print("\n7. Running Visualization Analysis...")
    manufacturing_visualizer = ManufacturingVisualizer()

    try:
        # Create KPI Dashboard
        manufacturing_visualizer.create_kpi_dashboard(
            metrics_data={
                "efficiency": 91.58,  # From performance metrics
                "quality_score": 95.00,  # From quality metrics
                "resource_efficiency": 89.97,  # From material efficiency
                "energy_efficiency": 0.61,  # From energy efficiency
            },
            save_path="kpi_dashboard.png",
        )

        # Prepare data with calculated metrics
        efficiency_data = analyzer.production_data.copy()
        efficiency_data["production_rate"] = (
            efficiency_data["output_amount"] / efficiency_data["cycle_time"]
        )

        quality_data = analyzer.quality_data.copy()
        quality_data["quality_score"] = (
            (100 - quality_data["defect_rate"]) * 0.4
            + quality_data["efficiency"] * 0.4
            + quality_data["thickness_uniformity"] * 0.2
        )

        resource_data = analyzer.material_flow.copy()
        # Use quantity_used instead of material_consumption
        resource_data["resource_efficiency"] = (
            resource_data["quantity_used"] - resource_data["waste_generated"]
        ) / resource_data["quantity_used"]

        monitor_data = {
            "efficiency": efficiency_data,
            "quality": quality_data,
            "resources": resource_data[
                ["timestamp", "quantity_used", "waste_generated", "resource_efficiency"]
            ],
        }

        manufacturing_visualizer.create_performance_dashboard(
            monitor_data, save_path="performance_dashboard.png"
        )

        # Additional detailed plots
        manufacturing_visualizer.plot_efficiency_trends(
            efficiency_data, save_path="efficiency_trends_detailed.png"
        )

        manufacturing_visualizer.plot_resource_usage(
            resource_data, save_path="resource_usage_detailed.png"
        )

        manufacturing_visualizer.plot_quality_metrics(
            quality_data, save_path="quality_metrics_detailed.png"
        )

    except Exception as e:
        print(f"Error in visualization analysis: {str(e)}")
        raise


def run_optimization_analysis(
    analyzer: SoliTekManufacturingAnalysis, visualizer: OptimizationVisualizer
) -> None:
    """Run and visualize AI optimization analysis."""
    print("\n6. Running AI Optimization Analysis...")

    try:
        # Train optimization model
        print("\nTraining AI Model...")
        training_metrics = analyzer.train_optimization_model()

        # Print training metrics
        print("\nAI Model Training Metrics:")
        print(f"R2 Score: {training_metrics['r2']:.3f}")
        print(f"RMSE: {training_metrics['rmse']:.3f}")
        print(f"MAE: {training_metrics['mae']:.3f}")

        print("\nFeature Importance:")
        for feature, importance in training_metrics["feature_importance"].items():
            print(f"{feature}: {float(importance):.3f}")

        # Create feature importance visualization
        visualizer.plot_feature_importance(training_metrics)

        # Analyze optimization potential
        print("\nAnalyzing Optimization Potential...")
        optimization_potential = analyzer.analyze_optimization_potential()
        print("\nOptimization Potential Analysis:")
        for param, improvement in optimization_potential.items():
            print(f"{param}: {improvement:.2f}%")

        # Run sample predictions
        print("\nRunning Sample Predictions...")
        test_params = {
            "input_amount": 100.0,
            "energy_used": 150.0,
            "cycle_time": 50.0,
            "efficiency": 21.0,
            "defect_rate": 2.0,
            "thickness_uniformity": 95.0,
        }

        predictions = analyzer.predict_batch_outcomes(test_params)
        print("\nSample Prediction Results:")
        print(f"Predicted Output: {predictions['predicted_output']:.2f}")
        print(f"Predicted Quality: {predictions['predicted_quality']:.2f}")
        print(f"Confidence Score: {predictions['confidence_score']:.2f}")

        # Create optimization visualization
        optimization_results = OptimizationResults(
            original_params=test_params,
            optimized_params={k: v * 1.1 for k, v in test_params.items()},
            improvement={k: 10.0 for k in test_params.keys()},
            optimization_success=True,
            optimization_message="Optimization completed successfully",
            iterations=100,
            objective_value=predictions["predicted_output"],
        )

        # Generate optimization visualizations
        visualizer.plot_optimization_impact(optimization_results)
        visualizer.plot_convergence_history(optimization_results)
        visualizer.plot_parameter_comparison(optimization_results)
        visualizer.create_optimization_dashboard(optimization_results, training_metrics)

        # Optimize process parameters
        print("\nOptimizing Process Parameters...")

        # Target values for optimization
        constraints = {
            "input_amount": test_params[
                "input_amount"
            ],  # Use current values as targets
            "energy_used": test_params["energy_used"],
            "cycle_time": test_params["cycle_time"],
            "efficiency": test_params["efficiency"],
            "defect_rate": test_params["defect_rate"],
            "thickness_uniformity": test_params["thickness_uniformity"],
        }

        optimized_params = analyzer.optimize_process_parameters(
            test_params, constraints=constraints
        )
        print("\nOptimized Parameters:")
        for param, value in optimized_params.items():
            print(f"{param}: {value:.2f}")

    except Exception as e:
        print(f"Error in optimization analysis: {str(e)}")
        raise


def run_demonstration():
    """Run complete system demonstration."""
    print("Starting CIRCMAN5.0 Complete Demonstration")
    print("-" * 50)

    # 1. Generate synthetic data
    print("\n1. Generating Test Data...")
    data_generator = ManufacturingDataGenerator(days=30)
    save_synthetic_data(data_generator)

    # 2. Initialize systems
    print("\n2. Initializing Analysis System...")
    analyzer = SoliTekManufacturingAnalysis()
    visualizer = OptimizationVisualizer()

    # 3. Load Data
    print("\n3. Loading Data...")
    synthetic_dir = results_manager.get_path("SYNTHETIC_DATA")
    analyzer.load_data(
        production_path=str(synthetic_dir / "test_production_data.csv"),
        quality_path=str(synthetic_dir / "test_quality_data.csv"),
        energy_path=str(synthetic_dir / "test_energy_data.csv"),
        material_path=str(synthetic_dir / "test_material_data.csv"),
    )

    try:
        # Run analysis components
        run_basic_analysis(analyzer)
        run_lca_analysis(analyzer)
        run_optimization_analysis(analyzer, visualizer)
        run_monitoring_analysis(analyzer)
        run_visualization_analysis(analyzer)

        # Generate final reports
        print("\n7. Generating Reports...")
        analyzer.generate_reports()

        # Print results locations
        print("\nResults saved in:")
        print(f"Reports: {results_manager.get_path('reports')}")
        print(f"Visualizations: {results_manager.get_path('visualizations')}")
        print(f"LCA Results: {results_manager.get_path('lca_results')}")
        print(f"AI Metrics: {results_manager.get_path('metrics')}")

    except Exception as e:
        print(f"\nError in demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    run_demonstration()
