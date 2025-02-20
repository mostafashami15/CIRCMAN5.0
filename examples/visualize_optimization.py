# examples/visualize_optimization.py

import json
from pathlib import Path
from circman5.manufacturing.reporting.optimization_visualizer import (
    OptimizationVisualizer,
)
from circman5.utils.results_manager import results_manager
from circman5.manufacturing.optimization.types import OptimizationResults, MetricsDict
from typing import Tuple


def create_sample_data() -> Tuple[OptimizationResults, MetricsDict]:
    """Create sample optimization results for demonstration."""
    # Create with proper type annotations
    optimization_results: OptimizationResults = OptimizationResults(
        original_params={
            "input_amount": 100.0,
            "energy_used": 50.0,
            "cycle_time": 30.0,
            "efficiency": 0.8,
        },
        optimized_params={
            "input_amount": 95.0,
            "energy_used": 45.0,
            "cycle_time": 28.0,
            "efficiency": 0.85,
        },
        improvement={
            "input_amount": -5.0,
            "energy_used": -10.0,
            "cycle_time": -6.67,
            "efficiency": 6.25,
        },
        optimization_success=True,
        optimization_message="Optimization successful",
        iterations=100,
        objective_value=0.95,
    )

    training_metrics: MetricsDict = MetricsDict(
        mse=0.02,
        rmse=0.14,
        mae=0.12,
        r2=0.95,
        cv_r2_mean=0.94,
        cv_r2_std=0.02,
        feature_importance={
            "input_amount": 0.35,
            "energy_used": 0.25,
            "cycle_time": 0.20,
            "efficiency": 0.20,
        },
    )

    # Save sample data
    results_dir = results_manager.get_path("lca_results")
    with open(results_dir / "optimization_results.json", "w") as f:
        json.dump(optimization_results, f, indent=2)
    with open(results_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    return optimization_results, training_metrics


def main():
    """Generate visualization examples."""
    print("Creating optimization visualizations...")

    # Initialize visualizer
    visualizer = OptimizationVisualizer()

    # Create and save sample data
    optimization_results, training_metrics = create_sample_data()

    # Create visualizations
    try:
        # 1. Optimization Impact Plot
        impact_plot = visualizer.plot_optimization_impact(optimization_results)
        print(f"Created optimization impact plot: {impact_plot}")

        # 2. Feature Importance Plot
        importance_plot = visualizer.plot_feature_importance(training_metrics)
        print(f"Created feature importance plot: {importance_plot}")

        # 3. Parameter Comparison Plot
        comparison_plot = visualizer.plot_parameter_comparison(optimization_results)
        print(f"Created parameter comparison plot: {comparison_plot}")

        # 4. Convergence History Plot
        convergence_plot = visualizer.plot_convergence_history(optimization_results)
        print(f"Created convergence history plot: {convergence_plot}")

        # 5. Create comprehensive dashboard
        dashboard = visualizer.create_optimization_dashboard(
            optimization_results, training_metrics
        )
        print(f"Created optimization dashboard: {dashboard}")

        print(
            "\nAll visualizations have been saved to:",
            results_manager.get_path("visualizations"),
        )

    except Exception as e:
        print(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    main()
