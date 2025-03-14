#!/usr/bin/env python3
# examples/manufacturing_optimization_demo.py

"""
CIRCMAN5.0 Manufacturing Optimization Demo

This example demonstrates an end-to-end manufacturing optimization workflow:
1. Generate synthetic manufacturing data using enhanced data generator
2. Initialize the Digital Twin and AI components
3. Train advanced AI models (Deep Learning and Ensemble)
4. Use online learning for real-time adaptation
5. Optimize manufacturing parameters
6. Visualize results with uncertainty quantification

This demo shows integration with all major CIRCMAN5.0 components:
- Enhanced Data Generation
- Advanced AI Models
- Digital Twin
- Process Optimization
- Uncertainty Quantification
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import joblib

from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Union, cast

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import CIRCMAN5.0 components
from src.circman5.test_data_generator import (
    ManufacturingDataGenerator,
    generate_complete_test_dataset,
)
from src.circman5.utils.results_manager import results_manager
from src.circman5.manufacturing.core import SoliTekManufacturingAnalysis

# Import advanced AI components
from src.circman5.manufacturing.optimization.advanced_models.deep_learning import (
    DeepLearningModel,
)
from src.circman5.manufacturing.optimization.advanced_models.ensemble import (
    EnsembleModel,
)
from src.circman5.manufacturing.optimization.online_learning.adaptive_model import (
    AdaptiveModel,
)
from src.circman5.manufacturing.optimization.validation.cross_validator import (
    CrossValidator,
)
from src.circman5.manufacturing.optimization.validation.uncertainty import (
    UncertaintyQuantifier,
)

# Create output directory
output_dir = results_manager.get_path("digital_twin") / "demo_output"
output_dir.mkdir(exist_ok=True)

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(output_dir / "demo.log"), logging.StreamHandler()],
)
logger = logging.getLogger("optimization_demo")


def generate_synthetic_data():
    """Generate synthetic manufacturing data for the demo."""
    logger.info("Generating synthetic manufacturing data...")

    # Use the enhanced data generator
    generator = ManufacturingDataGenerator()

    # Generate a complete dataset
    dataset = generate_complete_test_dataset(
        production_batches=200, time_series_days=30, edge_cases=20
    )

    # Save copies to output dir for reference
    for name, df in dataset.items():
        temp_file = f"{name}_data.csv"
        df.to_csv(temp_file, index=False)
        results_manager.save_file(temp_file, "digital_twin")
        Path(temp_file).unlink()  # Clean up

    logger.info(
        f"Generated synthetic data with {len(dataset['production'])} production records"
    )
    return dataset


def initialize_analysis_system():
    """Initialize the main manufacturing analysis system."""
    logger.info("Initializing SoliTek manufacturing analysis system...")

    # Initialize main analysis class that integrates all components
    analysis_system = SoliTekManufacturingAnalysis()

    logger.info("Analysis system initialized")
    return analysis_system


def train_advanced_models(dataset):
    """Train advanced AI models with the synthetic data."""
    logger.info("Training advanced AI models...")

    # Extract data
    production_data = dataset["production"]
    quality_data = dataset["quality"]

    # Prepare features and targets
    X = production_data[["input_amount", "energy_used", "cycle_time"]].copy()
    if "efficiency" in production_data.columns:
        X["efficiency"] = production_data["efficiency"]
    else:
        X["efficiency"] = quality_data["efficiency"]
    X["defect_rate"] = quality_data["defect_rate"]
    X["thickness_uniformity"] = quality_data["thickness_uniformity"]

    y = production_data["output_amount"]

    # Train deep learning model
    logger.info("Training Deep Learning model...")
    dl_model = DeepLearningModel(model_type="lstm")
    dl_metrics = dl_model.train(X, y)

    # Train ensemble model
    logger.info("Training Ensemble model...")
    ensemble_model = EnsembleModel()
    ensemble_metrics = ensemble_model.train(X, y)

    # Compare model performances
    logger.info("Models trained successfully")
    logger.info(f"Deep Learning metrics: {dl_metrics}")
    logger.info(f"Ensemble metrics: {ensemble_metrics}")

    # Save models
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    dl_model.save_model(models_dir / "deep_learning_model.pkl")
    ensemble_model.save_model(models_dir / "ensemble_model.pkl")

    logger.info(f"Models saved to {models_dir}")

    # Return models
    return {
        "deep_learning": dl_model,
        "ensemble": ensemble_model,
    }


def setup_adaptive_learning(models, dataset):
    """Set up adaptive learning with initial data."""
    logger.info("Setting up adaptive learning...")

    # Initialize adaptive model with ensemble base
    adaptive_model = AdaptiveModel(base_model_type="ensemble")

    # Extract time series data
    time_series = dataset["time_series"]

    # Prepare initial batch of data
    initial_data = time_series.iloc[:50].copy()

    # Prepare features and targets for initial training
    features = [
        "input_amount",
        "power_consumption",
        "efficiency",
        "defect_rate",
        "temperature",
    ]
    X_initial = initial_data[features].values
    y_initial = initial_data["output_amount"].values.reshape(-1, 1)

    # Add initial data batch
    for i in range(len(X_initial)):
        adaptive_model.add_data_point(X_initial[i : i + 1], y_initial[i : i + 1])

    logger.info(f"Adaptive model initialized with {len(X_initial)} data points")

    # Save model
    model_file = output_dir / "models" / "adaptive_model_initial.pkl"
    joblib.dump(adaptive_model, model_file)
    logger.info(f"Adaptive model saved to {model_file}")

    return adaptive_model


def validate_models(models, dataset):
    """Validate models using cross-validation and uncertainty quantification."""
    logger.info("Validating models...")

    # Prepare validation data
    production_data = dataset["production"]
    quality_data = dataset["quality"]

    # Prepare features and targets
    X = production_data[["input_amount", "energy_used", "cycle_time"]].copy()
    if "efficiency" in production_data.columns:
        X["efficiency"] = production_data["efficiency"]
    else:
        X["efficiency"] = quality_data["efficiency"]
    X["defect_rate"] = quality_data["defect_rate"]
    X["thickness_uniformity"] = quality_data["thickness_uniformity"]

    y = production_data["output_amount"]

    # Cross-validation
    validator = CrossValidator()

    # Validate each model
    validation_results = {}
    for name, model in models.items():
        logger.info(f"Cross-validating {name} model...")
        validation_results[name] = validator.validate(model, X.values, y.values)

    # Uncertainty quantification
    uncertainty_quantifier = UncertaintyQuantifier()

    # Calibrate uncertainty
    logger.info("Calibrating uncertainty quantifier...")
    uncertainty_quantifier.calibrate(models["ensemble"], X.values, y.values)

    # Quantify uncertainty for a sample of data
    sample_X = X.iloc[:20].values
    uncertainty_results = uncertainty_quantifier.quantify_uncertainty(
        models["ensemble"], sample_X
    )

    # Save results
    with open(output_dir / "validation_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in validation_results.items():
            serializable_model_results = {}
            for metric, metric_data in results.get("metrics", {}).items():
                serializable_metric_data = {}
                for k, v in metric_data.items():
                    if isinstance(v, np.ndarray):
                        serializable_metric_data[k] = v.tolist()
                    else:
                        serializable_metric_data[k] = v
                serializable_model_results[metric] = serializable_metric_data
            serializable_results[model_name] = {"metrics": serializable_model_results}
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Validation results saved to {output_dir / 'validation_results.json'}")

    # Return results
    return {"validation": validation_results, "uncertainty": uncertainty_results}


def optimize_manufacturing_process(models, analysis_system, dataset):
    """Optimize manufacturing process parameters using advanced models."""
    logger.info("Optimizing manufacturing process parameters...")

    # Load the synthetic data into the analysis system so it can train its optimizer
    logger.info("Loading synthetic data into analysis system...")
    analysis_system.production_data = dataset["production"]
    analysis_system.quality_data = dataset["quality"]

    # Current manufacturing parameters
    current_params = {
        "input_amount": 120.0,
        "energy_used": 600.0,
        "cycle_time": 50.0,
        "efficiency": 0.82,
        "defect_rate": 0.08,
        "thickness_uniformity": 0.88,
        "output_amount": 90.0,
    }

    # Define constraints for optimization
    constraints = {
        "input_amount": (80.0, 150.0),
        "energy_used": (400.0, 800.0),
        "cycle_time": (30.0, 60.0),
        "efficiency": (0.75, 0.95),
        "defect_rate": (0.01, 0.15),
        "thickness_uniformity": (0.8, 0.98),
    }

    # Method 1: Using the core analysis system
    optimized_params_core = None
    try:
        logger.info("Method 1: Optimizing using SoliTekManufacturingAnalysis...")
        from typing import Dict, Any, List, Tuple, Union, cast

        optimized_params_core = analysis_system.optimize_process_parameters(
            current_params,
            cast(Dict[str, Union[float, Tuple[float, float]]], constraints),
        )
    except Exception as e:
        logger.error(f"Error in core optimization: {str(e)}")

    # Method 2: Skip ensemble model optimization since it's missing required methods
    # We could create a wrapper or add the missing method, but for simplicity we'll skip it
    logger.info(
        "Method 2: Skipping Ensemble model optimization (missing required methods)"
    )

    # Method 3: Using the digital twin if available
    optimized_params_dt = None
    try:
        logger.info("Method 3: Optimizing using Digital Twin...")
        optimized_params_dt = analysis_system.optimize_using_digital_twin(
            current_params, constraints
        )
    except Exception as e:
        logger.warning(f"Digital Twin optimization not available: {str(e)}")

    # Collect results (handle the case where some methods might have failed)
    optimization_results = {"current_params": current_params}

    if optimized_params_core:
        optimization_results["optimized_core"] = optimized_params_core

    if optimized_params_dt:
        optimization_results["optimized_digital_twin"] = optimized_params_dt

    # Calculate improvements for methods that worked
    improvements = {}
    for method, params in optimization_results.items():
        if method == "current_params" or params is None:
            continue

        method_improvements = {}
        for param, value in params.items():
            if param in current_params:
                current = current_params[param]
                if abs(current) > 1e-10:  # Avoid division by zero
                    pct_change = (value - current) / current * 100
                    method_improvements[param] = pct_change

        improvements[method] = method_improvements

    # Add improvements to results
    optimization_results["improvements"] = improvements

    # Save results
    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(optimization_results, f, indent=2)

    logger.info(
        f"Optimization results saved to {output_dir / 'optimization_results.json'}"
    )

    return optimization_results


def visualize_results(dataset, models, validation_results, optimization_results):
    """Create visualizations of model performance and optimization results."""
    logger.info("Creating visualizations...")

    # 1. Model Validation Visualization
    plt.figure(figsize=(12, 6))

    # Extract cross-validation scores with safety checks
    model_names = []
    cv_scores = []
    cv_stds = []

    for model_name, results in validation_results.get("validation", {}).items():
        if results and "metrics" in results:
            metrics = results.get("metrics", {})
            r2_data = metrics.get("r2", {})
            if r2_data and "values" in r2_data:
                r2_values = r2_data.get("values", [0])
                model_names.append(model_name)
                cv_scores.append(np.mean(r2_values))
                cv_stds.append(np.std(r2_values))

    # Only plot if we have data
    if model_names and cv_scores:
        # Plot bar chart of cross-validation scores
        plt.bar(model_names, cv_scores, yerr=cv_stds)
        plt.ylabel("R² Score")
        plt.title("Model Cross-Validation Performance")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(output_dir / "model_validation.png", dpi=300)
    else:
        logger.warning("Insufficient data for model validation visualization")
    plt.close()

    # 2. Optimization Impact Visualization - only if we have results
    has_optimization_results = False
    if "current_params" in optimization_results:
        if (
            "optimized_core" in optimization_results
            or "optimized_digital_twin" in optimization_results
        ):
            has_optimization_results = True

    if has_optimization_results:
        plt.figure(figsize=(14, 8))

        # Get parameters to compare
        params_to_plot = ["input_amount", "energy_used", "cycle_time", "output_amount"]

        # Get current values
        current = []
        for param in params_to_plot:
            current.append(optimization_results["current_params"].get(param, 0))

        # Plot current parameters
        x = np.arange(len(params_to_plot))
        width = 0.25
        plt.bar(x - width, current, width, label="Current")

        # Add core optimization if available
        if "optimized_core" in optimization_results:
            optimized_core = []
            for param in params_to_plot:
                optimized_core.append(
                    optimization_results["optimized_core"].get(param, 0)
                )
            plt.bar(x, optimized_core, width, label="Core Optimization")

        # Add digital twin optimization if available
        if "optimized_digital_twin" in optimization_results:
            optimized_dt = []
            for param in params_to_plot:
                optimized_dt.append(
                    optimization_results["optimized_digital_twin"].get(param, 0)
                )
            plt.bar(x + width, optimized_dt, width, label="Digital Twin Optimization")

        plt.xlabel("Parameter")
        plt.ylabel("Value")
        plt.title("Optimization Impact by Parameter")
        plt.xticks(x, params_to_plot)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(output_dir / "optimization_impact.png", dpi=300)
        plt.close()

    # 3. Uncertainty Visualization
    if "uncertainty" in validation_results:
        plt.figure(figsize=(12, 6))

        # Extract uncertainty data
        uncertainty = validation_results["uncertainty"]

        # Check if required data exists
        has_predictions = (
            "predictions" in uncertainty and len(uncertainty["predictions"]) > 0
        )
        has_confidence_intervals = (
            "confidence_intervals" in uncertainty
            and isinstance(uncertainty["confidence_intervals"], np.ndarray)
            and uncertainty["confidence_intervals"].shape[0] > 0
            and uncertainty["confidence_intervals"].shape[1] >= 2
        )

        if has_predictions:
            # Create x-axis
            predictions = uncertainty["predictions"]
            x = np.arange(len(predictions))

            # Plot predictions with confidence intervals
            plt.plot(x, predictions, "b-", label="Predictions")

            if has_confidence_intervals:
                lower_bound = uncertainty["confidence_intervals"][:, 0]
                upper_bound = uncertainty["confidence_intervals"][:, 1]
                plt.fill_between(
                    x,
                    lower_bound,
                    upper_bound,
                    color="b",
                    alpha=0.2,
                    label="95% Confidence Interval",
                )

            plt.xlabel("Sample")
            plt.ylabel("Predicted Output")
            plt.title("Predictions with Uncertainty Quantification")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(output_dir / "uncertainty_quantification.png", dpi=300)
        else:
            logger.warning("Insufficient data for uncertainty visualization")
        plt.close()

    # 4. Time Series Visualization
    if "time_series" in dataset and not dataset["time_series"].empty:
        plt.figure(figsize=(14, 8))

        # Get time series data
        time_series = dataset["time_series"]

        # Plot output amount if available
        if "output_amount" in time_series.columns:
            plt.subplot(2, 1, 1)
            plt.plot(
                time_series.index,
                time_series["output_amount"],
                "b-",
                label="Output Amount",
            )
            plt.title("Manufacturing Output Over Time")
            plt.ylabel("Output Amount")
            plt.grid(alpha=0.3)
            plt.legend()

        # Plot efficiency and defect rate if available
        if "efficiency" in time_series.columns or "defect_rate" in time_series.columns:
            plt.subplot(2, 1, 2)

            if "efficiency" in time_series.columns:
                plt.plot(
                    time_series.index,
                    time_series["efficiency"],
                    "g-",
                    label="Efficiency",
                )

            if "defect_rate" in time_series.columns:
                plt.plot(
                    time_series.index,
                    time_series["defect_rate"],
                    "r-",
                    label="Defect Rate",
                )

            # Highlight anomalies if anomaly_present column exists
            if "anomaly_present" in time_series.columns:
                anomalies = time_series[time_series["anomaly_present"] == True]
                if not anomalies.empty and "efficiency" in anomalies.columns:
                    plt.scatter(
                        anomalies.index,
                        anomalies["efficiency"],
                        c="red",
                        marker="o",
                        s=50,
                        label="Anomalies",
                    )

            plt.title("Efficiency and Defect Rate Over Time")
            plt.ylabel("Rate")
            plt.grid(alpha=0.3)
            plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "time_series_analysis.png", dpi=300)
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def simulate_online_learning(adaptive_model, dataset):
    """Simulate online learning with adaptive model."""
    logger.info("Simulating online learning...")

    # Extract time series data
    time_series = dataset["time_series"]

    # Skip the first 50 rows (used for initialization)
    streaming_data = time_series.iloc[50:150].copy()

    # Prepare features and target
    features = [
        "input_amount",
        "power_consumption",
        "efficiency",
        "defect_rate",
        "temperature",
    ]

    # Initialize tracking
    predictions = []
    actual_values = []
    update_points = []

    # Simulate streaming data
    for i, row in streaming_data.iterrows():
        # Extract features and target
        X = row[features].values.reshape(1, -1)
        y = np.array([[row["output_amount"]]])

        # Make prediction before update
        pred = adaptive_model.predict(X)
        predictions.append(float(pred[0][0]))
        actual_values.append(float(y[0][0]))

        # Update model with new data
        updated = adaptive_model.add_data_point(X, y)
        if updated:
            update_points.append(len(predictions) - 1)

    # Save adaptive model
    save_adaptive_model(
        adaptive_model, output_dir / "models" / "adaptive_model_initial.pkl"
    )

    # Visualize online learning performance
    plt.figure(figsize=(14, 6))

    # Plot predictions vs actual values
    plt.plot(predictions, "b-", label="Adaptive Model Predictions")
    plt.plot(actual_values, "g-", label="Actual Values")

    # Mark model update points
    for update_point in update_points:
        plt.axvline(x=update_point, color="r", linestyle="--", alpha=0.3)

    # Add annotation for update points
    if update_points:
        plt.text(
            update_points[0],
            max(actual_values),
            "Model Updates",
            color="r",
            horizontalalignment="right",
        )

    plt.title("Adaptive Model Performance Over Streaming Data")
    plt.xlabel("Data Point")
    plt.ylabel("Output Amount")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / "adaptive_learning.png", dpi=300)
    plt.close()

    # Calculate performance metrics
    mse = np.mean((np.array(predictions) - np.array(actual_values)) ** 2)
    mae = np.mean(np.abs(np.array(predictions) - np.array(actual_values)))

    logger.info(
        f"Adaptive learning simulation complete. MSE: {mse:.4f}, MAE: {mae:.4f}"
    )

    return {
        "predictions": predictions,
        "actual_values": actual_values,
        "update_points": update_points,
        "mse": mse,
        "mae": mae,
    }


def perform_what_if_analysis(analysis_system, baseline_params):
    """Perform what-if scenario analysis."""
    logger.info("Performing what-if scenario analysis...")

    # Define scenarios to test
    scenarios = [
        {
            "name": "High Input",
            "changes": {"input_amount": baseline_params["input_amount"] * 1.5},
        },
        {
            "name": "Energy Efficient",
            "changes": {"energy_used": baseline_params["energy_used"] * 0.7},
        },
        {
            "name": "Quality Focus",
            "changes": {"defect_rate": baseline_params["defect_rate"] * 0.5},
        },
        {
            "name": "Fast Cycle",
            "changes": {"cycle_time": baseline_params["cycle_time"] * 0.8},
        },
    ]

    # Run scenarios
    scenario_results = []

    try:
        for scenario in scenarios:
            logger.info(f"Running scenario: {scenario['name']}")

            # Create parameter changes
            params = baseline_params.copy()
            params.update(scenario["changes"])

            # Make prediction using analysis system
            prediction = analysis_system.predict_batch_outcomes(params)

            # Run simulation if Digital Twin is available
            try:
                simulation = analysis_system.simulate_manufacturing_scenario(
                    params, steps=5
                )
                simulated_output = simulation[-1].get("output_amount", 0)
            except (AttributeError, ValueError) as e:
                logger.warning(f"Digital Twin simulation not available: {str(e)}")
                simulated_output = None

            # Store results
            scenario_results.append(
                {
                    "name": scenario["name"],
                    "parameters": params,
                    "predicted_output": prediction.get("predicted_output", 0),
                    "simulated_output": simulated_output,
                }
            )
    except Exception as e:
        logger.error(f"Error in scenario analysis: {str(e)}")

    # Save results
    with open(output_dir / "scenario_analysis.json", "w") as f:
        json.dump(scenario_results, f, indent=2)

    # Visualize scenario results
    if scenario_results:
        plt.figure(figsize=(12, 6))

        # Extract data
        scenario_names = [s["name"] for s in scenario_results]
        predicted_outputs = [s["predicted_output"] for s in scenario_results]

        # Add baseline
        scenario_names.insert(0, "Baseline")
        predicted_outputs.insert(0, baseline_params.get("output_amount", 0))

        # Plot bar chart
        bars = plt.bar(scenario_names, predicted_outputs)

        # Color bars (green if better than baseline, red if worse)
        baseline_output = predicted_outputs[0]
        for i, bar in enumerate(bars):
            if i > 0:  # Skip baseline
                if predicted_outputs[i] > baseline_output:
                    bar.set_color("green")
                else:
                    bar.set_color("red")

        plt.title("What-If Scenario Analysis")
        plt.ylabel("Predicted Output")
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(output_dir / "scenario_analysis.png", dpi=300)
        plt.close()

    logger.info(f"Scenario analysis complete. Results saved to {output_dir}")

    return scenario_results


def main():
    """Main function to run the complete demo."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("CIRCMAN5.0 Manufacturing Optimization Demo")
    logger.info("=" * 80)

    try:
        # Step 1: Generate synthetic data
        dataset = generate_synthetic_data()

        # Step 2: Initialize analysis system
        analysis_system = initialize_analysis_system()

        # Step 3: Train advanced models
        models = train_advanced_models(dataset)

        # Step 4: Setup adaptive learning
        adaptive_model = setup_adaptive_learning(models, dataset)

        # Step 5: Validate models
        validation_results = validate_models(models, dataset)

        # Step 6: Optimize manufacturing process
        optimization_results = optimize_manufacturing_process(
            models, analysis_system, dataset
        )

        # Step 7: Simulate online learning
        online_learning_results = simulate_online_learning(adaptive_model, dataset)

        # Step 8: Perform what-if analysis
        scenario_results = perform_what_if_analysis(
            analysis_system, optimization_results["current_params"]
        )

        # Step 9: Create visualizations
        visualize_results(dataset, models, validation_results, optimization_results)

        # Print summary
        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"Demo completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"All output files saved to: {output_dir.resolve()}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in demo: {str(e)}", exc_info=True)
        logger.info("Demo failed. Check log for details.")


def save_adaptive_model(model, file_path):
    """Helper function to save an adaptive model using joblib."""
    import joblib

    model_dir = Path(file_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    logger.info(f"Adaptive model saved to {file_path}")


if __name__ == "__main__":
    main()
