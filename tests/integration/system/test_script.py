"""
Test script for SoliTek Manufacturing Analysis with AI optimization.
"""

from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
import pandas as pd
import numpy as np


def main():
    # Create test instance
    print("Creating SoliTek Manufacturing Analysis instance...")
    analyzer = SoliTekManufacturingAnalysis()

    # Generate sample data
    print("\nGenerating sample data...")
    n_samples = 100
    dates = pd.date_range("2025-01-01", periods=n_samples, freq="H")

    # Production data
    analyzer.production_data = pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(n_samples)],
            "timestamp": dates,
            "input_amount": np.random.uniform(80, 120, n_samples),
            "output_amount": np.random.uniform(75, 110, n_samples),
            "energy_used": np.random.uniform(140, 160, n_samples),
            "cycle_time": np.random.uniform(45, 55, n_samples),
        }
    )

    # Quality data
    analyzer.quality_data = pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(n_samples)],
            "test_timestamp": dates,
            "efficiency": np.random.uniform(20, 22, n_samples),
            "defect_rate": np.random.uniform(1, 3, n_samples),
            "thickness_uniformity": np.random.uniform(94, 96, n_samples),
        }
    )

    # Train the model
    print("\nTraining AI optimization model...")
    metrics = analyzer.train_optimization_model()
    print("Training Metrics:", metrics)

    # Test optimization
    print("\nTesting process optimization...")
    current_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    optimized = analyzer.optimize_process_parameters(current_params)
    print("Optimized Parameters:", optimized)

    # Test prediction
    print("\nTesting outcome prediction...")
    predictions = analyzer.predict_batch_outcomes(current_params)
    print("Predicted Outcomes:", predictions)

    # Test optimization potential analysis
    print("\nAnalyzing optimization potential...")
    improvements = analyzer.analyze_optimization_potential()
    print("Potential Improvements:", improvements)

    # Generate performance report
    print("\nGenerating performance report...")
    analyzer.generate_performance_report("test_performance_report.png")


if __name__ == "__main__":
    main()
