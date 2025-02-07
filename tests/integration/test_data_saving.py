"""Test data saving locations."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from circman5.visualization.manufacturing_visualizer import ManufacturingVisualizer
from circman5.analysis.lca.core import LCAAnalyzer, LifeCycleImpact
from circman5.monitoring import ManufacturingMonitor
from circman5.config.project_paths import project_paths


def test_all_components_saving():
    # Create test data with correct columns
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    production_data = pd.DataFrame(
        {
            "timestamp": dates,
            "production_rate": np.random.normal(100, 10, 10),
            "energy_efficiency": np.random.normal(0.8, 0.1, 10),
        }
    )

    # Get run directory
    run_dir = project_paths.get_run_directory()

    # Test visualizer
    viz = ManufacturingVisualizer()
    viz.plot_efficiency_trends(production_data)
    assert (run_dir / "visualizations" / "efficiency_trends.png").exists()

    # Test monitor
    monitor = ManufacturingMonitor()
    monitor.metrics_history["efficiency"] = production_data
    monitor.save_metrics("efficiency")
    assert (run_dir / "reports" / "efficiency_metrics.csv").exists()

    # Test LCA
    lca = LCAAnalyzer()
    impact = lca.perform_full_lca(
        material_inputs={"silicon": 100.0, "glass": 200.0},
        energy_consumption=1000.0,
        lifetime_years=25.0,
        annual_energy_generation=2000.0,
        grid_carbon_intensity=0.5,
        recycling_rates={"silicon": 0.8, "glass": 0.9},
        transport_distance=100.0,
    )
    lca.save_results(impact, "TEST_001")
    assert (run_dir / "reports" / "lca_impact_TEST_001.xlsx").exists()
