# tests/integration/test_digital_twin_visualization.py

"""
Integration tests for Digital Twin visualization capabilities.
"""

import pytest
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.results_manager import results_manager


def test_digital_twin_visualization_generates_file():
    """Test that the digital twin visualization generates a file."""
    # Initialize the manufacturing analysis system
    analyzer = SoliTekManufacturingAnalysis()

    # Generate test data
    generator = ManufacturingDataGenerator(days=5)
    production_data = generator.generate_production_data()
    energy_data = generator.generate_energy_data()
    quality_data = generator.generate_quality_data()
    material_data = generator.generate_material_flow_data()

    # Create temporary files for the test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save data to temporary files
        prod_path = Path(temp_dir) / "production_data.csv"
        energy_path = Path(temp_dir) / "energy_data.csv"
        quality_path = Path(temp_dir) / "quality_data.csv"
        material_path = Path(temp_dir) / "material_data.csv"

        production_data.to_csv(prod_path, index=False)
        energy_data.to_csv(energy_path, index=False)
        quality_data.to_csv(quality_path, index=False)
        material_data.to_csv(material_path, index=False)

        # Load data into the analyzer
        analyzer.load_data(
            production_path=str(prod_path),
            energy_path=str(energy_path),
            quality_path=str(quality_path),
            material_path=str(material_path),
        )

        # Create a temporary file for the visualization
        viz_path = Path(temp_dir) / "digital_twin_visualization.png"

        # Generate visualization
        result = analyzer.generate_digital_twin_visualization(save_path=viz_path)

        # Check that visualization was generated
        assert result is True
        assert viz_path.exists()
        assert viz_path.stat().st_size > 0  # File should have content

        # Test dashboard generation
        dashboard_path = Path(temp_dir) / "digital_twin_dashboard.png"
        dashboard_result = analyzer.generate_digital_twin_dashboard(
            save_path=dashboard_path
        )

        # Check that dashboard was generated
        assert dashboard_result is True
        assert dashboard_path.exists()
        assert dashboard_path.stat().st_size > 0  # File should have content

        # Test historical visualization
        history_path = Path(temp_dir) / "digital_twin_history.png"
        history_result = analyzer.visualize_digital_twin_history(
            metrics=["production_line.production_rate", "production_line.temperature"],
            save_path=history_path,
        )

        # Check that historical visualization was generated
        assert history_result is True
        assert history_path.exists()
        assert history_path.stat().st_size > 0  # File should have content


def test_digital_twin_simulation_visualization():
    """Test visualizing digital twin after running a simulation."""
    # Initialize the manufacturing analysis system
    analyzer = SoliTekManufacturingAnalysis()

    # Generate test data
    generator = ManufacturingDataGenerator(days=5)
    production_data = generator.generate_production_data()
    energy_data = generator.generate_energy_data()
    quality_data = generator.generate_quality_data()
    material_data = generator.generate_material_flow_data()

    # Create temporary files for the test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save data to temporary files
        prod_path = Path(temp_dir) / "production_data.csv"
        energy_path = Path(temp_dir) / "energy_data.csv"
        quality_path = Path(temp_dir) / "quality_data.csv"
        material_path = Path(temp_dir) / "material_data.csv"

        production_data.to_csv(prod_path, index=False)
        energy_data.to_csv(energy_path, index=False)
        quality_data.to_csv(quality_path, index=False)
        material_data.to_csv(material_path, index=False)

        # Load data into the analyzer
        analyzer.load_data(
            production_path=str(prod_path),
            energy_path=str(energy_path),
            quality_path=str(quality_path),
            material_path=str(material_path),
        )

        # Run a simulation
        parameters = {
            "production_line.temperature": 25.0,
            "production_line.production_rate": 150.0,
            "production_line.energy_consumption": 100.0,
        }
        simulation_result = analyzer.simulate_manufacturing_scenario(
            parameters, steps=10
        )

        # Create a temporary file for the visualization
        viz_path = Path(temp_dir) / "simulation_visualization.png"

        # Generate visualization after simulation
        result = analyzer.generate_digital_twin_visualization(save_path=viz_path)

        # Check that visualization was generated
        assert result is True
        assert viz_path.exists()
        assert viz_path.stat().st_size > 0  # File should have content

        # Generate dashboard after simulation
        dashboard_path = Path(temp_dir) / "simulation_dashboard.png"
        dashboard_result = analyzer.generate_digital_twin_dashboard(
            save_path=dashboard_path
        )

        # Check that dashboard was generated
        assert dashboard_result is True
        assert dashboard_path.exists()
        assert dashboard_path.stat().st_size > 0  # File should have content
