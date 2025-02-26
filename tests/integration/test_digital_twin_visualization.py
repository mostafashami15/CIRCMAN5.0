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


def test_material_flow_sankey_visualization():
    """Test generating a material flow Sankey diagram."""
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

        # Run a simulation to generate state data
        simulation_result = analyzer.simulate_manufacturing_scenario(steps=10)

        # Create a temporary file for the Sankey diagram
        sankey_path = Path(temp_dir) / "material_flow_sankey.png"

        # Generate Sankey diagram
        result = analyzer.generate_material_flow_sankey(save_path=sankey_path)

        # Check that Sankey diagram was generated
        assert result is True
        assert sankey_path.exists()
        assert sankey_path.stat().st_size > 0  # File should have content


def test_efficiency_heatmap_visualization():
    """Test generating an efficiency metrics heatmap."""
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

        # Run a simulation to generate state data
        simulation_result = analyzer.simulate_manufacturing_scenario(steps=10)

        # Create a temporary file for the heatmap
        heatmap_path = Path(temp_dir) / "efficiency_heatmap.png"

        # Generate efficiency heatmap
        result = analyzer.generate_efficiency_heatmap(save_path=heatmap_path)

        # Check that heatmap was generated
        assert result is True
        assert heatmap_path.exists()
        assert heatmap_path.stat().st_size > 0  # File should have content


def test_enhanced_dashboard_visualization():
    """Test generating an enhanced dashboard with advanced visualizations."""
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

        # Run a simulation to generate state data
        simulation_result = analyzer.simulate_manufacturing_scenario(steps=10)

        # Create a temporary file for the enhanced dashboard
        dashboard_path = Path(temp_dir) / "enhanced_dashboard.png"

        # Generate enhanced dashboard
        result = analyzer.generate_enhanced_dashboard(save_path=dashboard_path)

        # Check that enhanced dashboard was generated
        assert result is True
        assert dashboard_path.exists()
        assert dashboard_path.stat().st_size > 0  # File should have content


def test_process_visualization():
    """Test generating process-specific visualization."""
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
        simulation_result = analyzer.simulate_manufacturing_scenario(steps=10)

        # Create a temporary file for the visualization
        process_path = Path(temp_dir) / "process_visualization.png"

        # Generate process visualization
        result = analyzer.generate_process_visualization(save_path=process_path)

        # Check that visualization was generated
        assert result is True
        assert process_path.exists()
        assert process_path.stat().st_size > 0  # File should have content


def test_state_comparison_visualization():
    """Test comparing digital twin states with visualization."""
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

        # Run a simulation to generate an initial state
        initial_state = analyzer.digital_twin.get_current_state()

        # Run another simulation to change the state
        parameters = {
            "production_line": {
                "status": "running",
                "temperature": 25.0,
                "production_rate": 100.0,
                "energy_consumption": 120.0,
            }
        }
        analyzer.simulate_manufacturing_scenario(parameters=parameters, steps=10)
        updated_state = analyzer.digital_twin.get_current_state()

        # Create a temporary file for the visualization
        comparison_path = Path(temp_dir) / "state_comparison.png"

        # Generate state comparison visualization
        result = analyzer.compare_digital_twin_states(
            state1=initial_state,
            state2=updated_state,
            labels=("Initial", "After Simulation"),
            save_path=comparison_path,
        )

        # Check that visualization was generated
        assert result is True
        assert comparison_path.exists()
        assert comparison_path.stat().st_size > 0  # File should have content


def test_parameter_sensitivity_analysis():
    """Test parameter sensitivity analysis visualization."""
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

        # Run a simulation to generate some state data
        analyzer.simulate_manufacturing_scenario(steps=5)

        # Create a temporary file for the visualization
        sensitivity_path = Path(temp_dir) / "parameter_sensitivity.png"

        # Generate parameter sensitivity visualization
        result = analyzer.analyze_parameter_sensitivity(
            parameter="production_line.temperature",
            min_value=15.0,
            max_value=35.0,
            steps=5,
            save_path=sensitivity_path,
        )

        # Check that visualization was generated
        assert result is True
        assert sensitivity_path.exists()
        assert sensitivity_path.stat().st_size > 0  # File should have content
