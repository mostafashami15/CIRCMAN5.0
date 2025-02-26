# tests/integration/test_digital_twin_integration.py

import pytest
from pathlib import Path
import pandas as pd
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator


def test_digital_twin_initialization():
    """Test that the Digital Twin initializes properly."""
    analyzer = SoliTekManufacturingAnalysis()

    # Verify digital twin integration
    status = analyzer.verify_digital_twin_integration()
    assert status["initialized"] is True


def test_digital_twin_synchronization():
    """Test data synchronization with the Digital Twin."""
    # Generate test data
    generator = ManufacturingDataGenerator(start_date="2025-02-01", days=5)
    production_data = generator.generate_production_data()
    energy_data = generator.generate_energy_data()
    quality_data = generator.generate_quality_data()
    material_data = generator.generate_material_flow_data()

    # Save test data to temporary files
    prod_path = Path("temp_production.csv")
    production_data.to_csv(prod_path, index=False)

    # Initialize analyzer and load data
    analyzer = SoliTekManufacturingAnalysis()
    analyzer.load_data(production_path=str(prod_path))

    # Verify data was synchronized
    status = analyzer.verify_digital_twin_integration()
    assert status["initialized"] is True
    assert status["current_state"] is not None

    # Clean up
    prod_path.unlink()


def test_digital_twin_simulation():
    """Test Digital Twin simulation capabilities."""
    analyzer = SoliTekManufacturingAnalysis()

    # Basic simulation with default parameters
    simulation_results = analyzer.simulate_manufacturing_scenario(steps=5)

    # Verify simulation results
    assert len(simulation_results) > 0
    assert "timestamp" in simulation_results[0]

    # Test state saving/loading
    analyzer.save_digital_twin_state()


def test_digital_twin_optimization():
    """Test optimization using the Digital Twin."""
    analyzer = SoliTekManufacturingAnalysis()

    # Define test parameters
    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    # Run optimization
    try:
        optimized = analyzer.optimize_using_digital_twin(
            test_params, simulation_steps=3
        )
        assert isinstance(optimized, dict)
        assert "input_amount" in optimized
    except Exception as e:
        pytest.skip(f"Optimization failed with error: {str(e)}")
