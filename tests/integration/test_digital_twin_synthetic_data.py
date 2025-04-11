# tests/integration/test_digital_twin_synthetic_data.py

"""
Integration tests for Digital Twin with synthetic data generation.

This module tests the integration between the Digital Twin and synthetic data generator,
ensuring the Digital Twin can properly process realistic manufacturing data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.test_data_generator import ManufacturingDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def reset_digital_twin():
    """Reset the Digital Twin singleton before and after each test."""
    DigitalTwin._reset()
    yield
    DigitalTwin._reset()


@pytest.fixture(scope="function")
def initialized_twin(reset_digital_twin):
    """Create and initialize a Digital Twin instance."""
    twin = DigitalTwin()
    twin.initialize()
    return twin


@pytest.fixture(scope="function")
def data_generator():
    """Create a synthetic data generator."""
    return ManufacturingDataGenerator()


def test_digital_twin_with_production_data(initialized_twin, data_generator):
    """Test Digital Twin with synthetic production data."""
    # Generate production data
    production_data = data_generator.generate_production_data()

    logger.info(f"Generated {len(production_data)} production data records")

    # Sample every 10th record to speed up the test
    sample_data = production_data.iloc[::10, :].reset_index(drop=True)

    # Process each production record
    for index, row in sample_data.iterrows():
        # Convert row to a dictionary for update
        state_update = {
            "timestamp": row["timestamp"].isoformat()
            if hasattr(row["timestamp"], "isoformat")
            else str(row["timestamp"]),
            "batch_id": row["batch_id"],
            "production_line": {
                "status": row["status"],
                "production_rate": float(
                    row["output_amount"] / row["cycle_time"] * 60
                ),  # Convert to units per hour
                "energy_consumption": float(row["energy_used"]),
            },
            "process": {
                "input_amount": float(row["input_amount"]),
                "output_amount": float(row["output_amount"]),
                "cycle_time": float(row["cycle_time"]),
                "yield_rate": float(row["yield_rate"]),
                "product_type": row["product_type"],
            },
        }

        # Update Digital Twin with production data
        update_success = initialized_twin.update(state_update)
        assert (
            update_success
        ), f"Failed to update Digital Twin with production data record {index}"

    # Verify state history was recorded
    history = initialized_twin.get_state_history()
    assert (
        len(history) > 0
    ), "Digital Twin should record state history from production data"

    # Check that the last state contains production line information
    last_state = initialized_twin.get_current_state()
    assert (
        "production_line" in last_state
    ), "Production line data should be in the final state"
    assert "process" in last_state, "Process data should be in the final state"

    logger.info("Digital Twin successfully processed production data")


def test_digital_twin_with_time_series_data(initialized_twin, data_generator):
    """Test Digital Twin with synthetic time series data."""
    # Generate time series data
    time_series_data = data_generator.generate_time_series_data(
        days=1, interval_minutes=30
    )

    logger.info(f"Generated {len(time_series_data)} time series data points")

    # Process each time series point
    successful_updates = 0
    for index, row in time_series_data.iterrows():
        if index % 10 != 0:  # Process every 10th record to speed up test
            continue

        # Convert row to dictionary for update
        state_update = {
            "timestamp": row["timestamp"].isoformat()
            if hasattr(row["timestamp"], "isoformat")
            else str(row["timestamp"]),
            "production_line": {
                "status": "running",
                "temperature": float(row["temperature"]),
                "energy_consumption": float(row["energy_used"]),
                "production_rate": float(
                    row["output_amount"] / row["cycle_time"] * 60
                ),  # Convert to units per hour
            },
            "process": {
                "input_amount": float(row["input_amount"]),
                "efficiency": float(row["efficiency"]),
                "cycle_time": float(row["cycle_time"]),
            },
        }

        # Update Digital Twin
        update_success = initialized_twin.update(state_update)
        if update_success:
            successful_updates += 1

    logger.info(
        f"Successfully updated Digital Twin with {successful_updates} time series data points"
    )

    # Verify time series data was processed
    assert successful_updates > 0, "Digital Twin should process time series data"

    # Simulate future behavior based on time series trends
    simulation_results = initialized_twin.simulate(steps=10)
    assert (
        len(simulation_results) == 11
    ), "Simulation should include initial state plus requested steps"

    # Check that simulation results contain key metrics from time series data
    assert "production_line" in simulation_results[-1]
    assert "temperature" in simulation_results[-1]["production_line"]
    assert "energy_consumption" in simulation_results[-1]["production_line"]

    logger.info(
        "Digital Twin successfully processed time series data and performed simulation"
    )


def test_digital_twin_with_edge_cases(initialized_twin, data_generator):
    """Test Digital Twin with synthetic edge case data."""
    # Generate edge cases
    edge_cases = data_generator.generate_edge_cases(num_cases=5)

    logger.info(f"Generated {len(edge_cases)} edge cases")

    # Test each edge case
    for index, edge_case in edge_cases.iterrows():
        # Create a state update from the edge case
        state_update = {
            "case_id": edge_case["case_id"],
            "case_type": edge_case["case_type"],
            "production_line": {
                "status": "running",
                "energy_consumption": float(edge_case["energy_used"]),
            },
            "process": {
                "input_amount": float(edge_case["input_amount"]),
                "efficiency": float(edge_case["efficiency"]),
                "cycle_time": float(edge_case["cycle_time"]),
                "defect_rate": float(edge_case["defect_rate"]),
                "thickness_uniformity": float(edge_case["thickness_uniformity"]),
            },
        }

        # Update Digital Twin with edge case
        # Note: We don't assert success here since some edge cases might be rejected by validation
        update_result = initialized_twin.update(state_update)

        logger.info(
            f"Edge case {edge_case['case_type']} update result: {update_result}"
        )

    # After processing edge cases, verify the Digital Twin can still simulate
    simulation_results = initialized_twin.simulate(steps=5)
    assert (
        len(simulation_results) > 0
    ), "Digital Twin should be able to simulate after edge cases"

    logger.info("Digital Twin edge case processing completed")


def test_digital_twin_with_lca_data(initialized_twin, data_generator):
    """Test Digital Twin with lifecycle assessment (LCA) data."""
    # Generate LCA datasets
    lca_datasets = data_generator.generate_complete_lca_dataset()

    logger.info(
        f"Generated LCA datasets with {len(lca_datasets['material_flow'])} material records and "
        f"{len(lca_datasets['energy_consumption'])} energy records"
    )

    # Process 10 material flow records
    material_sample = lca_datasets["material_flow"].head(10)
    for index, row in material_sample.iterrows():
        # Convert to state update
        state_update = {
            "timestamp": row["timestamp"].isoformat()
            if hasattr(row["timestamp"], "isoformat")
            else str(row["timestamp"]),
            "batch_id": row["batch_id"],
            "materials": {
                row["material_type"].lower(): {
                    "quantity": float(row["quantity_used"]),
                    "waste": float(row["waste_generated"]),
                    "recycled": float(row["recycled_amount"]),
                }
            },
            "production_line": {
                "id": row["production_line"],
                "status": "running",
            },
        }

        # Update Digital Twin
        initialized_twin.update(state_update)

    # Process 10 energy consumption records
    energy_sample = lca_datasets["energy_consumption"].head(10)
    for index, row in energy_sample.iterrows():
        # Convert to state update
        state_update = {
            "timestamp": row["timestamp"].isoformat()
            if hasattr(row["timestamp"], "isoformat")
            else str(row["timestamp"]),
            "batch_id": row["batch_id"],
            "energy": {
                "source": row["energy_source"],
                "consumption": float(row["energy_consumption"]),
                "efficiency": float(row["efficiency_rate"]),
            },
            "production_line": {
                "id": row["production_line"],
                "status": "running",
            },
        }

        # Update Digital Twin
        initialized_twin.update(state_update)

    # Verify the Digital Twin has processed both material and energy data
    current_state = initialized_twin.get_current_state()

    # Check for materials and energy in state (may be in different structures based on DT implementation)
    assert "materials" in current_state or "materials" in current_state.get(
        "production_line", {}
    ), "Material data should be reflected in Digital Twin state"

    assert "energy" in current_state or "energy" in current_state.get(
        "production_line", {}
    ), "Energy data should be reflected in Digital Twin state"

    logger.info("Digital Twin successfully processed LCA data")


def test_realistic_manufacturing_simulation(initialized_twin, data_generator):
    """Test a realistic manufacturing simulation with the Digital Twin."""
    # Set up initial state with realistic values
    initial_state = {
        "timestamp": datetime.now().isoformat(),
        "system_status": "running",
        "production_line": {
            "status": "running",
            "temperature": 22.0,
            "energy_consumption": 100.0,
            "production_rate": 8.0,
            "efficiency": 0.90,
            "defect_rate": 0.05,
        },
        "materials": {
            "silicon_wafer": {"inventory": 1000, "quality": 0.95},
            "solar_glass": {"inventory": 500, "quality": 0.98},
            "eva": {"inventory": 300, "quality": 0.99},
            "backsheet": {"inventory": 450, "quality": 0.97},
            "frame": {"inventory": 600, "quality": 0.99},
        },
        "environment": {
            "temperature": 21.0,
            "humidity": 45.0,
        },
    }

    # Set the initial state
    initialized_twin.state_manager.set_state(initial_state)

    # Run a medium-length simulation (20 steps)
    simulation_results = initialized_twin.simulate(steps=20)

    # Validate simulation results for realistic patterns
    temperatures = [
        s["production_line"]["temperature"]
        for s in simulation_results
        if "production_line" in s and "temperature" in s["production_line"]
    ]

    energy_values = [
        s["production_line"]["energy_consumption"]
        for s in simulation_results
        if "production_line" in s and "energy_consumption" in s["production_line"]
    ]

    production_rates = [
        s["production_line"]["production_rate"]
        for s in simulation_results
        if "production_line" in s and "production_rate" in s["production_line"]
    ]

    # Check for trends in the data that would be expected in a real manufacturing system
    # 1. Temperature should increase with energy consumption
    assert (
        np.corrcoef(temperatures, energy_values)[0, 1] > 0
    ), "Temperature and energy consumption should be positively correlated"

    # 2. Energy consumption should trend upward over time when production line is running
    assert (
        energy_values[-1] > energy_values[0]
    ), "Energy consumption should increase over time in a running production line"

    # 3. Check that temperature increases are physically reasonable (not too extreme)
    temp_increases = np.diff(temperatures)
    assert (
        max(temp_increases) < 5.0
    ), "Temperature increases should be physically reasonable (not too extreme)"

    logger.info("Realistic manufacturing simulation validated successfully")
    logger.info(f"Temperature trend: {temperatures[0]:.2f} → {temperatures[-1]:.2f}")
    logger.info(f"Energy trend: {energy_values[0]:.2f} → {energy_values[-1]:.2f}")
    logger.info(
        f"Production rate trend: {production_rates[0]:.2f} → {production_rates[-1]:.2f}"
    )


# This allows running the tests directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
