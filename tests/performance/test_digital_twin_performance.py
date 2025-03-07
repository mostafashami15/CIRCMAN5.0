# tests/performance/test_digital_twin_performance.py

"""
Performance benchmark tests for the Digital Twin system.

This module implements performance tests to measure simulation speed,
state update latency, and synchronization performance of the Digital Twin.
"""

import pytest
import time
import statistics
import numpy as np
from pathlib import Path
import os
import json

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.utils.results_manager import results_manager

# Add performance path to results_manager if it doesn't exist
try:
    results_manager.get_path("performance")
except KeyError:
    # Create performance directory in the current run directory
    performance_dir = results_manager.current_run / "performance"
    os.makedirs(performance_dir, exist_ok=True)
    results_manager.run_dirs["performance"] = performance_dir
    print(f"Added performance path: {performance_dir}")


def test_simulation_performance():
    """Test the performance of the Digital Twin simulation."""
    # Initialize digital twin
    twin = DigitalTwin()
    twin.initialize()

    # Prepare test state
    test_state = {
        "timestamp": "2025-02-25T10:00:00",
        "production_line": {
            "status": "running",
            "temperature": 22.5,
            "energy_consumption": 100.0,
            "production_rate": 5.0,
        },
        "materials": {
            "silicon_wafer": {"inventory": 1000, "quality": 0.95},
            "solar_glass": {"inventory": 500, "quality": 0.98},
        },
    }
    twin.update(test_state)

    # Measure simulation performance
    simulation_times = []
    simulation_steps = 20
    iterations = 10

    print(f"Running {iterations} simulation tests with {simulation_steps} steps each")

    # Run multiple simulation cycles to get statistical data
    for i in range(iterations):
        start_time = time.time()
        twin.simulate(steps=simulation_steps)
        end_time = time.time()

        simulation_time = (end_time - start_time) * 1000  # Convert to ms
        simulation_times.append(simulation_time)

        # Reset state between runs
        twin.update(test_state)

    # Calculate statistics
    avg_time = statistics.mean(simulation_times)
    max_time = max(simulation_times)
    min_time = min(simulation_times)
    p95_time = (
        statistics.quantiles(simulation_times, n=20)[18]
        if len(simulation_times) >= 20
        else max_time
    )  # 95th percentile

    # Calculate per-step metrics
    avg_time_per_step = avg_time / simulation_steps

    # Log results
    print(f"Simulation Performance ({simulation_steps} steps):")
    print(f"  Total times:")
    print(f"    Average: {avg_time:.2f} ms")
    print(f"    Min: {min_time:.2f} ms")
    print(f"    Max: {max_time:.2f} ms")
    print(f"    95th percentile: {p95_time:.2f} ms")
    print(f"  Per-step times:")
    print(f"    Average: {avg_time_per_step:.2f} ms/step")

    # Save performance data
    performance_data = {
        "test": "simulation_performance",
        "timestamp": time.time(),
        "steps": simulation_steps,
        "iterations": iterations,
        "times": {
            "average_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "p95_ms": p95_time,
            "per_step_ms": avg_time_per_step,
        },
        "raw_data": simulation_times,
    }

    # Save performance data using results_manager
    results_dir = results_manager.get_path("performance")
    results_file = results_dir / f"simulation_performance_{int(time.time())}.json"

    with open(results_file, "w") as f:
        json.dump(performance_data, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Assert performance requirements
    assert (
        avg_time_per_step < 50.0
    ), f"Average simulation time per step too high: {avg_time_per_step:.2f}ms"
    assert (
        p95_time < 1000.0
    ), f"95th percentile simulation time too high: {p95_time:.2f}ms"


def test_state_update_latency():
    """Test the latency of Digital Twin state updates."""
    # Initialize digital twin
    twin = DigitalTwin()
    twin.initialize()

    # Measure state update performance
    update_times = []
    iterations = 100

    print(f"Running {iterations} state update tests")

    # Generate test states of increasing complexity
    test_states = []
    for i in range(iterations):
        # Create a state with increasing complexity
        complexity = (i % 10) + 1  # Cycle through complexity levels 1-10

        state = {
            "timestamp": f"2025-02-25T10:{i:02d}:00",
            "system_status": "running",
            "production_line": {
                "status": "running",
                "temperature": 22.5 + (i * 0.1),
                "energy_consumption": 100.0 + i,
                "production_rate": 5.0 + (i * 0.1),
            },
            "materials": {},
        }

        # Add varying numbers of materials
        for j in range(complexity):
            material_name = f"material_{j}"
            state["materials"][material_name] = {
                "inventory": 1000 - j * 10,
                "quality": 0.95 - (j * 0.01),
                "properties": {
                    "density": 2.5 + (j * 0.1),
                    "viscosity": 1.0 + (j * 0.05),
                },
            }

        test_states.append(state)

    # Run update tests
    for state in test_states:
        start_time = time.time()
        twin.update(state)
        end_time = time.time()

        update_time = (end_time - start_time) * 1000  # Convert to ms
        update_times.append(update_time)

    # Calculate statistics
    avg_time = statistics.mean(update_times)
    max_time = max(update_times)
    min_time = min(update_times)
    p95_time = (
        statistics.quantiles(update_times, n=20)[18]
        if len(update_times) >= 20
        else max_time
    )  # 95th percentile

    # Log results
    print(f"State Update Latency:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  95th percentile: {p95_time:.2f} ms")

    # Save performance data using results_manager
    results_dir = results_manager.get_path("performance")
    results_file = results_dir / f"state_update_performance_{int(time.time())}.json"

    performance_data = {
        "test": "state_update_latency",
        "timestamp": time.time(),
        "iterations": iterations,
        "times": {
            "average_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "p95_ms": p95_time,
        },
    }

    with open(results_file, "w") as f:
        json.dump(performance_data, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Assert performance requirements
    assert avg_time < 20.0, f"Average state update time too high: {avg_time:.2f}ms"
    assert (
        p95_time < 50.0
    ), f"95th percentile state update time too high: {p95_time:.2f}ms"


def test_state_history_performance():
    """Test the performance of state history operations."""
    # Initialize digital twin
    twin = DigitalTwin()
    twin.initialize()

    # Fill history with test states
    history_size = 500
    print(f"Generating {history_size} history states")

    for i in range(history_size):
        twin.update(
            {"timestamp": f"2025-02-25T10:{i//60:02d}:{i%60:02d}", "test_value": i}
        )

    # Measure retrieval performance
    retrieval_times = []
    iterations = 20

    print(f"Running {iterations} history retrieval tests")

    # Measure different history retrieval operations
    for i in range(iterations):
        # Full history retrieval
        start_time = time.time()
        full_history = twin.get_state_history()
        end_time = time.time()

        full_retrieval_time = (end_time - start_time) * 1000  # Convert to ms
        retrieval_times.append(("full", full_retrieval_time))

        # Limited history retrieval
        limit = (i % 5) * 100 + 10  # Cycle through different limits
        start_time = time.time()
        limited_history = twin.get_state_history(limit=limit)
        end_time = time.time()

        limited_retrieval_time = (end_time - start_time) * 1000  # Convert to ms
        retrieval_times.append(("limited", limited_retrieval_time))

    # Calculate statistics
    full_times = [t[1] for t in retrieval_times if t[0] == "full"]
    limited_times = [t[1] for t in retrieval_times if t[0] == "limited"]

    avg_full_time = statistics.mean(full_times)
    avg_limited_time = statistics.mean(limited_times)

    # Log results
    print(f"State History Performance:")
    print(f"  Full History Retrieval (avg): {avg_full_time:.2f} ms")
    print(f"  Limited History Retrieval (avg): {avg_limited_time:.2f} ms")

    # Save performance data using results_manager
    results_dir = results_manager.get_path("performance")
    results_file = results_dir / f"history_performance_{int(time.time())}.json"

    performance_data = {
        "test": "state_history_performance",
        "timestamp": time.time(),
        "history_size": history_size,
        "iterations": iterations,
        "times": {
            "full_retrieval_avg_ms": avg_full_time,
            "limited_retrieval_avg_ms": avg_limited_time,
        },
    }

    with open(results_file, "w") as f:
        json.dump(performance_data, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Assert performance requirements
    assert (
        avg_full_time < 100.0
    ), f"Average full history retrieval time too high: {avg_full_time:.2f}ms"
    assert (
        avg_limited_time < 50.0
    ), f"Average limited history retrieval time too high: {avg_limited_time:.2f}ms"
