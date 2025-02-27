# tests/unit/manufacturing/digital_twin/conftest.py

"""
Pytest fixtures for Digital Twin tests.

This module provides fixtures used across Digital Twin test modules.
"""

import datetime
import pytest
import tempfile
from pathlib import Path

from circman5.manufacturing.digital_twin.core.twin_core import (
    DigitalTwin,
    DigitalTwinConfig,
)
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
from circman5.manufacturing.digital_twin.core.synchronization import (
    SynchronizationManager,
    SyncMode,
)
from circman5.utils.results_manager import results_manager
from circman5.manufacturing.digital_twin.simulation.simulation_engine import (
    SimulationEngine,
)
from circman5.manufacturing.digital_twin.simulation.scenario_manager import (
    ScenarioManager,
)
from circman5.manufacturing.digital_twin.simulation.process_models import (
    PVManufacturingProcessModel,
    SiliconPurificationModel,
    WaferProductionModel,
)
from circman5.manufacturing.lifecycle.lca_analyzer import LCAAnalyzer
from circman5.manufacturing.lifecycle.visualizer import LCAVisualizer


@pytest.fixture
def sample_state() -> dict:
    """Sample manufacturing state for testing."""
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "system_status": "running",
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
        "environment": {"temperature": 22.0, "humidity": 45.0},
    }


@pytest.fixture
def state_manager():
    """Return a StateManager instance for testing."""
    return StateManager(history_length=10)


@pytest.fixture
def digital_twin():
    """Return a DigitalTwin instance for testing."""
    config = DigitalTwinConfig(name="TestTwin", update_frequency=0.1, history_length=5)
    return DigitalTwin(config)


@pytest.fixture
def initialized_twin(digital_twin):
    """Return an initialized DigitalTwin instance."""
    digital_twin.initialize()
    return digital_twin


@pytest.fixture
def sync_manager(state_manager):
    """Return a SynchronizationManager instance for testing."""
    return SynchronizationManager(
        state_manager, sync_mode=SyncMode.MANUAL, sync_interval=0.1
    )


@pytest.fixture
def temp_json_file():
    """Temporary JSON file for testing."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        temp_path = Path(temp.name)

    yield str(temp_path)

    # Clean up the file after the test
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture(scope="session", autouse=True)
def setup_digital_twin_directory():
    """Ensure digital_twin directory exists."""
    from circman5.utils.results_manager import results_manager

    # Create test directories
    try:
        # Attempt to get the path, this will fail if it doesn't exist
        dt_dir = results_manager.get_path("digital_twin")
    except KeyError:
        # Add the directory if it doesn't exist
        results_manager.run_dirs["digital_twin"] = (
            results_manager.get_run_dir() / "digital_twin"
        )
        dt_dir = results_manager.run_dirs["digital_twin"]

    # Ensure the directory exists
    dt_dir.mkdir(parents=True, exist_ok=True)

    yield


@pytest.fixture
def simulation_engine(state_manager) -> SimulationEngine:
    """Simulation engine fixture for testing."""
    return SimulationEngine(state_manager)


@pytest.fixture
def scenario_manager() -> ScenarioManager:
    """Scenario manager fixture for testing."""
    return ScenarioManager()


@pytest.fixture
def process_model_silicon() -> SiliconPurificationModel:
    """Silicon purification model fixture for testing."""
    return SiliconPurificationModel()


@pytest.fixture
def process_model_wafer() -> WaferProductionModel:
    """Wafer production model fixture for testing."""
    return WaferProductionModel()


@pytest.fixture
def lca_analyzer():
    """Return an LCAAnalyzer instance for testing."""
    return LCAAnalyzer()


@pytest.fixture
def lca_visualizer():
    """Return an LCAVisualizer instance for testing."""
    return LCAVisualizer()


@pytest.fixture
def sample_lca_state():
    """Sample state with detailed material and energy data for LCA testing."""
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "batch_id": "test_batch_001",
        "system_status": "running",
        "production_line": {
            "status": "running",
            "temperature": 25.0,
            "energy_consumption": 1000.0,
            "production_rate": 100.0,
            "energy_source": "grid_electricity",
        },
        "materials": {
            "silicon_wafer": {"inventory": 100.0, "quality": 0.95},
            "solar_glass": {"inventory": 150.0, "quality": 0.98},
            "eva_sheet": {"inventory": 50.0, "quality": 0.9},
            "backsheet": {"inventory": 30.0, "quality": 0.92},
            "aluminum_frame": {"inventory": 80.0, "quality": 0.97},
        },
        "environment": {"temperature": 22.0, "humidity": 45.0},
    }


@pytest.fixture
def lca_integration(initialized_twin):
    """Return an LCAIntegration instance for testing."""
    # Import inside the fixture to avoid circular imports
    from circman5.manufacturing.digital_twin.integration.lca_integration import (
        LCAIntegration,
    )

    return LCAIntegration(digital_twin=initialized_twin)
