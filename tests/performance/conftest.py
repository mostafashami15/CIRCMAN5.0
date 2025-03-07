# tests/performance/conftest.py

import pytest
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
from circman5.manufacturing.human_interface.core.interface_manager import (
    InterfaceManager,
)


@pytest.fixture(scope="function")
def setup_test_environment():
    """Set up test environment with digital twin and interface components."""
    # Create digital twin instance
    digital_twin = DigitalTwin()
    digital_twin.initialize()

    # Create state manager
    state_manager = StateManager()

    # Create interface manager
    interface_manager = InterfaceManager()
    interface_manager.initialize()

    # Return components in a dictionary
    env = {
        "digital_twin": digital_twin,
        "state_manager": state_manager,
        "interface_manager": interface_manager,
    }

    yield env

    # Cleanup
    interface_manager.shutdown()
