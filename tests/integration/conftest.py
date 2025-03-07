# tests/integration/conftest.py

import pytest
from circman5.manufacturing.human_interface.core.interface_manager import (
    interface_manager,
)
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
    digital_twin_adapter,
)
from circman5.manufacturing.human_interface.components.controls.process_control import (
    process_control,
)
from circman5.manufacturing.human_interface.services.command_service import (
    command_service,
)


@pytest.fixture
def setup_test_environment():
    """Set up a test environment with manually registered components."""
    # Initialize the interface
    interface_manager.initialize()

    # Manually register components that aren't auto-registered
    if "process_control" not in interface_manager.components:
        interface_manager.register_component("process_control", process_control)

    if "command_service" not in interface_manager.components:
        interface_manager.register_component("command_service", command_service)

    # Get the digital_twin from the process_control component
    # Important: Use this instead of creating a new instance
    digital_twin = process_control.digital_twin

    # Return initialized components for testing
    return {
        "interface_manager": interface_manager,
        "digital_twin_adapter": digital_twin_adapter,
        "digital_twin": digital_twin,
        "process_control": process_control,
        "command_service": command_service,
    }
