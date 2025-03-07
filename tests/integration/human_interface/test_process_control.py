# tests/integration/human_interface/test_process_control.py

import pytest


def test_process_control_functions(setup_test_environment):
    """Test that process control can control the digital twin state."""
    env = setup_test_environment
    interface_manager = env["interface_manager"]

    # Get process control component directly from interface manager
    process_control = interface_manager.get_component("process_control")

    # Use the digital twin instance from process_control
    digital_twin = process_control.digital_twin

    # Set initial state to idle
    digital_twin.update(
        {"system_status": "idle", "production_line": {"status": "idle"}}
    )

    # Verify initial state
    current_state = digital_twin.get_current_state()
    assert current_state["system_status"] == "idle"
    assert current_state["production_line"]["status"] == "idle"

    # Use process control to start the process
    result = process_control.start_process()
    assert result["success"] is True

    # Verify state changed
    current_state = digital_twin.get_current_state()
    assert current_state["system_status"] == "running"
    assert current_state["production_line"]["status"] == "running"

    # Use process control to stop the process
    result = process_control.stop_process()
    assert result["success"] is True

    # Verify state changed back
    current_state = digital_twin.get_current_state()
    assert current_state["system_status"] == "idle"
    assert current_state["production_line"]["status"] == "idle"
