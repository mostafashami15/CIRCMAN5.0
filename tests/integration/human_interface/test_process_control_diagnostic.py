# tests/integration/human_interface/test_process_control_diagnostic.py

import pytest
import time


def test_process_control_diagnostic(setup_test_environment):
    """Diagnose the disconnect between process control and digital twin state."""
    env = setup_test_environment
    interface_manager = env["interface_manager"]
    digital_twin = env["digital_twin"]
    digital_twin_adapter = env["digital_twin_adapter"]

    # Get process control component
    process_control = interface_manager.get_component("process_control")
    assert process_control is not None

    # Print process control attributes and methods to understand its implementation
    print("\nProcess Control Methods:")
    for attr_name in dir(process_control):
        if not attr_name.startswith("_") and callable(
            getattr(process_control, attr_name)
        ):
            print(f"  - {attr_name}")

    # Set initial state to idle
    digital_twin.update(
        {"system_status": "idle", "production_line": {"status": "idle"}}
    )
    print("\nInitial Digital Twin State:")
    print(f"  {digital_twin.get_current_state()}")

    # Try starting process using process_control
    print("\nCalling process_control.start_process():")
    result = process_control.start_process()
    print(f"  Result: {result}")

    # Check state immediately after
    print("\nDigital Twin State after start_process:")
    print(f"  {digital_twin.get_current_state()}")

    # Try direct update using digital_twin_adapter
    print("\nTrying direct update via digital_twin_adapter:")
    update_result = digital_twin_adapter.update_state(
        {"system_status": "running", "production_line": {"status": "running"}}
    )
    print(f"  Update result: {update_result}")

    # Check state again
    print("\nDigital Twin State after direct update:")
    print(f"  {digital_twin.get_current_state()}")

    # For debugging: let's see how process_control is trying to update the state
    # Look at its internal implementation
    print("\nProcess Control object details:")
    print(f"  Type: {type(process_control)}")

    # Check if process_control has a digital_twin attribute
    if hasattr(process_control, "digital_twin"):
        print(f"  Has digital_twin: Yes, type: {type(process_control.digital_twin)}")
    else:
        print("  Has digital_twin: No")

    # See what other components it has
    if hasattr(process_control, "state"):
        print(f"  Has state: Yes")

    if hasattr(process_control, "state_manager"):
        print(f"  Has state_manager: Yes")

    if hasattr(process_control, "event_adapter"):
        print(f"  Has event_adapter: Yes")
