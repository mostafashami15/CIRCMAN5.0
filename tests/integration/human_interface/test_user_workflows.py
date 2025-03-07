# tests/integration/human_interface/test_user_workflows.py

import pytest
import time


def test_manufacturing_monitoring_workflow(setup_test_environment):
    """Test a complete manufacturing monitoring workflow."""
    env = setup_test_environment
    interface_manager = env["interface_manager"]

    # Get required components
    process_control = interface_manager.get_component("process_control")
    dashboard_manager = interface_manager.get_component("dashboard_manager")

    # Use the digital_twin instance from process_control
    digital_twin = process_control.digital_twin

    # Step 1: System displays current manufacturing status
    dashboard_data = dashboard_manager.render_dashboard("main_dashboard")
    assert "panels" in dashboard_data
    assert "status" in dashboard_data["panels"]

    # Step 2: Set initial state
    digital_twin.update(
        {
            "system_status": "idle",
            "production_line": {"status": "idle", "temperature": 75.0},
        }
    )

    # Step 3: Start manufacturing process
    result = process_control.start_process()
    assert result["success"] is True

    # Verify state changed
    current_state = digital_twin.get_current_state()
    assert current_state["system_status"] == "running"
    assert current_state["production_line"]["status"] == "running"

    # Step 4: Simulate temperature increase (critical condition)
    digital_twin.update({"production_line": {"temperature": 95.8}})  # High temperature

    # Step 5: Use process control to adjust parameter
    result = process_control.adjust_process_parameter(
        "main_process", "temperature_setpoint", 85.0
    )
    assert result["success"] is True

    # Step 6: Simulate temperature returning to normal
    digital_twin.update(
        {"production_line": {"temperature": 85.3}}  # Normal temperature
    )

    # Step 7: Stop process
    result = process_control.stop_process()
    assert result["success"] is True

    # Verify process stopped
    current_state = digital_twin.get_current_state()
    assert current_state["system_status"] == "idle"
    assert current_state["production_line"]["status"] == "idle"
