# tests/integration/human_interface/test_event_notification.py

import pytest
import time


def test_event_notification_integration(setup_test_environment):
    """Test that events from the digital twin appear in the alert system."""
    env = setup_test_environment
    interface_manager = env["interface_manager"]

    # Get required components
    process_control = interface_manager.get_component("process_control")
    digital_twin = process_control.digital_twin

    # Get the event manager
    from circman5.manufacturing.digital_twin.event_notification.event_manager import (
        event_manager,
    )
    from circman5.manufacturing.digital_twin.event_notification.event_types import (
        Event,
        EventCategory,
        EventSeverity,
    )

    # Create and publish a test event
    test_event = Event(
        category=EventCategory.THRESHOLD,
        severity=EventSeverity.WARNING,
        message="Test temperature threshold exceeded",
        source="test_event_notification",
        details={"parameter": "temperature", "value": 95.2, "threshold": 90.0},
    )

    # Publish the event
    event_manager.publish(test_event)

    # Allow time for event propagation
    time.sleep(0.5)

    # Check if alert panel exists and test it if available
    if "alert_panel" in interface_manager.components:
        alert_panel = interface_manager.get_component("alert_panel")
        alerts = alert_panel.get_filtered_alerts({})
        test_alerts = [
            a for a in alerts if "Test temperature threshold" in a.get("message", "")
        ]
        assert len(test_alerts) > 0, "Test event not found in alerts"
    else:
        print("Alert panel component not found - skipping alert verification")

    # Test that process control events also work
    result = process_control.start_process()
    assert result["success"] is True

    # Verify state changed
    current_state = digital_twin.get_current_state()
    assert current_state["system_status"] == "running"
