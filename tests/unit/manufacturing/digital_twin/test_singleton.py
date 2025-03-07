import pytest
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin


def test_digital_twin_singleton():
    """Test that DigitalTwin is a proper singleton."""
    # Create two instances
    twin1 = DigitalTwin()
    twin2 = DigitalTwin()

    # They should be the same object
    assert twin1 is twin2

    # Test that state is shared
    test_state = {"system_status": "running", "test_value": 42}

    # Update state in first instance
    twin1.update(test_state)

    # Check state in second instance
    current_state = twin2.get_current_state()
    assert "test_value" in current_state
    assert current_state["test_value"] == 42
    assert current_state["system_status"] == "running"
