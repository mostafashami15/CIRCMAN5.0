# tests/integration/test_digital_twin_events.py
import pytest
from unittest.mock import MagicMock
import time

from circman5.manufacturing.digital_twin.core.twin_core import (
    DigitalTwin,
    DigitalTwinConfig,
)
from circman5.manufacturing.digital_twin.event_notification.event_types import (
    EventCategory,
    EventSeverity,
)
from circman5.manufacturing.digital_twin.event_notification.subscribers import (
    Subscriber,
)


class TestEventSubscriber(Subscriber):
    """Subscriber for testing events."""

    def __init__(self):
        super().__init__(name="test_subscriber")
        self.received_events = []

    def handle_event(self, event):
        self.received_events.append(event)


@pytest.fixture
def twin_with_subscriber():
    """Set up a digital twin with a test subscriber."""
    # Create digital twin
    config = DigitalTwinConfig(name="TestTwin", update_frequency=0.1, history_length=5)
    twin = DigitalTwin(config)
    twin.initialize()

    # Create and register subscriber
    subscriber = TestEventSubscriber()
    subscriber.register()

    yield twin, subscriber


def test_twin_update_generates_events(twin_with_subscriber):
    """Test that updating the digital twin generates events."""
    twin, subscriber = twin_with_subscriber

    # Update the twin
    twin.update({"system_status": "running"})

    # Wait a moment for event processing
    time.sleep(0.1)

    # Verify events were received
    assert len(subscriber.received_events) > 0
    assert any(
        event.category == EventCategory.SYSTEM for event in subscriber.received_events
    )


def test_twin_threshold_events(twin_with_subscriber):
    """Test that threshold breaches generate events."""
    twin, subscriber = twin_with_subscriber

    # Add threshold configuration to digital twin
    twin.parameter_thresholds = {
        "production_line.temperature": {
            "name": "Temperature",
            "value": 25.0,
            "comparison": "greater_than",
            "severity": "WARNING",
        }
    }

    # Update with values that breach threshold
    twin.update({"production_line": {"temperature": 30.0}})

    # Wait a moment for event processing
    time.sleep(0.1)

    # Verify threshold events were received
    threshold_events = [
        e for e in subscriber.received_events if e.category == EventCategory.THRESHOLD
    ]
    assert len(threshold_events) > 0
    assert threshold_events[0].severity == EventSeverity.WARNING
    assert "Temperature" in threshold_events[0].message
    assert threshold_events[0].details["threshold"] == 25.0
    assert threshold_events[0].details["actual_value"] == 30.0
