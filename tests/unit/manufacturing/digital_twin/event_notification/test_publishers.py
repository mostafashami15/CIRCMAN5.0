# tests/unit/manufacturing/digital_twin/event_notification/test_publishers.py
import pytest
from unittest.mock import patch, MagicMock

from circman5.manufacturing.digital_twin.event_notification.event_types import (
    Event,
    EventCategory,
    EventSeverity,
    SystemStateEvent,
    ThresholdEvent,
    OptimizationEvent,
    ErrorEvent,
    UserActionEvent,
)
from circman5.manufacturing.digital_twin.event_notification.publishers import (
    Publisher,
    DigitalTwinPublisher,
)


@pytest.fixture
def publisher():
    """Create a Publisher instance for testing."""
    with patch(
        "circman5.manufacturing.digital_twin.event_notification.publishers.event_manager"
    ) as mock_em:
        publisher = Publisher(source="test_source")
        yield publisher, mock_em


@pytest.fixture
def digital_twin_publisher():
    """Create a DigitalTwinPublisher instance for testing."""
    with patch(
        "circman5.manufacturing.digital_twin.event_notification.publishers.event_manager"
    ) as mock_em:
        publisher = DigitalTwinPublisher()
        yield publisher, mock_em


def test_publish_event(publisher):
    """Test publishing an event."""
    pub, mock_em = publisher

    # Create and publish event
    event = Event(message="Test event")
    pub.publish_event(event)

    # Verify event source was set and event was published
    assert event.source == "test_source"
    mock_em.publish.assert_called_once_with(event)


def test_publish_system_state_event(publisher):
    """Test publishing a system state event."""
    pub, mock_em = publisher

    # Publish system state event
    pub.publish_system_state_event(previous_state="idle", new_state="running")

    # Verify event was published
    assert mock_em.publish.call_count == 1
    published_event = mock_em.publish.call_args[0][0]

    # Verify event properties
    assert isinstance(published_event, SystemStateEvent)
    assert published_event.category == EventCategory.SYSTEM
    assert published_event.severity == EventSeverity.INFO
    assert published_event.source == "test_source"
    assert "idle" in published_event.message
    assert "running" in published_event.message
    assert published_event.details["previous_state"] == "idle"
    assert published_event.details["new_state"] == "running"


def test_publish_threshold_event(publisher):
    """Test publishing a threshold event."""
    pub, mock_em = publisher

    # Publish threshold event
    pub.publish_threshold_event(
        parameter="temperature", threshold=25.0, actual_value=30.0
    )

    # Verify event was published
    assert mock_em.publish.call_count == 1
    published_event = mock_em.publish.call_args[0][0]

    # Verify event properties
    assert isinstance(published_event, ThresholdEvent)
    assert published_event.category == EventCategory.THRESHOLD
    assert published_event.severity == EventSeverity.WARNING
    assert published_event.source == "test_source"
    assert "temperature" in published_event.message
    assert published_event.details["parameter"] == "temperature"
    assert published_event.details["threshold"] == 25.0
    assert published_event.details["actual_value"] == 30.0


def test_digital_twin_publish_state_update(digital_twin_publisher):
    """Test publishing a state update from Digital Twin publisher."""
    pub, mock_em = digital_twin_publisher

    # Create previous and new states
    previous_state = {"system_status": "idle", "timestamp": "2025-01-01T12:00:00"}
    new_state = {"system_status": "running", "timestamp": "2025-01-01T12:01:00"}

    # Publish state update
    pub.publish_state_update(previous_state, new_state)

    # Verify event was published
    assert mock_em.publish.call_count == 1
    published_event = mock_em.publish.call_args[0][0]

    # Verify event properties
    assert isinstance(published_event, SystemStateEvent)
    assert published_event.details["previous_state"] == previous_state
    assert published_event.details["new_state"] == new_state


def test_digital_twin_publish_parameter_threshold(digital_twin_publisher):
    """Test publishing a parameter threshold event from Digital Twin publisher."""
    pub, mock_em = digital_twin_publisher

    # Create state
    state = {"system_status": "running", "timestamp": "2025-01-01T12:00:00"}

    # Publish parameter threshold event
    pub.publish_parameter_threshold_event(
        parameter_path="production_line.temperature",
        parameter_name="Temperature",
        threshold=25.0,
        actual_value=30.0,
        state=state,
    )

    # Verify event was published
    assert mock_em.publish.call_count == 1
    published_event = mock_em.publish.call_args[0][0]

    # Verify event properties
    assert isinstance(published_event, ThresholdEvent)
    assert published_event.details["parameter_path"] == "production_line.temperature"
    assert published_event.details["state_snapshot"] == state
