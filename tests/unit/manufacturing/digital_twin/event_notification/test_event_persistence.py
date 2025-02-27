# tests/unit/manufacturing/digital_twin/event_notification/test_event_persistence.py

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from circman5.manufacturing.digital_twin.event_notification.event_types import (
    Event,
    EventCategory,
    EventSeverity,
)
from circman5.manufacturing.digital_twin.event_notification.event_persistence import (
    EventPersistence,
)


@pytest.fixture
def event_persistence():
    """Create an EventPersistence instance for testing."""
    # Mock results_manager to avoid file operations
    with patch(
        "circman5.manufacturing.digital_twin.event_notification.event_persistence.results_manager"
    ) as mock_rm:
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up mock to return temp_dir path
            mock_rm.get_path.return_value = Path(temp_dir)
            mock_rm.get_run_dir.return_value = Path(temp_dir)
            # Create EventPersistence instance
            persistence = EventPersistence(max_events=10)
            yield persistence


@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    return Event(
        message="Test event",
        source="test_source",
        category=EventCategory.SYSTEM,
        severity=EventSeverity.INFO,
    )


def test_save_event(event_persistence, sample_event):
    """Test saving an event."""
    # Save event
    event_persistence.save_event(sample_event)

    # Verify event is in memory
    assert len(event_persistence.events) == 1
    assert event_persistence.events[0] is sample_event


def test_max_events_limit(event_persistence):
    """Test enforcing max events limit."""
    # Save more events than the limit
    for i in range(15):
        event = Event(message=f"Event {i}")
        event_persistence.save_event(event)

    # Verify only the latest max_events are kept
    assert len(event_persistence.events) == 10
    assert event_persistence.events[0].message == "Event 5"
    assert event_persistence.events[-1].message == "Event 14"


def test_get_events_filtering(event_persistence):
    """Test getting events with filtering."""
    # Save events with different categories and severities
    event1 = Event(category=EventCategory.SYSTEM, severity=EventSeverity.INFO)
    event2 = Event(category=EventCategory.THRESHOLD, severity=EventSeverity.WARNING)
    event3 = Event(category=EventCategory.ERROR, severity=EventSeverity.ERROR)

    event_persistence.save_event(event1)
    event_persistence.save_event(event2)
    event_persistence.save_event(event3)

    # Filter by category
    system_events = event_persistence.get_events(category=EventCategory.SYSTEM)
    assert len(system_events) == 1
    assert system_events[0] is event1

    # Filter by severity
    warning_events = event_persistence.get_events(severity=EventSeverity.WARNING)
    assert len(warning_events) == 1
    assert warning_events[0] is event2


def test_acknowledge_event(event_persistence, sample_event):
    """Test acknowledging an event."""
    # Save event
    event_persistence.save_event(sample_event)

    # Acknowledge event
    result = event_persistence.acknowledge_event(sample_event.event_id)

    # Verify event is acknowledged
    assert result is True
    assert sample_event.acknowledged is True


def test_clear_events(event_persistence):
    """Test clearing events."""
    # Save some events
    for i in range(5):
        event = Event(message=f"Event {i}")
        event_persistence.save_event(event)

    # Clear events
    count = event_persistence.clear_events()

    # Verify all events were cleared
    assert count == 5
    assert len(event_persistence.events) == 0
