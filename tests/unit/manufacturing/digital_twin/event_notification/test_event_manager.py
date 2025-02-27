# tests/unit/manufacturing/digital_twin/event_notification/test_event_manager.py

import pytest
from unittest.mock import MagicMock, patch
import threading
import time

from circman5.manufacturing.digital_twin.event_notification.event_types import (
    Event,
    EventCategory,
    EventSeverity,
    SystemStateEvent,
)
from circman5.manufacturing.digital_twin.event_notification.event_manager import (
    EventManager,
    EventHandler,
    EventFilter,
)


@pytest.fixture
def event_manager():
    """Get a fresh EventManager instance."""
    # Reset the singleton for testing
    EventManager._instance = None
    # Mock the persistence to avoid file operations during tests
    with patch(
        "circman5.manufacturing.digital_twin.event_notification.event_manager.EventPersistence"
    ):
        manager = EventManager()
        manager.persistence_enabled = False  # Disable persistence for tests
        yield manager


@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    return Event(
        message="Test event",
        source="test_source",
        category=EventCategory.SYSTEM,
        severity=EventSeverity.INFO,
    )


@pytest.fixture
def sample_threshold_event():
    """Create a sample threshold event for testing."""
    return Event(
        message="Threshold breach",
        source="test_source",
        category=EventCategory.THRESHOLD,
        severity=EventSeverity.WARNING,
    )


def test_event_manager_singleton(event_manager):
    """Test that EventManager is a singleton."""
    manager1 = EventManager()
    manager2 = EventManager()

    assert manager1 is manager2
    assert manager1 is event_manager


def test_subscribe_handler(event_manager, sample_event):
    """Test subscribing a handler to events."""
    handler_called = False
    event_received = None

    def test_handler(event):
        nonlocal handler_called, event_received
        handler_called = True
        event_received = event

    # Subscribe handler
    event_manager.subscribe(test_handler, EventCategory.SYSTEM)

    # Publish event
    event_manager.publish(sample_event)

    # Verify handler was called
    assert handler_called
    assert event_received is sample_event


def test_subscribe_to_specific_category(
    event_manager, sample_event, sample_threshold_event
):
    """Test subscribing a handler to a specific category."""
    handled_events = []

    def test_handler(event):
        handled_events.append(event)

    # Subscribe handler to SYSTEM category only
    event_manager.subscribe(test_handler, EventCategory.SYSTEM)

    # Publish different events
    event_manager.publish(sample_event)  # SYSTEM
    event_manager.publish(sample_threshold_event)  # THRESHOLD

    # Verify only SYSTEM events were handled
    assert len(handled_events) == 1
    assert handled_events[0] is sample_event


def test_subscribe_to_specific_severity(event_manager):
    """Test subscribing a handler to a specific severity."""
    handled_events = []

    def test_handler(event):
        handled_events.append(event)

    # Subscribe handler to WARNING severity only
    event_manager.subscribe(test_handler, severity=EventSeverity.WARNING)

    # Create and publish events with different severities
    info_event = Event(severity=EventSeverity.INFO)
    warning_event = Event(severity=EventSeverity.WARNING)
    error_event = Event(severity=EventSeverity.ERROR)

    event_manager.publish(info_event)
    event_manager.publish(warning_event)
    event_manager.publish(error_event)

    # Verify only WARNING events were handled
    assert len(handled_events) == 1
    assert handled_events[0] is warning_event


def test_unsubscribe_handler(event_manager, sample_event):
    """Test unsubscribing a handler."""
    handler_called = False

    def test_handler(event):
        nonlocal handler_called
        handler_called = True

    # Subscribe handler
    event_manager.subscribe(test_handler, EventCategory.SYSTEM)

    # Unsubscribe handler
    event_manager.unsubscribe(test_handler, EventCategory.SYSTEM)

    # Publish event
    event_manager.publish(sample_event)

    # Verify handler was not called
    assert not handler_called


def test_add_filter(event_manager, sample_event):
    """Test adding a filter for events."""
    handler_called = False

    def test_handler(event):
        nonlocal handler_called
        handler_called = True

    def test_filter(event):
        return "pass" in event.message

    # Subscribe handler
    event_manager.subscribe(test_handler, EventCategory.SYSTEM)

    # Add filter
    event_manager.add_filter(EventCategory.SYSTEM, test_filter)

    # Publish event that doesn't pass filter
    event_manager.publish(sample_event)

    # Verify handler was not called
    assert not handler_called

    # Create event that passes filter
    passing_event = Event(message="This should pass", category=EventCategory.SYSTEM)

    # Publish event that passes filter
    event_manager.publish(passing_event)

    # Verify handler was called
    assert handler_called


def test_remove_filter(event_manager, sample_event):
    """Test removing a filter."""
    handler_called = False

    def test_handler(event):
        nonlocal handler_called
        handler_called = True

    def test_filter(event):
        return False  # Never pass

    # Subscribe handler
    event_manager.subscribe(test_handler, EventCategory.SYSTEM)

    # Add filter
    event_manager.add_filter(EventCategory.SYSTEM, test_filter)

    # Verify filter blocks event
    event_manager.publish(sample_event)
    assert not handler_called

    # Remove filter
    event_manager.remove_filter(EventCategory.SYSTEM, test_filter)

    # Verify event now passes
    event_manager.publish(sample_event)
    assert handler_called


def test_multiple_handlers(event_manager, sample_event):
    """Test multiple handlers receiving the same event."""
    handler1_called = False
    handler2_called = False

    def test_handler1(event):
        nonlocal handler1_called
        handler1_called = True

    def test_handler2(event):
        nonlocal handler2_called
        handler2_called = True

    # Subscribe handlers
    event_manager.subscribe(test_handler1, EventCategory.SYSTEM)
    event_manager.subscribe(test_handler2, EventCategory.SYSTEM)

    # Publish event
    event_manager.publish(sample_event)

    # Verify both handlers were called
    assert handler1_called
    assert handler2_called


def test_handler_exception_doesnt_break_others(event_manager, sample_event):
    """Test that an exception in one handler doesn't affect others."""
    handler2_called = False

    def test_handler1(event):
        raise Exception("Test exception")

    def test_handler2(event):
        nonlocal handler2_called
        handler2_called = True

    # Subscribe handlers
    event_manager.subscribe(test_handler1, EventCategory.SYSTEM)
    event_manager.subscribe(test_handler2, EventCategory.SYSTEM)

    # Publish event
    event_manager.publish(sample_event)

    # Verify second handler was still called
    assert handler2_called


def test_thread_safety(event_manager):
    """Test thread safety of event manager."""
    received_events = []

    def test_handler(event):
        # Simulate some processing time
        time.sleep(0.01)
        received_events.append(event)

    # Subscribe handler
    event_manager.subscribe(test_handler)

    # Define function to publish events from a thread
    def publish_events(count):
        for i in range(count):
            event = Event(message=f"Event {i}")
            event_manager.publish(event)

    # Create and start threads
    num_threads = 5
    events_per_thread = 10
    threads = []

    for _ in range(num_threads):
        thread = threading.Thread(target=publish_events, args=(events_per_thread,))
        threads.append(thread)
        thread.start()

    # Wait for threads to complete
    for thread in threads:
        thread.join()

    # Verify all events were received
    assert len(received_events) == num_threads * events_per_thread
