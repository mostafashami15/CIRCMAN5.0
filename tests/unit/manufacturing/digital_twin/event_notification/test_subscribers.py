# tests/unit/manufacturing/digital_twin/event_notification/test_subscribers.py

import pytest
from unittest.mock import patch, MagicMock, call

from circman5.manufacturing.digital_twin.event_notification.event_types import (
    Event,
    EventCategory,
    EventSeverity,
    ThresholdEvent,
)
from circman5.manufacturing.digital_twin.event_notification.subscribers import (
    Subscriber,
    LoggingSubscriber,
    ThresholdAlertSubscriber,
    OptimizationSubscriber,
)
from circman5.manufacturing.digital_twin.event_notification.event_manager import (
    EventManager,
)


@pytest.fixture
def subscriber():
    """Create a Subscriber instance for testing."""
    with patch(
        "circman5.manufacturing.digital_twin.event_notification.subscribers.event_manager"
    ) as mock_em:
        subscriber = Subscriber(name="test_subscriber")
        yield subscriber, mock_em


@pytest.fixture
def logging_subscriber():
    """Create a LoggingSubscriber instance for testing."""
    with patch(
        "circman5.manufacturing.digital_twin.event_notification.subscribers.event_manager"
    ) as mock_em:
        with patch(
            "circman5.manufacturing.digital_twin.event_notification.subscribers.setup_logger"
        ) as mock_logger:
            subscriber = LoggingSubscriber(name="test_logging")
            # Set up logger mock for verification
            subscriber.logger = MagicMock()
            yield subscriber, mock_em


@pytest.fixture
def threshold_subscriber():
    """Create a ThresholdAlertSubscriber instance for testing."""
    with patch(
        "circman5.manufacturing.digital_twin.event_notification.subscribers.event_manager"
    ) as mock_em:
        subscriber = ThresholdAlertSubscriber(name="test_threshold")
        # Set up logger mock for verification
        subscriber.logger = MagicMock()
        yield subscriber, mock_em


def test_subscriber_register(subscriber):
    """Test registering a subscriber."""
    sub, mock_em = subscriber

    # Register subscriber
    sub.register(category=EventCategory.SYSTEM)

    # Verify subscriber was registered
    mock_em.subscribe.assert_called_once()
    assert mock_em.subscribe.call_args[1]["handler"] == sub.handle_event
    assert mock_em.subscribe.call_args[1]["category"] == EventCategory.SYSTEM
    assert sub.registered is True


def test_subscriber_unregister(subscriber):
    """Test unregistering a subscriber."""
    sub, mock_em = subscriber

    # Register and then unregister subscriber
    sub.register(category=EventCategory.SYSTEM)
    sub.unregister(category=EventCategory.SYSTEM)

    # Verify subscriber was unregistered
    mock_em.unsubscribe.assert_called_once()
    assert mock_em.unsubscribe.call_args[1]["handler"] == sub.handle_event
    assert mock_em.unsubscribe.call_args[1]["category"] == EventCategory.SYSTEM
    assert sub.registered is False


def test_logging_subscriber_handle_event(logging_subscriber):
    """Test LoggingSubscriber handling different event severities."""
    sub, _ = logging_subscriber

    # Create events with different severities
    info_event = Event(severity=EventSeverity.INFO, message="Info message")
    warning_event = Event(severity=EventSeverity.WARNING, message="Warning message")
    error_event = Event(severity=EventSeverity.ERROR, message="Error message")
    critical_event = Event(severity=EventSeverity.CRITICAL, message="Critical message")

    # Handle events
    sub.handle_event(info_event)
    sub.handle_event(warning_event)
    sub.handle_event(error_event)
    sub.handle_event(critical_event)

    # Verify logging calls
    sub.logger.info.assert_called_once_with("EVENT: Info message")
    sub.logger.warning.assert_called_once_with("EVENT: Warning message")
    sub.logger.error.assert_called_once_with("EVENT: Error message")
    sub.logger.critical.assert_called_once_with("EVENT: Critical message")


def test_threshold_subscriber_handle_event(threshold_subscriber):
    """Test ThresholdAlertSubscriber handling threshold events."""
    sub, _ = threshold_subscriber

    # Create a mock handler
    mock_handler = MagicMock()
    sub.add_alert_handler(mock_handler)

    # Create a threshold event
    threshold_event = ThresholdEvent(
        parameter="temperature", threshold=25.0, actual_value=30.0
    )

    # Handle event
    sub.handle_event(threshold_event)

    # Verify handler was called
    mock_handler.assert_called_once_with(threshold_event)

    # Verify non-threshold events are ignored
    system_event = Event(category=EventCategory.SYSTEM)
    sub.handle_event(system_event)

    # Handler should still have been called only once
    mock_handler.assert_called_once()


def test_threshold_subscriber_add_remove_handler(threshold_subscriber):
    """Test adding and removing alert handlers."""
    sub, _ = threshold_subscriber

    # Create mock handlers
    mock_handler1 = MagicMock()
    mock_handler2 = MagicMock()

    # Add handlers
    sub.add_alert_handler(mock_handler1)
    sub.add_alert_handler(mock_handler2)

    # Create and handle a threshold event
    threshold_event = ThresholdEvent(
        parameter="temperature", threshold=25.0, actual_value=30.0
    )
    sub.handle_event(threshold_event)

    # Verify both handlers were called
    mock_handler1.assert_called_once_with(threshold_event)
    mock_handler2.assert_called_once_with(threshold_event)

    # Remove first handler
    sub.remove_alert_handler(mock_handler1)

    # Handle another event
    another_event = ThresholdEvent(
        parameter="pressure", threshold=100.0, actual_value=120.0
    )
    sub.handle_event(another_event)

    # Verify first handler was not called again, but second was
    mock_handler1.assert_called_once()  # Still only called once
    assert mock_handler2.call_count == 2
