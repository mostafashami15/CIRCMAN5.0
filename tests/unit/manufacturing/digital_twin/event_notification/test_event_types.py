# tests/unit/manufacturing/digital_twin/event_notification/test_event_types.py

import pytest
import datetime
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


def test_event_creation():
    """Test basic event creation."""
    event = Event(message="Test event", source="test_source")

    assert event.event_id is not None
    assert event.timestamp is not None
    assert event.category == EventCategory.SYSTEM
    assert event.severity == EventSeverity.INFO
    assert event.source == "test_source"
    assert event.message == "Test event"
    assert isinstance(event.details, dict)
    assert not event.acknowledged


def test_event_to_dict():
    """Test event serialization to dict."""
    event = Event(message="Test event", source="test_source", details={"key": "value"})

    event_dict = event.to_dict()

    assert event_dict["event_id"] == event.event_id
    assert event_dict["timestamp"] == event.timestamp
    assert event_dict["category"] == EventCategory.SYSTEM.value
    assert event_dict["severity"] == EventSeverity.INFO.value
    assert event_dict["source"] == "test_source"
    assert event_dict["message"] == "Test event"
    assert event_dict["details"] == {"key": "value"}
    assert not event_dict["acknowledged"]


def test_event_from_dict():
    """Test event creation from dict."""
    event_dict = {
        "event_id": "test_id",
        "timestamp": "2025-01-01T12:00:00",
        "category": "process",
        "severity": "warning",
        "source": "test_source",
        "message": "Test event",
        "details": {"key": "value"},
        "acknowledged": True,
    }

    event = Event.from_dict(event_dict)

    assert event.event_id == "test_id"
    assert event.timestamp == "2025-01-01T12:00:00"
    assert event.category == EventCategory.PROCESS
    assert event.severity == EventSeverity.WARNING
    assert event.source == "test_source"
    assert event.message == "Test event"
    assert event.details == {"key": "value"}
    assert event.acknowledged


def test_system_state_event():
    """Test system state event creation."""
    event = SystemStateEvent(
        previous_state="idle", new_state="running", source="test_source"
    )

    assert event.category == EventCategory.SYSTEM
    assert event.severity == EventSeverity.INFO
    assert event.source == "test_source"
    assert "idle" in event.message
    assert "running" in event.message
    assert event.details["previous_state"] == "idle"
    assert event.details["new_state"] == "running"


def test_threshold_event():
    """Test threshold event creation."""
    event = ThresholdEvent(
        parameter="temperature",
        threshold=25.0,
        actual_value=30.0,
        severity=EventSeverity.WARNING,
        source="test_source",
    )

    assert event.category == EventCategory.THRESHOLD
    assert event.severity == EventSeverity.WARNING
    assert event.source == "test_source"
    assert "temperature" in event.message
    assert "25.0" in event.message
    assert "30.0" in event.message
    assert event.details["parameter"] == "temperature"
    assert event.details["threshold"] == 25.0
    assert event.details["actual_value"] == 30.0


def test_optimization_event():
    """Test optimization event creation."""
    event = OptimizationEvent(
        optimization_type="energy",
        potential_improvement=15.5,
        recommended_action="Adjust parameters",
        source="test_source",
    )

    assert event.category == EventCategory.OPTIMIZATION
    assert event.severity == EventSeverity.INFO
    assert event.source == "test_source"
    assert "energy" in event.message
    assert "15.5" in event.message
    assert event.details["optimization_type"] == "energy"
    assert event.details["potential_improvement"] == 15.5
    assert event.details["recommended_action"] == "Adjust parameters"


def test_error_event():
    """Test error event creation."""
    event = ErrorEvent(
        error_type="ValueError",
        error_message="Invalid parameter",
        stacktrace="test stacktrace",
        severity=EventSeverity.ERROR,
        source="test_source",
    )

    assert event.category == EventCategory.ERROR
    assert event.severity == EventSeverity.ERROR
    assert event.source == "test_source"
    assert "ValueError" in event.message
    assert "Invalid parameter" in event.message
    assert event.details["error_type"] == "ValueError"
    assert event.details["error_message"] == "Invalid parameter"
    assert event.details["stacktrace"] == "test stacktrace"


def test_user_action_event():
    """Test user action event creation."""
    event = UserActionEvent(
        user_id="test_user",
        action="login",
        action_details={"ip": "127.0.0.1"},
        source="test_source",
    )

    assert event.category == EventCategory.USER
    assert event.severity == EventSeverity.INFO
    assert event.source == "test_source"
    assert "test_user" in event.message
    assert "login" in event.message
    assert event.details["user_id"] == "test_user"
    assert event.details["action"] == "login"
    assert event.details["action_details"] == {"ip": "127.0.0.1"}
