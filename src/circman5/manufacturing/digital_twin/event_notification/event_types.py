# src/circman5/manufacturing/digital_twin/event_notification/event_types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import datetime
import uuid


class EventSeverity(Enum):
    """Severity levels for events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventCategory(Enum):
    """Categories for events."""

    SYSTEM = "system"
    PROCESS = "process"
    OPTIMIZATION = "optimization"
    THRESHOLD = "threshold"
    USER = "user"
    ERROR = "error"


@dataclass
class Event:
    """Base class for all events in the system."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    category: EventCategory = EventCategory.SYSTEM
    severity: EventSeverity = EventSeverity.INFO
    source: str = "system"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "severity": self.severity.value,
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.datetime.now().isoformat()),
            category=EventCategory(data.get("category", EventCategory.SYSTEM.value)),
            severity=EventSeverity(data.get("severity", EventSeverity.INFO.value)),
            source=data.get("source", "system"),
            message=data.get("message", ""),
            details=data.get("details", {}),
            acknowledged=data.get("acknowledged", False),
        )


@dataclass
class SystemStateEvent(Event):
    """Event for system state changes."""

    def __init__(self, previous_state: str, new_state: str, **kwargs):
        # Extract details from kwargs but don't keep them in kwargs
        original_details = kwargs.pop("details", {})

        # Create new details dictionary combining our details with any originals
        details = {
            "previous_state": previous_state,
            "new_state": new_state,
            **original_details,
        }

        super().__init__(
            category=EventCategory.SYSTEM,
            message=f"System state changed from {previous_state} to {new_state}",
            details=details,
            **kwargs,
        )


@dataclass
class ThresholdEvent(Event):
    """Event for threshold breaches."""

    def __init__(self, parameter: str, threshold: float, actual_value: float, **kwargs):
        # Extract severity but don't keep it in kwargs
        severity = kwargs.pop("severity", EventSeverity.WARNING)

        # Extract details from kwargs
        original_details = kwargs.pop("details", {})

        # Create new details dictionary combining our details with any originals
        details = {
            "parameter": parameter,
            "threshold": threshold,
            "actual_value": actual_value,
            **original_details,
        }

        super().__init__(
            category=EventCategory.THRESHOLD,
            severity=severity,
            message=f"Threshold breach: {parameter} exceeded {threshold} with value {actual_value}",
            details=details,
            **kwargs,
        )


@dataclass
class OptimizationEvent(Event):
    """Event for optimization opportunities."""

    def __init__(
        self,
        optimization_type: str,
        potential_improvement: float,
        recommended_action: str,
        **kwargs,
    ):
        severity = kwargs.pop("severity", EventSeverity.INFO)
        original_details = kwargs.pop("details", {})

        details = {
            "optimization_type": optimization_type,
            "potential_improvement": potential_improvement,
            "recommended_action": recommended_action,
            **original_details,
        }

        super().__init__(
            category=EventCategory.OPTIMIZATION,
            severity=severity,
            message=f"Optimization opportunity: {optimization_type} with {potential_improvement:.2f}% improvement potential",
            details=details,
            **kwargs,
        )


@dataclass
class ErrorEvent(Event):
    """Event for system errors."""

    def __init__(
        self,
        error_type: str,
        error_message: str,
        stacktrace: Optional[str] = None,
        **kwargs,
    ):
        severity = kwargs.pop("severity", EventSeverity.ERROR)
        original_details = kwargs.pop("details", {})

        details = {
            "error_type": error_type,
            "error_message": error_message,
            "stacktrace": stacktrace,
            **original_details,
        }

        super().__init__(
            category=EventCategory.ERROR,
            severity=severity,
            message=f"Error: {error_type} - {error_message}",
            details=details,
            **kwargs,
        )


@dataclass
class UserActionEvent(Event):
    """Event for user actions."""

    def __init__(
        self, user_id: str, action: str, action_details: Dict[str, Any], **kwargs
    ):
        severity = kwargs.pop("severity", EventSeverity.INFO)
        original_details = kwargs.pop("details", {})

        details = {
            "user_id": user_id,
            "action": action,
            "action_details": action_details,
            **original_details,
        }

        super().__init__(
            category=EventCategory.USER,
            severity=severity,
            message=f"User {user_id} performed action: {action}",
            details=details,
            **kwargs,
        )
