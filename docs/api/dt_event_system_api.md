# Digital Twin Event System API Reference

## 1. Introduction

The Event System API provides a publisher-subscriber pattern for event distribution across the CIRCMAN5.0 Digital Twin system. This document covers the event types, event management, and usage patterns for the event notification system.

## 2. Event Types

### 2.1 EventCategory Enumeration

Defines categories for events throughout the system:

```python
from circman5.manufacturing.digital_twin.event_notification.event_types import EventCategory

class EventCategory(Enum):
    """Categories for events."""
    SYSTEM = "system"         # System-level events
    PROCESS = "process"       # Manufacturing process events
    OPTIMIZATION = "optimization"  # Optimization-related events
    THRESHOLD = "threshold"   # Parameter threshold breach events
    USER = "user"             # User action events
    ERROR = "error"           # Error events
```

### 2.2 EventSeverity Enumeration

Defines severity levels for events:

```python
from circman5.manufacturing.digital_twin.event_notification.event_types import EventSeverity

class EventSeverity(Enum):
    """Severity levels for events."""
    INFO = "info"           # Informational, no action required
    WARNING = "warning"     # Potential issue, attention may be needed
    ERROR = "error"         # Error condition, action required
    CRITICAL = "critical"   # Critical condition, immediate action required
```

### 2.3 Event Classes

#### 2.3.1 Base Event Class

```python
from circman5.manufacturing.digital_twin.event_notification.event_types import Event

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
```

#### 2.3.2 Specialized Event Types

##### SystemStateEvent

```python
@dataclass
class SystemStateEvent(Event):
    """Event for system state changes."""
    def __init__(self, previous_state: str, new_state: str, **kwargs):
        details = {
            "previous_state": previous_state,
            "new_state": new_state,
            **kwargs.pop("details", {})
        }

        super().__init__(
            category=EventCategory.SYSTEM,
            message=f"System state changed from {previous_state} to {new_state}",
            details=details,
            **kwargs
        )
```

##### ThresholdEvent

```python
@dataclass
class ThresholdEvent(Event):
    """Event for threshold breaches."""
    def __init__(self, parameter: str, threshold: float, actual_value: float, **kwargs):
        severity = kwargs.pop("severity", EventSeverity.WARNING)

        details = {
            "parameter": parameter,
            "threshold": threshold,
            "actual_value": actual_value,
            **kwargs.pop("details", {})
        }

        super().__init__(
            category=EventCategory.THRESHOLD,
            severity=severity,
            message=f"Threshold breach: {parameter} exceeded {threshold} with value {actual_value}",
            details=details,
            **kwargs
        )
```

##### OptimizationEvent

```python
@dataclass
class OptimizationEvent(Event):
    """Event for optimization opportunities."""
    def __init__(
        self,
        optimization_type: str,
        potential_improvement: float,
        recommended_action: str,
        **kwargs
    ):
        severity = kwargs.pop("severity", EventSeverity.INFO)

        details = {
            "optimization_type": optimization_type,
            "potential_improvement": potential_improvement,
            "recommended_action": recommended_action,
            **kwargs.pop("details", {})
        }

        super().__init__(
            category=EventCategory.OPTIMIZATION,
            severity=severity,
            message=f"Optimization opportunity: {optimization_type} with {potential_improvement:.2f}% improvement potential",
            details=details,
            **kwargs
        )
```

##### ErrorEvent

```python
@dataclass
class ErrorEvent(Event):
    """Event for system errors."""
    def __init__(
        self,
        error_type: str,
        error_message: str,
        stacktrace: Optional[str] = None,
        **kwargs
    ):
        severity = kwargs.pop("severity", EventSeverity.ERROR)

        details = {
            "error_type": error_type,
            "error_message": error_message,
            "stacktrace": stacktrace,
            **kwargs.pop("details", {})
        }

        super().__init__(
            category=EventCategory.ERROR,
            severity=severity,
            message=f"Error: {error_type} - {error_message}",
            details=details,
            **kwargs
        )
```

##### UserActionEvent

```python
@dataclass
class UserActionEvent(Event):
    """Event for user actions."""
    def __init__(
        self, user_id: str, action: str, action_details: Dict[str, Any], **kwargs
    ):
        severity = kwargs.pop("severity", EventSeverity.INFO)

        details = {
            "user_id": user_id,
            "action": action,
            "action_details": action_details,
            **kwargs.pop("details", {})
        }

        super().__init__(
            category=EventCategory.USER,
            severity=severity,
            message=f"User {user_id} performed action: {action}",
            details=details,
            **kwargs
        )
```

## 3. Event Manager

The EventManager provides a central hub for event distribution.

```python
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
```

### 3.1 Subscription Management

```python
# EventHandler type definition
EventHandler = Callable[[Event], None]
```

#### 3.1.1 subscribe()

```python
def subscribe(
    handler: EventHandler,
    category: Optional[EventCategory] = None,
    severity: Optional[EventSeverity] = None
) -> None
```

Subscribes a handler to events.

**Parameters:**
- `handler` (EventHandler): Event handler function.
- `category` (EventCategory, optional): Optional category to subscribe to (all if None).
- `severity` (EventSeverity, optional): Optional severity to subscribe to (all if None).

#### 3.1.2 unsubscribe()

```python
def unsubscribe(
    handler: EventHandler,
    category: Optional[EventCategory] = None,
    severity: Optional[EventSeverity] = None
) -> None
```

Unsubscribes a handler from events.

**Parameters:**
- `handler` (EventHandler): Event handler function to unsubscribe.
- `category` (EventCategory, optional): Optional category to unsubscribe from (all if None).
- `severity` (EventSeverity, optional): Optional severity to unsubscribe from (all if None).

### 3.2 Event Filtering

```python
# EventFilter type definition
EventFilter = Callable[[Event], bool]
```

#### 3.2.1 add_filter()

```python
def add_filter(category: EventCategory, filter_func: EventFilter) -> None
```

Adds a filter for a specific event category.

**Parameters:**
- `category` (EventCategory): Event category to filter.
- `filter_func` (EventFilter): Filter function that returns True for events to be processed.

#### 3.2.2 remove_filter()

```python
def remove_filter(category: EventCategory, filter_func: EventFilter) -> None
```

Removes a filter for a specific event category.

**Parameters:**
- `category` (EventCategory): Event category to remove filter from.
- `filter_func` (EventFilter): Filter function to remove.

### 3.3 Event Publishing

#### 3.3.1 publish()

```python
def publish(event: Event) -> None
```

Publishes an event to all subscribed handlers.

**Parameters:**
- `event` (Event): Event to publish.

### 3.4 Event Retrieval and Management

#### 3.4.1 get_events()

```python
def get_events(
    category: Optional[EventCategory] = None,
    severity: Optional[EventSeverity] = None,
    limit: int = 100
) -> List[Event]
```

Gets events from persistence with optional filtering.

**Parameters:**
- `category` (EventCategory, optional): Optional category filter.
- `severity` (EventSeverity, optional): Optional severity filter.
- `limit` (int): Maximum number of events to return.

**Returns:**
- `List[Event]`: List of events matching the criteria.

#### 3.4.2 acknowledge_event()

```python
def acknowledge_event(event_id: str) -> bool
```

Marks an event as acknowledged.

**Parameters:**
- `event_id` (str): ID of the event to acknowledge.

**Returns:**
- `bool`: True if event was found and acknowledged.

#### 3.4.3 clear_events()

```python
def clear_events(older_than_days: Optional[int] = None) -> int
```

Clears events from persistence.

**Parameters:**
- `older_than_days` (int, optional): Optional, clear events older than this many days.

**Returns:**
- `int`: Number of events cleared.

## 4. Event Publishers

The system provides specialized event publishers for different components.

### 4.1 EventPublisherBase

```python
from circman5.manufacturing.digital_twin.event_notification.publishers import EventPublisherBase

class EventPublisherBase:
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.event_manager = event_manager

    def publish(self, event: Event) -> None:
        """Publish an event."""
        if not hasattr(event, 'source') or not event.source:
            event.source = self.source_name
        self.event_manager.publish(event)
```

### 4.2 DigitalTwinPublisher

```python
from circman5.manufacturing.digital_twin.event_notification.publishers import DigitalTwinPublisher

class DigitalTwinPublisher(EventPublisherBase):
    def __init__(self):
        super().__init__("digital_twin")

    def publish_state_update(self, previous_state, updated_state):
        """Publish state update event."""
        event = SystemStateEvent(
            previous_state=previous_state.get("system_status", "unknown"),
            new_state=updated_state.get("system_status", "unknown"),
            severity=EventSeverity.INFO
        )
        self.publish(event)

    def publish_parameter_threshold_event(
        self, parameter_path, parameter_name,
        threshold, actual_value, state, severity=EventSeverity.WARNING
    ):
        """Publish threshold breach event."""
        event = ThresholdEvent(
            parameter=parameter_name,
            threshold=threshold,
            actual_value=actual_value,
            severity=severity,
            details={
                "parameter_path": parameter_path,
                "state_timestamp": state.get("timestamp", "unknown")
            }
        )
        self.publish(event)

    def publish_error_event(
        self, error_type, error_message,
        severity=EventSeverity.ERROR, stacktrace=None
    ):
        """Publish error event."""
        event = ErrorEvent(
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            stacktrace=stacktrace
        )
        self.publish(event)

    def publish_simulation_result(
        self, simulation_id, parameters, results, improvement
    ):
        """Publish simulation result event."""
        event = Event(
            category=EventCategory.PROCESS,
            severity=EventSeverity.INFO,
            message=f"Simulation {simulation_id} completed with {improvement:.2f}% improvement",
            details={
                "simulation_id": simulation_id,
                "parameters": parameters,
                "results_summary": results,
                "improvement": improvement
            }
        )
        self.publish(event)
```

## 5. Usage Examples

### 5.1 Subscribing to Events

```python
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.digital_twin.event_notification.event_types import EventCategory, EventSeverity

# Define event handler
def handle_threshold_event(event):
    print(f"Threshold breach: {event.message}")
    print(f"Parameter: {event.details['parameter']}")
    print(f"Value: {event.details['actual_value']}")
    print(f"Threshold: {event.details['threshold']}")

# Subscribe to threshold events
event_manager.subscribe(
    handle_threshold_event,
    category=EventCategory.THRESHOLD
)

# Subscribe to critical events regardless of category
def handle_critical_event(event):
    print(f"CRITICAL: {event.message}")

event_manager.subscribe(
    handle_critical_event,
    severity=EventSeverity.CRITICAL
)
```

### 5.2 Publishing Events

```python
from circman5.manufacturing.digital_twin.event_notification.event_types import ThresholdEvent, EventSeverity
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager

# Create a threshold event
threshold_event = ThresholdEvent(
    parameter="temperature",
    threshold=25.0,
    actual_value=26.5,
    severity=EventSeverity.WARNING
)

# Publish the event
event_manager.publish(threshold_event)
```

### 5.3 Using Digital Twin Publisher

```python
from circman5.manufacturing.digital_twin.event_notification.publishers import DigitalTwinPublisher
from circman5.manufacturing.digital_twin.event_notification.event_types import EventSeverity

# Create publisher
publisher = DigitalTwinPublisher()

# Publish threshold breach
publisher.publish_parameter_threshold_event(
    parameter_path="production_line.temperature",
    parameter_name="Temperature",
    threshold=25.0,
    actual_value=26.5,
    state={"timestamp": "2025-02-24T14:30:22.123456"},
    severity=EventSeverity.WARNING
)

# Publish error
publisher.publish_error_event(
    error_type="SimulationError",
    error_message="Simulation failed to converge",
    severity=EventSeverity.ERROR
)
```

### 5.4 Event Filtering

```python
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.digital_twin.event_notification.event_types import EventCategory

# Define a filter for threshold events
def temperature_filter(event):
    # Only process temperature threshold events
    if "parameter" in event.details and event.details["parameter"] == "temperature":
        return True
    return False

# Add the filter
event_manager.add_filter(EventCategory.THRESHOLD, temperature_filter)
```

### 5.5 Event Retrieval

```python
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.digital_twin.event_notification.event_types import EventCategory, EventSeverity

# Get recent warning events
warning_events = event_manager.get_events(
    severity=EventSeverity.WARNING,
    limit=10
)

# Get threshold events
threshold_events = event_manager.get_events(
    category=EventCategory.THRESHOLD,
    limit=20
)

# Acknowledge an event
event_manager.acknowledge_event("event_id_123456")

# Clear old events
event_manager.clear_events(older_than_days=30)
```

## 6. Thread Safety

The Event Notification System is designed to be thread-safe:

1. Event manager operations are protected by locks
2. Event publishing is thread-safe
3. Subscription management is synchronized
4. Event handlers are executed outside locks to prevent blocking

Example of thread-safe usage:

```python
import threading
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.digital_twin.event_notification.event_types import Event, EventCategory, EventSeverity

# Define handler
def event_handler(event):
    print(f"Received event: {event.message}")

# Subscribe in main thread
event_manager.subscribe(event_handler, category=EventCategory.SYSTEM)

# Function to publish events from worker thread
def publish_worker():
    for i in range(5):
        event = Event(
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message=f"Worker thread event {i}",
            details={"thread_id": threading.current_thread().ident}
        )
        event_manager.publish(event)

# Create and start worker thread
worker = threading.Thread(target=publish_worker)
worker.start()
worker.join()
```

## 7. Best Practices

### 7.1 Event Creation

1. Use the most specific event type for the situation
2. Provide clear, descriptive messages
3. Include all relevant details in the details dictionary
4. Use appropriate severity levels

### 7.2 Event Handling

1. Keep event handlers lightweight
2. Avoid blocking operations in handlers
3. Handle exceptions within handlers
4. Use filtering to focus on relevant events

### 7.3 Event Publishers

1. Create specialized publishers for specific components
2. Use consistent source naming
3. Include context information in events
4. Consider event volume and performance implications

## 8. Configuration

The Event Notification System is configurable through the constants service:

```json
"EVENT_NOTIFICATION": {
    "persistence_enabled": true,
    "max_events": 1000,
    "default_alert_severity": "warning",
    "publish_state_changes": true,
    "publish_threshold_breaches": true,
    "publish_simulation_results": true
}
```

This configuration can be accessed through the constants service:

```python
from circman5.adapters.services.constants_service import ConstantsService

constants = ConstantsService()
event_config = constants.get_digital_twin_config().get("EVENT_NOTIFICATION", {})
persistence_enabled = event_config.get("persistence_enabled", True)
max_events = event_config.get("max_events", 1000)
```
