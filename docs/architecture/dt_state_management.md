# Digital Twin State Management Architecture

## 1. Overview

The State Management system forms the backbone of the CIRCMAN5.0 Digital Twin architecture. It provides a robust foundation for maintaining, tracking, validating, and persisting the digital twin's state across its lifecycle. This document details the architectural approach to state management within the system.

## 2. Core Principles

The state management architecture is built on these fundamental principles:

1. **Single Source of Truth**: The system maintains one definitive representation of the digital twin state at any given time
2. **Historical State Tracking**: All state changes are tracked in a configurable history
3. **Immutable State**: States are treated as immutable snapshots, creating new states rather than modifying existing ones
4. **Validation**: All states undergo validation to ensure data integrity
5. **Persistence**: States can be exported and imported for continuity across sessions
6. **Thread Safety**: Concurrent access is properly managed to prevent race conditions

## 3. State Structure

### 3.1 State Representation

The digital twin state is represented as a hierarchical dictionary (JSON-compatible) structure:

```json
{
    "timestamp": "2025-02-24T14:30:22.123456",
    "system_status": "running",
    "production_line": {
        "status": "running",
        "temperature": 22.5,
        "energy_consumption": 120.5,
        "production_rate": 8.3,
        "efficiency": 0.92,
        "defect_rate": 0.02
    },
    "materials": {
        "silicon_wafer": {
            "inventory": 850,
            "quality": 0.95
        },
        "solar_glass": {
            "inventory": 420,
            "quality": 0.98
        }
    },
    "environment": {
        "temperature": 22.0,
        "humidity": 45.0
    }
}
```

### 3.2 State Components

The state is organized into these primary components:

1. **Metadata**: Timestamp and system status information
2. **Production Line**: Process-specific parameters and status
3. **Materials**: Inventory levels and quality information
4. **Environment**: Environmental conditions affecting manufacturing
5. **Quality Metrics**: Product quality indicators
6. **Resource Metrics**: Resource utilization information

## 4. State Manager Architecture

### 4.1 Component Design

The StateManager is implemented as a singleton with thread safety to ensure consistent state access throughout the system:

```
StateManager
├── Attributes
│   ├── current_state: Dict[str, Any]
│   ├── state_history: deque
│   ├── history_length: int
│   ├── _lock: threading.RLock
│   └── constants: ConstantsService
└── Methods
    ├── set_state(state)
    ├── update_state(updates)
    ├── get_current_state()
    ├── get_history(limit)
    ├── get_state_at_time(timestamp)
    ├── clear_history()
    ├── validate_state(state)
    ├── export_state(file_path)
    └── import_state(file_path)
```

### 4.2 Singleton Pattern Implementation

The StateManager uses a thread-safe singleton pattern to ensure only one instance exists throughout the application:

```python
def __new__(cls, *args, **kwargs):
    """Ensure only one instance is created."""
    if cls._instance is None:
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(StateManager, cls).__new__(cls)
    return cls._instance
```

## 5. State Change Management

### 5.1 State Update Workflow

The state update process follows these steps:

1. **Validation**: The new state or updates are validated
2. **History Management**: The current state is added to history
3. **State Application**: The new state becomes the current state
4. **Event Publication**: State change events are published to the event system
5. **Resource Management**: History length is managed to prevent memory issues

### 5.2 State Validation

State validation ensures all states meet the required structure and data types:

1. State must be a dictionary
2. Timestamp must be in ISO format if present
3. Required fields must be present
4. Numeric values must be within acceptable ranges
5. Nested objects must follow the defined structure

## 6. History Management

### 6.1 History Storage

State history is maintained in a double-ended queue (deque) for efficient operations:

```python
self.state_history = collections.deque(maxlen=self.history_length)
```

### 6.2 History Operations

The system supports these history operations:

1. **Addition**: New states are added to history during updates
2. **Retrieval**: Historical states can be retrieved with optional limits
3. **Time-based Access**: States can be retrieved by timestamp
4. **Clearing**: History can be cleared when needed
5. **Length Management**: The history length is configurable

## 7. State Persistence

### 7.1 Persistence Operations

The state persistence system provides:

1. **Export**: Current state can be exported to a JSON file
2. **Import**: States can be imported from previously exported files
3. **Timestamps**: All persisted states include timestamps for tracking
4. **Validation**: Imported states undergo validation
5. **Path Management**: File operations are handled through the results_manager

### 7.2 Implementation

```python
def export_state(self, file_path=None):
    """Export the current state to a JSON file."""
    with self._lock:
        state_copy = copy.deepcopy(self.current_state)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(state_copy, temp_file, indent=2)
        save_path = temp_file.name

    # Use results manager to save to appropriate location
    if file_path:
        shutil.copy2(save_path, file_path)
    else:
        from ...utils.results_manager import results_manager
        file_path = results_manager.save_file(save_path, "digital_twin")

    os.unlink(save_path)  # Remove temp file
    return True
```

## 8. Thread Safety Considerations

### 8.1 Lock Implementation

A reentrant lock protects state access and modification:

```python
self._lock = threading.RLock()
```

### 8.2 Synchronized Operations

All state operations are protected by the lock:

```python
def set_state(self, state):
    """Set the current state and add to history."""
    valid, message = self.validate_state(state)
    if not valid:
        self.logger.warning(f"Invalid state: {message}")
        return

    with self._lock:
        if self.current_state:
            self.state_history.append(copy.deepcopy(self.current_state))
        self.current_state = copy.deepcopy(state)
```

## 9. Integration with Other Components

### 9.1 Digital Twin Core

The StateManager is primarily used by the Digital Twin Core for state tracking:

```python
# In DigitalTwin class
self.state_manager = StateManager(history_length=self.config.history_length)
current_state = self.state_manager.get_current_state()
```

### 9.2 Event System

State changes can trigger events through the Digital Twin's event publisher:

```python
# After state update
self.event_publisher.publish_state_update(previous_state, updated_state)
```

### 9.3 Simulation Engine

The simulation engine uses the state manager to access current state for simulations:

```python
# In SimulationEngine
def run_simulation(self, steps=10, initial_state=None, parameters=None):
    # Get state from state manager if not provided
    if initial_state is None:
        initial_state = self.state_manager.get_current_state()
```

### 9.4 Human Interface

The interface system accesses state through an adapter:

```python
# In digital_twin_adapter
def get_current_state(self):
    try:
        return self.digital_twin.get_current_state()
    except Exception as e:
        self.logger.error(f"Error getting current state: {str(e)}")
        return {"system_status": "error", "error": str(e)}
```

## 10. Mathematical Foundation

The state tracking system implements a discrete-time state history model where:

Let $S_t$ be the state at time $t$, with $t \in \{0, 1, 2, ..., T\}$

The state history is then the sequence:
$H = \{S_0, S_1, S_2, ..., S_T\}$

Where $S_T$ is the current state, and $S_0, S_1, ..., S_{T-1}$ are historical states.

## 11. Performance Considerations

1. Deep copies are used to prevent reference issues
2. The deque data structure provides efficient history tracking with O(1) operations
3. History length is configurable to manage memory usage
4. Validation is selectively applied to maintain performance
5. State updates use recursive deep merging for efficiency
6. Locks are held for minimal durations to reduce contention
