# Digital Twin Troubleshooting Guide

## 1. Introduction

This troubleshooting guide addresses common issues encountered when working with the CIRCMAN5.0 Digital Twin system. It provides diagnostic procedures, solutions to common problems, and guidance for resolving complex issues during development, integration, and operation.

## 2. System Diagnostics

### 2.1 Checking System Health

#### 2.1.1 Digital Twin Core Status

To check if the Digital Twin Core is properly initialized and running:

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

# Get digital twin instance
twin = DigitalTwin()

# Check if initialized
is_running = hasattr(twin, "is_running") and twin.is_running
print(f"Digital Twin running: {is_running}")

# Check current state
state = twin.get_current_state()
print(f"State timestamp: {state.get('timestamp', 'No timestamp')}")
print(f"System status: {state.get('system_status', 'Unknown')}")
```

#### 2.1.2 Event System Status

To check the status of the Event Notification System:

```python
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.digital_twin.event_notification.event_types import EventCategory

# Get recent events
events = event_manager.get_events(limit=10)
print(f"Recent events: {len(events)}")

# Check event categories
category_counts = {}
for event in events:
    category = event.category.value
    category_counts[category] = category_counts.get(category, 0) + 1

print("Event category distribution:", category_counts)
```

#### 2.1.3 Configuration Status

To check if configuration is properly loaded:

```python
from circman5.adapters.services.constants_service import ConstantsService

# Check configuration loading
constants = ConstantsService()
try:
    dt_config = constants.get_digital_twin_config()
    print("Digital Twin configuration loaded successfully")
    print(f"Keys: {list(dt_config.keys())}")
except Exception as e:
    print(f"Error loading Digital Twin configuration: {str(e)}")
```

### 2.2 Enabling Detailed Logging

#### 2.2.1 Setting Up Debug Logging

```python
import logging

# Configure root logger
logging.basicConfig(level=logging.DEBUG)

# Set specific loggers to DEBUG
loggers = [
    "digital_twin_core",
    "state_manager",
    "simulation_engine",
    "event_manager",
    "ai_integration",
    "lca_integration"
]

for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
```

#### 2.2.2 Analyzing Log Files

```python
def analyze_logs(log_file):
    """Analyze log file for common issues."""
    error_patterns = [
        "Error", "Exception", "Failed", "Invalid",
        "Could not", "Unable to", "TimeoutError"
    ]

    errors = []
    warnings = []

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            for pattern in error_patterns:
                if pattern in line:
                    errors.append((line_num, line.strip()))
                    break
            if "WARNING" in line:
                warnings.append((line_num, line.strip()))

    print(f"Found {len(errors)} errors and {len(warnings)} warnings")
    print("\nTop 5 errors:")
    for i, (line_num, line) in enumerate(errors[:5], 1):
        print(f"{i}. Line {line_num}: {line}")

    print("\nTop 5 warnings:")
    for i, (line_num, line) in enumerate(warnings[:5], 1):
        print(f"{i}. Line {line_num}: {line}")

    # Count occurrences of specific error types
    error_types = {}
    for _, line in errors:
        for err_type in ["ValueError", "KeyError", "TypeError", "IndexError", "AttributeError"]:
            if err_type in line:
                error_types[err_type] = error_types.get(err_type, 0) + 1
                break

    print("\nError type distribution:")
    for err_type, count in error_types.items():
        print(f"{err_type}: {count}")
```

## 3. Common Issues and Solutions

### 3.1 Initialization Issues

#### 3.1.1 Digital Twin Not Initializing

**Symptoms:**
- `AttributeError` when accessing digital twin methods
- Error messages about "digital twin not initialized"
- Digital twin methods returning None or errors

**Causes:**
- Failure to call `initialize()` before using the digital twin
- Configuration errors preventing initialization
- Dependencies not properly initialized

**Solutions:**

1. Check initialization status:
```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

# Get digital twin instance
twin = DigitalTwin()

# Check if initialized
initialized = hasattr(twin, "_initialized") and twin._initialized
is_running = hasattr(twin, "is_running") and twin.is_running

print(f"Digital Twin initialized: {initialized}")
print(f"Digital Twin running: {is_running}")
```

2. Attempt manual initialization:
```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

# Get digital twin instance
twin = DigitalTwin()

# Force re-initialization
try:
    # Reset singleton state (for diagnostic purposes only)
    if hasattr(DigitalTwin, "_reset"):
        DigitalTwin._reset()

    # Create new instance and initialize
    twin = DigitalTwin()
    init_success = twin.initialize()

    print(f"Initialization success: {init_success}")
    if not init_success:
        print("Initialization failed, check logs for details")
except Exception as e:
    print(f"Error during initialization: {str(e)}")
```

3. Check state manager:
```python
from circman5.manufacturing.digital_twin.core.state_manager import StateManager

# Get state manager instance
state_manager = StateManager()

# Check if initialized
initialized = hasattr(state_manager, "_initialized") and state_manager._initialized

print(f"State Manager initialized: {initialized}")
```

#### 3.1.2 Singleton Pattern Issues

**Symptoms:**
- Multiple instances seem to exist
- State changes in one part of code not visible in another
- Inconsistent behavior when accessing components

**Causes:**
- Improper singleton implementation
- Module reloading in development environments
- Missing thread safety

**Solutions:**

1. Check singleton implementation:
```python
def check_singleton_implementation(singleton_class):
    """Check if singleton implementation is correct."""
    # Create two instances
    instance1 = singleton_class()
    instance2 = singleton_class()

    # Check if they are the same object
    is_same = instance1 is instance2
    print(f"Instances are the same object: {is_same}")

    # Check key attributes
    attrs = ["_instance", "_initialized"]
    for attr in attrs:
        if hasattr(singleton_class, attr):
            print(f"Class has {attr}: Yes")
        else:
            print(f"Class has {attr}: No")

    return is_same
```

2. Reset singleton before tests:
```python
# For DigitalTwin
if hasattr(DigitalTwin, "_reset_instance"):
    DigitalTwin._reset_instance()
elif hasattr(DigitalTwin, "_instance"):
    DigitalTwin._instance = None

# For StateManager
if hasattr(StateManager, "_reset_instance"):
    StateManager._reset_instance()
elif hasattr(StateManager, "_instance"):
    StateManager._instance = None
```

### 3.2 Configuration Issues

#### 3.2.1 Missing Configuration Files

**Symptoms:**
- `FileNotFoundError` when initializing components
- Error messages mentioning "config file not found"
- Components using default values instead of configured ones

**Causes:**
- Configuration files not in expected location
- File permissions issues
- Path resolution problems

**Solutions:**

1. Check configuration file paths:
```python
from circman5.adapters.config.digital_twin import DigitalTwinAdapter
adapter = DigitalTwinAdapter()
print(f"Looking for config at: {adapter.config_path}")
print(f"File exists: {adapter.config_path.exists()}")
```

2. Create missing configuration files:
```python
from pathlib import Path
import json
from circman5.adapters.config.digital_twin import DigitalTwinAdapter

adapter = DigitalTwinAdapter()
config_path = adapter.config_path
config_dir = config_path.parent

# Create directory if missing
config_dir.mkdir(parents=True, exist_ok=True)

# Get default config
default_config = adapter.get_defaults()

# Write to file
with open(config_path, 'w') as f:
    json.dump(default_config, f, indent=2)

print(f"Created default configuration at: {config_path}")
```

#### 3.2.2 Invalid Configuration Values

**Symptoms:**
- Warning logs about falling back to default values
- Unexpected behavior in component operation
- `ValueError` or `TypeError` exceptions

**Causes:**
- Manually edited configuration with invalid values
- Missing required configuration keys
- Type mismatches in configuration values

**Solutions:**

1. Validate configuration with adapter:
```python
from circman5.adapters.config.digital_twin import DigitalTwinAdapter

adapter = DigitalTwinAdapter()
config = adapter.load_config()

# Manually validate
is_valid = adapter.validate_config(config)
print(f"Configuration valid: {is_valid}")

# Check specific values
if "DIGITAL_TWIN_CONFIG" in config:
    dt_config = config["DIGITAL_TWIN_CONFIG"]
    print(f"Update frequency: {dt_config.get('update_frequency')}")
    print(f"History length: {dt_config.get('history_length')}")
```

2. Reset configuration to defaults:
```python
import json
from circman5.adapters.config.digital_twin import DigitalTwinAdapter

adapter = DigitalTwinAdapter()
default_config = adapter.get_defaults()

# Write defaults to file
with open(adapter.config_path, 'w') as f:
    json.dump(default_config, f, indent=2)

print(f"Reset configuration to defaults at: {adapter.config_path}")
```

### 3.3 State Management Issues

#### 3.3.1 State History Loss

**Symptoms:**
- Missing historical states
- History shorter than expected
- State retrieval returning fewer items than expected

**Causes:**
- Incorrect history_length configuration
- State manager reinitialization
- Memory constraints

**Solutions:**

1. Check history configuration:
```python
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
from circman5.adapters.services.constants_service import ConstantsService

# Get configured history length
constants = ConstantsService()
dt_config = constants.get_digital_twin_config()
state_config = dt_config.get("STATE_MANAGEMENT", {})
configured_length = state_config.get("default_history_length", 1000)

# Get actual history length
state_manager = StateManager()
actual_length = state_manager.history_length

print(f"Configured history length: {configured_length}")
print(f"Actual history length: {actual_length}")
```

2. Check current history size:
```python
from circman5.manufacturing.digital_twin.core.state_manager import StateManager

# Get state manager instance
state_manager = StateManager()

# Check history size
history = state_manager.get_history()
print(f"Current history size: {len(history)}")

# Check memory usage (rough estimate)
import sys
history_memory = sys.getsizeof(history)
print(f"Approximate history memory usage: {history_memory / (1024*1024):.2f} MB")
```

3. Monitor state history updates:
```python
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
import time

# Get state manager
state_manager = StateManager()

# Current history size
initial_size = len(state_manager.get_history())
print(f"Initial history size: {initial_size}")

# Monitor for 30 seconds
start_time = time.time()
end_time = start_time + 30
while time.time() < end_time:
    current_size = len(state_manager.get_history())
    if current_size != initial_size:
        print(f"History size changed: {initial_size} -> {current_size}")
        initial_size = current_size
    time.sleep(1)
```

#### This seems to be a critical issue that needs explanation:
- Check if state_manager.get_history() returns the expected number of entries
- Verify if digital_twin.get_state_history() is functioning properly
- Ensure history isn't being cleared inadvertently by another process

#### 3.3.2 State Validation Failures

**Symptoms:**
- Warning logs about invalid states
- Unexpected behavior in state-dependent components
- Error messages about state validation

**Causes:**
- Incorrect state structure
- Missing required fields
- Invalid field values

**Solutions:**

1. Check current state structure:
```python
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
import json

# Get state manager instance
state_manager = StateManager()

# Get current state
current_state = state_manager.get_current_state()

# Print pretty state
print(json.dumps(current_state, indent=2))

# Validate explicitly
is_valid, message = state_manager.validate_state(current_state)
print(f"State valid: {is_valid}")
if not is_valid:
    print(f"Validation message: {message}")
```

2. Create a valid state:
```python
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
import datetime

# Get state manager instance
state_manager = StateManager()

# Create minimal valid state
valid_state = {
    "timestamp": datetime.datetime.now().isoformat(),
    "system_status": "running",
    "production_line": {
        "status": "running",
        "temperature": 22.5,
        "energy_consumption": 100.0
    }
}

# Set and validate
success = state_manager.set_state(valid_state)
print(f"Set minimal valid state: {success}")
```

### 3.4 Simulation Issues

#### 3.4.1 Simulation Instability

**Symptoms:**
- Extreme or unrealistic simulation results
- NaN or infinity values in simulation
- Simulation crashes

**Causes:**
- Invalid parameter values
- Numerical instability in simulation equations
- Boundary condition violations

**Solutions:**

1. Check simulation parameters:
```python
from circman5.adapters.services.constants_service import ConstantsService

# Get simulation parameters
constants = ConstantsService()
dt_config = constants.get_digital_twin_config()
sim_params = dt_config.get("SIMULATION_PARAMETERS", {})

print("Simulation parameters:")
for key, value in sim_params.items():
    print(f"{key}: {value}")
```

2. Run stabilized simulation:
```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
import numpy as np

# Get digital twin instance
twin = DigitalTwin()

# Set stable parameters
stable_params = {
    "production_line": {
        "temperature": 22.5,  # Set to optimal temperature
        "energy_consumption": 100.0,  # Moderate energy consumption
        "production_rate": 5.0,  # Moderate production rate
        "status": "running"
    }
}

# Run simulation with stability checks
try:
    simulation_results = twin.simulate(steps=10, parameters=stable_params)

    # Check for instability
    for i, state in enumerate(simulation_results):
        if "production_line" in state:
            prod_line = state["production_line"]
            for key, value in prod_line.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    print(f"Instability detected at step {i}, parameter {key}: {value}")

    print(f"Simulation completed with {len(simulation_results)} steps")
except Exception as e:
    print(f"Simulation error: {str(e)}")
```

3. Implement parameter bounds:
```python
def apply_parameter_bounds(parameters):
    """Apply bounds to parameters to ensure stability."""
    bounded_params = parameters.copy()

    # Define bounds for critical parameters
    bounds = {
        "production_line.temperature": (10.0, 40.0),
        "production_line.energy_consumption": (0.0, 1000.0),
        "production_line.production_rate": (0.0, 20.0)
    }

    # Apply bounds
    for param_path, (min_val, max_val) in bounds.items():
        parts = param_path.split('.')

        # Navigate to the parameter
        target = bounded_params
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Apply bounds if parameter exists
        last_part = parts[-1]
        if last_part in target and isinstance(target[last_part], (int, float)):
            target[last_part] = max(min_val, min(target[last_part], max_val))

    return bounded_params
```

#### 3.4.2 Simulation Performance Issues

**Symptoms:**
- Slow simulation execution
- Excessive CPU or memory usage
- System unresponsiveness during simulation

**Causes:**
- Inefficient simulation algorithms
- Excessive state complexity
- Resource leaks

**Solutions:**

1. Profile simulation performance:
```python
import time
import cProfile
import pstats
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

# Get digital twin instance
twin = DigitalTwin()

# Time simulation performance
step_counts = [10, 50, 100]
run_times = []

for steps in step_counts:
    start_time = time.time()
    twin.simulate(steps=steps)
    end_time = time.time()
    run_time = end_time - start_time
    run_times.append(run_time)
    print(f"Simulation with {steps} steps took {run_time:.3f} seconds")

# Calculate steps per second
for steps, run_time in zip(step_counts, run_times):
    steps_per_second = steps / run_time
    print(f"Performance: {steps_per_second:.1f} steps per second for {steps} steps")

# Run detailed profiling for larger simulation
pr = cProfile.Profile()
pr.enable()
twin.simulate(steps=100)
pr.disable()

# Print profiling results
s = pstats.Stats(pr)
s.sort_stats('cumulative')
s.print_stats(20)  # Print top 20 time-consuming functions
```

2. Simplify simulation:
```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

# Get digital twin instance
twin = DigitalTwin()

# Create simplified state (remove unnecessary components)
simplified_params = {
    "production_line": {
        "status": "running",
        "temperature": 22.5,
        "energy_consumption": 100.0,
        "production_rate": 5.0
    }
}

# Run simulation with simplified state
start_time = time.time()
twin.simulate(steps=100, parameters=simplified_params)
end_time = time.time()
run_time = end_time - start_time

print(f"Simplified simulation took {run_time:.3f} seconds")
```

### 3.5 Integration Issues

#### 3.5.1 AI Integration Problems

**Symptoms:**
- Failed optimizations
- Error messages from AI components
- Unexpected optimization results

**Causes:**
- Model training issues
- Parameter extraction failures
- Constraint violations

**Solutions:**

1. Test parameter extraction:
```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration

# Get digital twin and AI integration
twin = DigitalTwin()
ai = AIIntegration(twin)

# Extract parameters
try:
    params = ai.extract_parameters_from_state()
    print("Successfully extracted parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
except Exception as e:
    print(f"Parameter extraction error: {str(e)}")
```

2. Check optimization with simplified constraints:
```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration

# Get digital twin and AI integration
twin = DigitalTwin()
ai = AIIntegration(twin)

# Extract current parameters
current_params = ai.extract_parameters_from_state()

# Set simple constraints
simple_constraints = {
    "energy_used": (10.0, 200.0),  # Wide range
    "cycle_time": (15.0, 60.0)     # Wide range
}

# Run optimization with simple constraints
try:
    optimized_params = ai.optimize_parameters(
        current_params=current_params,
        constraints=simple_constraints
    )

    print("Optimization succeeded:")
    for key, value in optimized_params.items():
        print(f"{key}: {value}")
except Exception as e:
    print(f"Optimization error: {str(e)}")
```

3. Check model training status:
```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration

# Get digital twin and AI integration
twin = DigitalTwin()
ai = AIIntegration(twin)

# Check model training status
model = ai.model
trained = hasattr(model, "is_trained") and model.is_trained

print(f"Model trained: {trained}")
if not trained:
    # Attempt training with synthetic data
    try:
        success = ai.train_model_from_digital_twin(history_limit=10)
        print(f"Training attempt result: {success}")
    except Exception as e:
        print(f"Training error: {str(e)}")
```

#### 3.5.2 LCA Integration Issues

**Symptoms:**
- LCA calculations failing
- Missing environmental impact results
- Error messages from LCA components

**Causes:**
- Missing material or energy data
- Invalid impact factors
- Calculation errors

**Solutions:**

1. Check data extraction:
```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.lca_integration import LCAIntegration

# Get digital twin and LCA integration
twin = DigitalTwin()
lca = LCAIntegration(twin)

# Extract material data
try:
    material_data = lca.extract_material_data_from_state()
    print(f"Material data rows: {len(material_data)}")
    if not material_data.empty:
        print(f"Material columns: {list(material_data.columns)}")
        print(material_data.head())
    else:
        print("Material data is empty")
except Exception as e:
    print(f"Material data extraction error: {str(e)}")

# Extract energy data
try:
    energy_data = lca.extract_energy_data_from_state()
    print(f"Energy data rows: {len(energy_data)}")
    if not energy_data.empty:
        print(f"Energy columns: {list(energy_data.columns)}")
        print(energy_data.head())
    else:
        print("Energy data is empty")
except Exception as e:
    print(f"Energy data extraction error: {str(e)}")
```

2. Check impact factor configuration:
```python
from circman5.adapters.services.constants_service import ConstantsService

# Get impact factors
constants = ConstantsService()
try:
    impact_factors = constants.get_impact_factors()
    print("Impact factor keys:", list(impact_factors.keys()))

    # Check specific factors
    if "MATERIAL_IMPACT_FACTORS" in impact_factors:
        material_factors = impact_factors["MATERIAL_IMPACT_FACTORS"]
        print("Material impact factors:", list(material_factors.keys()))
    else:
        print("Missing MATERIAL_IMPACT_FACTORS")

    if "GRID_CARBON_INTENSITIES" in impact_factors:
        grid_intensities = impact_factors["GRID_CARBON_INTENSITIES"]
        print("Grid intensities:", grid_intensities)
    else:
        print("Missing GRID_CARBON_INTENSITIES")
except Exception as e:
    print(f"Error accessing impact factors: {str(e)}")
```

### 3.6 Event System Issues

#### 3.6.1 Event Subscription Problems

**Symptoms:**
- Events not being received by subscribers
- Missing notifications for system changes
- No events appearing in logs

**Causes:**
- Incorrect event subscription
- Event filtering issues
- Event publishing errors

**Solutions:**

1. Test event system:
```python
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.digital_twin.event_notification.event_types import Event, EventCategory, EventSeverity

# Flag to check if event received
event_received = False

# Define a test handler
def test_handler(event):
    global event_received
    print(f"Test handler received event: {event.message}")
    event_received = True

# Subscribe to test events
event_manager.subscribe(test_handler, category=EventCategory.SYSTEM)

# Create and publish test event
test_event = Event(
    category=EventCategory.SYSTEM,
    severity=EventSeverity.INFO,
    source="troubleshooting",
    message="Test event for diagnostics"
)

event_manager.publish(test_event)

# Check if event was received
print(f"Event received by handler: {event_received}")

# Check event persistence
events = event_manager.get_events(category=EventCategory.SYSTEM, limit=1)
if events:
    print(f"Most recent system event: {events[0].message}")
else:
    print("No system events found in persistence")
```

2. Check event publisher implementation:
```python
from circman5.manufacturing.digital_twin.event_notification.publishers import DigitalTwinPublisher
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.digital_twin.event_notification.event_types import EventCategory, EventSeverity

# Flag to check if event received
event_received = False

# Define a test handler
def test_handler(event):
    global event_received
    print(f"Test handler received event: {event.message}")
    event_received = True

# Subscribe to threshold events
event_manager.subscribe(test_handler, category=EventCategory.THRESHOLD)

# Create publisher
publisher = DigitalTwinPublisher()

# Publish threshold event
publisher.publish_parameter_threshold_event(
    parameter_path="production_line.temperature",
    parameter_name="Temperature",
    threshold=25.0,
    actual_value=26.5,
    state={"timestamp": "2025-02-24T14:30:22.123456"},
    severity=EventSeverity.WARNING
)

# Check if event was received
print(f"Event received by handler: {event_received}")
```

3. Monitor all events:
```python
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.digital_twin.event_notification.event_types import EventCategory

# Create a handler for all events
def monitor_all_events(event):
    print(f"[{event.severity.value}] [{event.category.value}] {event.message}")

# Subscribe to all categories
for category in EventCategory:
    event_manager.subscribe(monitor_all_events, category=category)

print("Event monitoring active for all categories")
```

## 4. Advanced Troubleshooting

### 4.1 Monitoring Digital Twin State

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
import time
import json

# Get digital twin instance
twin = DigitalTwin()

# Monitor state changes
previous_state = None
monitoring_interval = 1.0  # seconds
monitoring_duration = 30.0  # seconds

start_time = time.time()
print(f"Starting state monitoring for {monitoring_duration} seconds...")

while time.time() - start_time < monitoring_duration:
    current_state = twin.get_current_state()

    if previous_state is not None:
        # Compare with previous state
        changes = {}

        # Check top-level changes
        for key in set(current_state.keys()) | set(previous_state.keys()):
            if key not in previous_state:
                changes[key] = f"Added: {current_state[key]}"
            elif key not in current_state:
                changes[key] = "Removed"
            elif current_state[key] != previous_state[key]:
                if isinstance(current_state[key], dict) and isinstance(previous_state[key], dict):
                    # For nested dictionaries, check individual fields
                    nested_changes = {}
                    for nested_key in set(current_state[key].keys()) | set(previous_state[key].keys()):
                        if nested_key not in previous_state[key]:
                            nested_changes[nested_key] = f"Added: {current_state[key][nested_key]}"
                        elif nested_key not in current_state[key]:
                            nested_changes[nested_key] = "Removed"
                        elif current_state[key][nested_key] != previous_state[key][nested_key]:
                            nested_changes[nested_key] = f"{previous_state[key][nested_key]} -> {current_state[key][nested_key]}"
                    if nested_changes:
                        changes[key] = nested_changes
                else:
                    changes[key] = f"{previous_state[key]} -> {current_state[key]}"

        # Print changes if any
        if changes:
            print("\nState changes detected:")
            print(json.dumps(changes, indent=2))

    previous_state = current_state
    time.sleep(monitoring_interval)

print("State monitoring completed")
```

### 4.2 System Recovery

#### 4.2.1 State Recovery

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
import datetime

# Get instances
twin = DigitalTwin()
state_manager = StateManager()

# Create recovery state
recovery_state = {
    "timestamp": datetime.datetime.now().isoformat(),
    "system_status": "recovered",
    "production_line": {
        "status": "idle",
        "temperature": 22.0,
        "energy_consumption": 0.0,
        "production_rate": 0.0
    },
    "materials": {
        "silicon_wafer": {"inventory": 1000, "quality": 0.95},
        "solar_glass": {"inventory": 500, "quality": 0.98}
    },
    "environment": {"temperature": 22.0, "humidity": 45.0}
}

# Apply recovery state
state_manager.set_state(recovery_state)
print("Applied recovery state")

# Verify recovery
try:
    current_state = twin.get_current_state()
    print(f"System status: {current_state.get('system_status')}")
    print(f"Production line status: {current_state.get('production_line', {}).get('status')}")
except Exception as e:
    print(f"Verification error: {str(e)}")
```

#### 4.2.2 Component Reinitialization

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager

# Reset digital twin
print("Resetting Digital Twin...")
if hasattr(DigitalTwin, "_reset_instance"):
    DigitalTwin._reset_instance()
elif hasattr(DigitalTwin, "_instance"):
    DigitalTwin._instance = None

# Reset state manager
print("Resetting State Manager...")
if hasattr(StateManager, "_reset_instance"):
    StateManager._reset_instance()
elif hasattr(StateManager, "_instance"):
    StateManager._instance = None

# Create fresh instances
twin = DigitalTwin()
state_manager = StateManager()

# Initialize components
print("Initializing components...")
twin_init = twin.initialize()
print(f"Digital Twin initialization: {twin_init}")

# Check event manager
events = event_manager.get_events(limit=1)
print(f"Event system accessible: {events is not None}")

print("Component reinitialization completed")
```

#### 4.2.3 Full System Reset

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter
from circman5.adapters.services.constants_service import ConstantsService

# Reset all singleton components
print("Performing full system reset...")

# Reset digital twin and state manager
if hasattr(DigitalTwin, "_reset_instance"): DigitalTwin._reset_instance()
elif hasattr(DigitalTwin, "_instance"): DigitalTwin._instance = None

if hasattr(StateManager, "_reset_instance"): StateManager._reset_instance()
elif hasattr(StateManager, "_instance"): StateManager._instance = None

# Reset constants service
if hasattr(ConstantsService, "_reset_instance"): ConstantsService._reset_instance()
elif hasattr(ConstantsService, "_instance"): ConstantsService._instance = None

# Create new instances
twin = DigitalTwin()
state_manager = StateManager()
constants = ConstantsService()

# Initialize digital twin
twin_init = twin.initialize()
print(f"Digital Twin initialization: {twin_init}")

# Initialize interface if available
if interface_manager:
    interface_init = interface_manager.initialize()
    print(f"Interface initialization: {interface_init}")

print("Full system reset completed")
```

### 4.3 Configuration Recovery

```python
from circman5.adapters.config.digital_twin import DigitalTwinAdapter
from circman5.adapters.config.impact_factors import ImpactFactorsAdapter
from circman5.adapters.config.manufacturing import ManufacturingAdapter
from circman5.adapters.config.optimization import OptimizationAdapter
from circman5.adapters.config.visualization import VisualizationAdapter
from circman5.adapters.config.monitoring import MonitoringAdapter
import json

# List of adapters to reset
adapters = [
    DigitalTwinAdapter(),
    ImpactFactorsAdapter(),
    ManufacturingAdapter(),
    OptimizationAdapter(),
    VisualizationAdapter(),
    MonitoringAdapter()
]

print("Resetting all configuration files to defaults...")

# Reset each adapter to defaults
for adapter in adapters:
    try:
        # Get default configuration
        defaults = adapter.get_defaults()

        # Write defaults to file
        with open(adapter.config_path, 'w') as f:
            json.dump(defaults, f, indent=2)

        print(f"Reset {adapter.__class__.__name__} configuration")
    except Exception as e:
        print(f"Error resetting {adapter.__class__.__name__}: {str(e)}")

print("Configuration reset completed")
```

### 4.4 Performance Profiling

```python
import cProfile
import pstats
import io
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

# Function to profile
def run_dt_operations():
    twin = DigitalTwin()
    twin.initialize()

    # Run several operations
    for _ in range(10):
        twin.update()

    # Run simulation
    twin.simulate(steps=50)

    # Get history
    twin.get_state_history(100)

# Run profiler
pr = cProfile.Profile()
pr.enable()

run_dt_operations()

pr.disable()

# Print sorted stats
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(20)
print(s.getvalue())
```

## 5. Common Error Messages

### 5.1 Initialization Errors

#### Error: "Digital Twin not initialized"
**Cause**: `initialize()` not called before using digital twin
**Solution**: Call `twin.initialize()` after creating the DigitalTwin instance

#### Error: "StateManager not initialized"
**Cause**: StateManager initialization failure
**Solution**: Check configuration and reset StateManager if necessary

### 5.2 Configuration Errors

#### Error: "Configuration file not found: path/to/config.json"
**Cause**: Missing configuration file
**Solution**: Create default configuration file or check paths

#### Error: "Invalid configuration: Missing required key 'DIGITAL_TWIN_CONFIG'"
**Cause**: Configuration file missing required sections
**Solution**: Reset to default configuration or add missing sections

### 5.3 Simulation Errors

#### Error: "Simulation parameter out of bounds: temperature = 105.0"
**Cause**: Parameter values outside valid ranges
**Solution**: Apply parameter bounds before simulation

#### Error: "Numerical instability in simulation: division by zero"
**Cause**: Calculation error due to invalid parameters
**Solution**: Check division operations and add guards against zero values

## 6. Diagnostic Scripts

### 6.1 Digital Twin Health Check

```python
def digital_twin_health_check():
    """Run a comprehensive health check on the Digital Twin system."""
    print("=== DIGITAL TWIN HEALTH CHECK ===")

    # Check digital twin initialization
    from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
    twin = DigitalTwin()

    print("\n1. Digital Twin Status:")
    initialized = hasattr(twin, "_initialized") and twin._initialized
    is_running = hasattr(twin, "is_running") and twin.is_running
    print(f"  Initialized: {initialized}")
    print(f"  Running: {is_running}")

    # Check configuration
    from circman5.adapters.services.constants_service import ConstantsService
    constants = ConstantsService()

    print("\n2. Configuration Status:")
    try:
        dt_config = constants.get_digital_twin_config()
        print(f"  Configuration loaded: Yes")

        # Check essential config sections
        essential_sections = ["DIGITAL_TWIN_CONFIG", "SIMULATION_PARAMETERS"]
        for section in essential_sections:
            if section in dt_config:
                print(f"  '{section}' section: Present")
            else:
                print(f"  '{section}' section: Missing")
    except Exception as e:
        print(f"  Configuration error: {str(e)}")

    # Check state manager
    from circman5.manufacturing.digital_twin.core.state_manager import StateManager
    state_manager = StateManager()

    print("\n3. State Manager Status:")
    state_manager_init = hasattr(state_manager, "_initialized") and state_manager._initialized
    print(f"  Initialized: {state_manager_init}")

    current_state = state_manager.get_current_state() if state_manager_init else None
    if current_state:
        print(f"  Current state available: Yes")
        print(f"  System status: {current_state.get('system_status', 'Unknown')}")
    else:
        print(f"  Current state available: No")

    history = state_manager.get_history() if state_manager_init else []
    print(f"  History states: {len(history)}")

    # Check event system
    from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager

    print("\n4. Event System Status:")
    try:
        recent_events = event_manager.get_events(limit=1)
        print(f"  Event system accessible: Yes")
        print(f"  Recent events available: {len(recent_events) > 0}")
    except Exception as e:
        print(f"  Event system error: {str(e)}")

    # Check simulation engine
    print("\n5. Simulation Engine:")
    if is_running:
        try:
            sim_results = twin.simulate(steps=1)
            print(f"  Simulation engine functional: Yes")
            print(f"  Simulation returned {len(sim_results)} states")
        except Exception as e:
            print(f"  Simulation error: {str(e)}")
    else:
        print(f"  Simulation engine status: Not available (Twin not running)")

    # Final assessment
    print("\n6. Overall Assessment:")
    if initialized and is_running and state_manager_init and current_state:
        print("  HEALTHY: Digital Twin system appears to be functioning correctly")
    elif initialized and state_manager_init but not is_running:
        print("  ISSUE: Digital Twin initialized but not running")
    elif not initialized:
        print("  CRITICAL: Digital Twin not initialized")
    else:
        print("  WARNING: Digital Twin has partial functionality")

    print("\n=== HEALTH CHECK COMPLETE ===")
```

### 6.2 Event System Diagnostics

```python
def event_system_diagnostics():
    """Run diagnostics on the event notification system."""
    print("=== EVENT SYSTEM DIAGNOSTICS ===")

    from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager
    from circman5.manufacturing.digital_twin.event_notification.event_types import Event, EventCategory, EventSeverity

    print("\n1. Event Manager Status:")

    # Check initialization
    initialized = hasattr(event_manager, "_initialized") and event_manager._initialized
    print(f"  Initialized: {initialized}")

    # Check event retrieval
    try:
        events = event_manager.get_events(limit=5)
        print(f"  Event retrieval: Working")
        print(f"  Recent events: {len(events)}")
    except Exception as e:
        print(f"  Event retrieval error: {str(e)}")

    print("\n2. Event Publishing Test:")

    # Set up test receiver
    event_received = False

    def test_receiver(event):
        nonlocal event_received
        event_received = True
        print(f"  Received event: {event.message}")

    # Subscribe to test events
    try:
        event_manager.subscribe(test_receiver, category=EventCategory.SYSTEM)
        print("  Subscription: Success")
    except Exception as e:
        print(f"  Subscription error: {str(e)}")

    # Publish test event
    try:
        test_event = Event(
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            source="diagnostics",
            message="Test event for diagnostics"
        )
        event_manager.publish(test_event)
        print("  Event publishing: Success")
    except Exception as e:
        print(f"  Event publishing error: {str(e)}")

    # Check if event was received
    print(f"  Event received by handler: {event_received}")

    # Check event persistence
    try:
        persisted_events = event_manager.get_events(category=EventCategory.SYSTEM, limit=1)
        if persisted_events and persisted_events[0].message == "Test event for diagnostics":
            print("  Event persistence: Working")
        else:
            print("  Event persistence: Not working or event not stored")
    except Exception as e:
        print(f"  Event persistence error: {str(e)}")

    # Check event categories
    print("\n3. Event Category Analysis:")
    category_counts = {}

    try:
        all_events = event_manager.get_events(limit=100)
        for event in all_events:
            category = event.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

        print("  Event categories found:")
        for category, count in category_counts.items():
            print(f"    {category}: {count} events")
    except Exception as e:
        print(f"  Category analysis error: {str(e)}")

    # Final assessment
    print("\n4. Overall Assessment:")
    if initialized and event_received:
        print("  HEALTHY: Event system appears to be functioning correctly")
    elif initialized and not event_received:
        print("  ISSUE: Event system initialized but event handling not working")
    elif not initialized:
        print("  CRITICAL: Event system not initialized")
    else:
        print("  WARNING: Event system has partial functionality")

    print("\n=== EVENT SYSTEM DIAGNOSTICS COMPLETE ===")
```

## 7. Appendices

### 7.1 Configuration File Locations

**Default configuration paths:**
- Digital Twin: `src/circman5/adapters/config/json/digital_twin.json`
- Impact Factors: `src/circman5/adapters/config/json/impact_factors.json`
- Manufacturing: `src/circman5/adapters/config/json/manufacturing.json`
- Monitoring: `src/circman5/adapters/config/json/monitoring.json`
- Optimization: `src/circman5/adapters/config/json/optimization.json`
- Visualization: `src/circman5/adapters/config/json/visualization.json`

### 7.2 Log File Locations

**Default log locations:**
- Development: `logs/digital_twin.log`
- Production: `/var/log/circman5/digital_twin.log` (Linux) or `C:\ProgramData\CIRCMAN5\logs\digital_twin.log` (Windows)

### 7.3 Support Resources

- Project Documentation: `docs/`
- API Reference: `docs/api/dt_api_reference.md`
- Mathematical Foundations: `docs/mathematical/`
- Implementation Guide: `docs/implementation/dt_implementation_guide.md`
