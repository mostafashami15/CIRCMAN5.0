# Digital Twin API Reference

## 1. Introduction

This API reference document provides comprehensive documentation for the CIRCMAN5.0 Digital Twin system API. It covers the core interfaces, classes, and methods available for interacting with the Digital Twin component of the CIRCMAN5.0 system.

## 2. Digital Twin Core API

### 2.1 DigitalTwin Class

The central coordinator for the digital twin system.

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
```

#### 2.1.1 Constructor

```python
DigitalTwin(config=None)
```

**Parameters:**
- `config` (DigitalTwinConfig, optional): Configuration settings for the digital twin.

**Notes:**
- Implements singleton pattern - only one instance exists in the application.
- If called multiple times, returns the existing instance.

#### 2.1.2 Methods

##### initialize()

```python
initialize() -> bool
```

Initializes the digital twin with initial state and connections.

**Returns:**
- `bool`: True if initialization was successful, False otherwise.

##### update(external_data=None)

```python
update(external_data=None) -> bool
```

Updates the digital twin state with new data from the physical system.

**Parameters:**
- `external_data` (Dict[str, Any], optional): Optional data from external sources to update the state.

**Returns:**
- `bool`: True if update was successful, False otherwise.

##### simulate(steps=None, parameters=None)

```python
simulate(steps=None, parameters=None) -> List[Dict[str, Any]]
```

Runs a simulation based on the current state.

**Parameters:**
- `steps` (int, optional): Number of simulation steps to run. Default from config if None.
- `parameters` (Dict[str, Any], optional): Optional parameters to modify for the simulation.

**Returns:**
- `List[Dict[str, Any]]`: List of simulated states.

##### get_current_state()

```python
get_current_state() -> Dict[str, Any]
```

Gets the current state of the digital twin.

**Returns:**
- `Dict[str, Any]`: Current state.

##### get_state_history(limit=None)

```python
get_state_history(limit=None) -> List[Dict[str, Any]]
```

Gets historical states of the digital twin.

**Parameters:**
- `limit` (int, optional): Optional limit on the number of historical states to retrieve.

**Returns:**
- `List[Dict[str, Any]]`: List of historical states.

##### save_state(file_path=None)

```python
save_state(file_path=None) -> bool
```

Saves the current state to a file.

**Parameters:**
- `file_path` (Union[str, Path], optional): Optional path to save the state. Uses results_manager if None.

**Returns:**
- `bool`: True if save was successful, False otherwise.

##### load_state(file_path)

```python
load_state(file_path) -> bool
```

Loads a state from a file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to load the state from.

**Returns:**
- `bool`: True if load was successful, False otherwise.

### 2.2 DigitalTwinConfig Class

Configuration class for the Digital Twin system.

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwinConfig
```

#### 2.2.1 Constructor

```python
DigitalTwinConfig(
    name="SoliTek_DigitalTwin",
    update_frequency=1.0,
    history_length=1000,
    simulation_steps=10,
    data_sources=["sensors", "manual_input", "manufacturing_system"],
    synchronization_mode="real_time",
    log_level="INFO"
)
```

**Parameters:**
- `name` (str): Name identifier for the digital twin.
- `update_frequency` (float): Frequency of state updates in Hz.
- `history_length` (int): Maximum number of historical states to keep.
- `simulation_steps` (int): Default number of steps for simulation.
- `data_sources` (List[str]): List of data sources to use for updates.
- `synchronization_mode` (str): Mode for synchronization ("real_time", "batch", "manual").
- `log_level` (str): Logging level for the digital twin.

#### 2.2.2 Methods

##### from_constants()

```python
@classmethod
from_constants() -> DigitalTwinConfig
```

Creates configuration from constants service.

**Returns:**
- `DigitalTwinConfig`: Config instance with values from constants service.

## 3. State Management API

### 3.1 StateManager Class

Manages the state of the digital twin system.

```python
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
```

#### 3.1.1 Constructor

```python
StateManager(history_length=None)
```

**Parameters:**
- `history_length` (int, optional): Maximum number of historical states to keep.

**Notes:**
- Implements singleton pattern - only one instance exists in the application.
- If called multiple times, returns the existing instance.

#### 3.1.2 Methods

##### set_state(state)

```python
set_state(state: Dict[str, Any]) -> None
```

Sets the current state and adds to history.

**Parameters:**
- `state` (Dict[str, Any]): New state to set.

##### update_state(updates)

```python
update_state(updates: Dict[str, Any]) -> None
```

Updates parts of the current state.

**Parameters:**
- `updates` (Dict[str, Any]): Dictionary with updates to apply to the current state.

##### get_current_state()

```python
get_current_state() -> Dict[str, Any]
```

Gets the current state.

**Returns:**
- `Dict[str, Any]`: Copy of the current state.

##### get_history(limit=None)

```python
get_history(limit=None) -> List[Dict[str, Any]]
```

Gets historical states.

**Parameters:**
- `limit` (int, optional): Optional limit on the number of historical states to retrieve.

**Returns:**
- `List[Dict[str, Any]]`: List of historical states.

##### get_state_at_time(timestamp)

```python
get_state_at_time(timestamp: str) -> Optional[Dict[str, Any]]
```

Gets the state at a specific time.

**Parameters:**
- `timestamp` (str): ISO format timestamp to look for.

**Returns:**
- `Optional[Dict[str, Any]]`: State at the specified time, or None if not found.

##### validate_state(state)

```python
validate_state(state: Any) -> Tuple[bool, str]
```

Validates a state dictionary.

**Parameters:**
- `state` (Any): State dictionary to validate.

**Returns:**
- `Tuple[bool, str]`: (is_valid, message) where message explains any validation issues.

##### export_state(file_path=None)

```python
export_state(file_path=None) -> bool
```

Exports the current state to a JSON file.

**Parameters:**
- `file_path` (Union[str, Path], optional): Optional path to save the state. Uses results_manager if None.

**Returns:**
- `bool`: True if export was successful, False otherwise.

## 4. Simulation API

### 4.1 SimulationEngine Class

Core simulation engine for the digital twin system.

```python
from circman5.manufacturing.digital_twin.simulation.simulation_engine import SimulationEngine
```

#### 4.1.1 Constructor

```python
SimulationEngine(state_manager: StateManager)
```

**Parameters:**
- `state_manager` (StateManager): StateManager instance to access system state.

#### 4.1.2 Methods

##### run_simulation(steps=10, initial_state=None, parameters=None)

```python
run_simulation(
    steps: int = 10,
    initial_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

Runs a simulation for the specified number of steps.

**Parameters:**
- `steps` (int): Number of simulation steps to run.
- `initial_state` (Dict[str, Any], optional): Optional custom initial state (uses current state if None).
- `parameters` (Dict[str, Any], optional): Optional parameter modifications for the simulation.

**Returns:**
- `List[Dict[str, Any]]`: List of simulated states.

### 4.2 ScenarioManager Class

Manages simulation scenarios.

```python
from circman5.manufacturing.digital_twin.simulation.scenario_manager import ScenarioManager
```

#### 4.2.1 Constructor

```python
ScenarioManager()
```

**Notes:**
- Implements singleton pattern - only one instance exists in the application.

#### 4.2.2 Methods

##### create_scenario(name, parameters, description="")

```python
create_scenario(
    name: str,
    parameters: Dict[str, Any],
    description: str = ""
) -> Optional[Scenario]
```

Creates and saves a new scenario.

**Parameters:**
- `name` (str): Unique name for the scenario.
- `parameters` (Dict[str, Any]): Scenario parameters.
- `description` (str, optional): Optional scenario description.

**Returns:**
- `Optional[Scenario]`: Created scenario object, or None if creation failed.

##### get_scenario(name)

```python
get_scenario(name: str) -> Optional[Scenario]
```

Gets a scenario by name.

**Parameters:**
- `name` (str): Name of the scenario to retrieve.

**Returns:**
- `Optional[Scenario]`: Scenario object, or None if not found.

##### list_scenarios()

```python
list_scenarios() -> List[str]
```

Lists all available scenarios.

**Returns:**
- `List[str]`: List of scenario names.

##### compare_scenarios(scenario_names)

```python
compare_scenarios(scenario_names: List[str]) -> Dict[str, Dict[str, float]]
```

Compares multiple scenarios.

**Parameters:**
- `scenario_names` (List[str]): List of scenario names to compare.

**Returns:**
- `Dict[str, Dict[str, float]]`: Comparison results.

## 5. Usage Examples

### 5.1 Basic Digital Twin Usage

```python
# Initialize the Digital Twin
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

# Get Digital Twin instance
twin = DigitalTwin()

# Initialize
twin.initialize()

# Get current state
current_state = twin.get_current_state()
print(f"Current system status: {current_state.get('system_status')}")

# Update with external data
external_data = {
    "production_line": {
        "temperature": 23.5,
        "energy_consumption": 120.5
    }
}
twin.update(external_data)

# Run simulation
parameters = {
    "production_line.temperature": 24.0,
    "production_line.status": "running"
}
simulation_results = twin.simulate(steps=20, parameters=parameters)

# Save current state
twin.save_state("production_state_backup.json")
```

### 5.2 Scenario Management

```python
# Create and run scenarios
from circman5.manufacturing.digital_twin.simulation.scenario_manager import ScenarioManager

# Get scenario manager
scenario_manager = ScenarioManager()

# Create a scenario
scenario_params = {
    "production_line.temperature": 22.5,
    "production_line.energy_consumption": 100.0,
    "production_line.production_rate": 8.5
}
scenario_manager.create_scenario(
    name="high_efficiency",
    parameters=scenario_params,
    description="Optimized for high efficiency production"
)

# Get a scenario
scenario = scenario_manager.get_scenario("high_efficiency")

# List all scenarios
all_scenarios = scenario_manager.list_scenarios()
print(f"Available scenarios: {all_scenarios}")

# Compare scenarios
comparison = scenario_manager.compare_scenarios(["high_efficiency", "energy_saving"])
```

## 6. Error Handling

All API methods follow consistent error handling patterns:

1. Public methods catch exceptions internally
2. Most methods return boolean success indicators where appropriate
3. Errors are logged through the logging system
4. Error events are published through the event system for critical errors

Example of proper error handling:

```python
try:
    # Get Digital Twin instance
    twin = DigitalTwin()

    # Initialize
    success = twin.initialize()
    if not success:
        print("Initialization failed, check logs for details")
        return

    # Run simulation with parameters
    simulation_results = twin.simulate(steps=20, parameters=parameters)

    # Process results
    if not simulation_results:
        print("Simulation failed or returned empty results")
        return

    # Print final state
    final_state = simulation_results[-1]
    print(f"Final system status: {final_state.get('system_status')}")

except Exception as e:
    print(f"Error in Digital Twin operation: {str(e)}")
```

## 7. Thread Safety

All singleton classes implement proper thread safety:

1. Initialization protected by locks
2. State updates synchronized
3. Component registration protected

Example of thread-safe usage:

```python
# Thread-safe state update
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
import threading

def update_process(temperature):
    state_manager = StateManager()
    updates = {
        "production_line": {
            "temperature": temperature
        }
    }
    state_manager.update_state(updates)

# Create multiple threads for concurrent updates
threads = []
for temp in range(20, 30):
    thread = threading.Thread(target=update_process, args=(temp,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()
```

## 8. Configuration

The Digital Twin API uses a configuration system for customizing behavior:

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwinConfig
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

# Create custom configuration
config = DigitalTwinConfig(
    name="Custom_DigitalTwin",
    update_frequency=2.0,
    history_length=500,
    simulation_steps=50,
    synchronization_mode="batch"
)

# Initialize with custom configuration
twin = DigitalTwin(config=config)
twin.initialize()
```

Alternatively, use configuration from constants service:

```python
# Get configuration from constants service
config = DigitalTwinConfig.from_constants()

# Initialize with configuration from constants service
twin = DigitalTwin(config=config)
twin.initialize()
```
