# Digital Twin Human Interface API

## 1. Introduction

This API reference document provides comprehensive documentation for the Human-Machine Interface (HMI) system of the CIRCMAN5.0 Digital Twin. It covers interfaces, classes, and methods that enable user interaction with the digital twin system, including visualization, control, and notification capabilities.

## 2. Interface Manager

The Interface Manager serves as the central coordinator for all HMI components.

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
```

### 2.1 Component Management

#### 2.1.1 register_component()

```python
def register_component(component_id: str, component: Any) -> None
```

Registers an interface component.

**Parameters:**
- `component_id` (str): Unique identifier for the component.
- `component` (Any): Component instance.

**Raises:**
- `ValueError`: If component ID already exists.

**Example:**
```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.manufacturing.human_interface.components.dashboard.custom_panel import CustomPanel

# Create custom panel
custom_panel = CustomPanel()

# Register with interface manager
interface_manager.register_component("custom_panel", custom_panel)
```

#### 2.1.2 get_component()

```python
def get_component(component_id: str) -> Any
```

Gets a registered component by ID.

**Parameters:**
- `component_id` (str): Component identifier.

**Returns:**
- `Any`: Component instance.

**Raises:**
- `KeyError`: If component not found.

**Example:**
```python
# Get a registered component
try:
    process_panel = interface_manager.get_component("process_panel")
    process_panel.refresh()
except KeyError:
    print("Process panel not found")
```

### 2.2 Event Management

#### 2.2.1 register_event_handler()

```python
def register_event_handler(event_type: str, handler: Callable) -> None
```

Registers a handler for interface events.

**Parameters:**
- `event_type` (str): Type of event to handle.
- `handler` (Callable): Handler function.

**Example:**
```python
# Define event handler
def handle_view_change(event_data):
    print(f"View changed to: {event_data.get('view_name')}")

# Register event handler
interface_manager.register_event_handler("view_changed", handle_view_change)
```

#### 2.2.2 trigger_event()

```python
def trigger_event(event_type: str, event_data: Dict[str, Any]) -> None
```

Triggers an interface event.

**Parameters:**
- `event_type` (str): Type of event to trigger.
- `event_data` (Dict[str, Any]): Event data.

**Example:**
```python
# Trigger custom event
interface_manager.trigger_event(
    "parameter_updated",
    {
        "parameter": "temperature",
        "value": 23.5,
        "timestamp": "2025-02-24T14:30:22"
    }
)
```

#### 2.2.3 handle_parameter_selection()

```python
def handle_parameter_selection(parameter: str, selected: bool) -> None
```

Handles parameter selection/deselection.

**Parameters:**
- `parameter` (str): Parameter name.
- `selected` (bool): Selection state.

**Example:**
```python
# Select a parameter for monitoring/visualization
interface_manager.handle_parameter_selection("temperature", True)

# Deselect a parameter
interface_manager.handle_parameter_selection("humidity", False)
```

### 2.3 View Management

#### 2.3.1 change_view()

```python
def change_view(view_name: str) -> None
```

Changes the active interface view.

**Parameters:**
- `view_name` (str): Name of view to activate.

**Example:**
```python
# Change to process view
interface_manager.change_view("process_view")

# Change to dashboard view
interface_manager.change_view("main_dashboard")
```

#### 2.3.2 toggle_panel()

```python
def toggle_panel(panel_id: str) -> bool
```

Toggles a panel's expanded state.

**Parameters:**
- `panel_id` (str): ID of panel to toggle.

**Returns:**
- `bool`: New expanded state.

**Example:**
```python
# Toggle process panel
expanded = interface_manager.toggle_panel("process_panel")
print(f"Process panel expanded: {expanded}")
```

#### 2.3.3 update_alert_settings()

```python
def update_alert_settings(filters: Dict[str, Any]) -> None
```

Updates alert display settings.

**Parameters:**
- `filters` (Dict[str, Any]): Alert filter settings.

**Example:**
```python
# Update alert settings
interface_manager.update_alert_settings({
    "severity": ["warning", "error"],
    "categories": ["threshold", "system"],
    "acknowledged": False
})
```

#### 2.3.4 save_custom_view()

```python
def save_custom_view(name: str, config: Dict[str, Any]) -> bool
```

Saves a custom dashboard view configuration.

**Parameters:**
- `name` (str): Name for the custom view.
- `config` (Dict[str, Any]): View configuration.

**Returns:**
- `bool`: Success status.

**Example:**
```python
# Save custom view
interface_manager.save_custom_view(
    "production_monitoring",
    {
        "panels": ["process_panel", "kpi_panel", "alert_panel"],
        "layout": "grid",
        "refresh_rate": 5
    }
)
```

### 2.4 Command Handling

#### 2.4.1 handle_command()

```python
def handle_command(command: str, params: Dict[str, Any]) -> Dict[str, Any]
```

Handles a command from the interface.

**Parameters:**
- `command` (str): Command name.
- `params` (Dict[str, Any]): Command parameters.

**Returns:**
- `Dict[str, Any]`: Command result.

**Example:**
```python
# Execute a command
result = interface_manager.handle_command(
    "update_parameter",
    {"parameter": "temperature", "value": 23.5}
)

# Check result
if result.get("success"):
    print("Command executed successfully")
else:
    print(f"Command failed: {result.get('error')}")
```

### 2.5 Initialization and Shutdown

#### 2.5.1 initialize()

```python
def initialize() -> bool
```

Initializes all interface components.

**Returns:**
- `bool`: True if initialization successful.

**Example:**
```python
# Initialize interface
success = interface_manager.initialize()
if success:
    print("Interface initialized successfully")
else:
    print("Interface initialization failed")
```

#### 2.5.2 shutdown()

```python
def shutdown() -> None
```

Cleans up resources and shuts down interface.

**Example:**
```python
# Shutdown interface
interface_manager.shutdown()
print("Interface shutdown complete")
```

## 3. Digital Twin Adapter

The Digital Twin Adapter provides a standardized interface for HMI components to interact with the Digital Twin.

```python
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter
```

### 3.1 State Management

#### 3.1.1 get_current_state()

```python
def get_current_state() -> Dict[str, Any]
```

Gets the current state of the digital twin.

**Returns:**
- `Dict[str, Any]`: Current state.

**Example:**
```python
# Get current state
state = digital_twin_adapter.get_current_state()

# Access state properties
system_status = state.get("system_status")
temperature = state.get("production_line", {}).get("temperature")

print(f"System status: {system_status}")
print(f"Temperature: {temperature}°C")
```

#### 3.1.2 get_state_history()

```python
def get_state_history(limit: Optional[int] = None) -> List[Dict[str, Any]]
```

Gets historical states of the digital twin.

**Parameters:**
- `limit` (int, optional): Optional limit on the number of historical states to retrieve.

**Returns:**
- `List[Dict[str, Any]]`: List of historical states.

**Example:**
```python
# Get last 10 states
history = digital_twin_adapter.get_state_history(limit=10)

# Process historical data
for state in history:
    timestamp = state.get("timestamp")
    temperature = state.get("production_line", {}).get("temperature")
    print(f"{timestamp}: {temperature}°C")
```

#### 3.1.3 update_state()

```python
def update_state(updates: Dict[str, Any]) -> bool
```

Updates the digital twin state.

**Parameters:**
- `updates` (Dict[str, Any]): Updates to apply.

**Returns:**
- `bool`: True if update successful.

**Example:**
```python
# Update production line parameters
success = digital_twin_adapter.update_state({
    "production_line": {
        "temperature": 23.5,
        "status": "running"
    }
})

if success:
    print("State updated successfully")
else:
    print("State update failed")
```

#### 3.1.4 save_state()

```python
def save_state(filename: Optional[str] = None) -> bool
```

Saves the current state to a file.

**Parameters:**
- `filename` (str, optional): Optional filename to save the state.

**Returns:**
- `bool`: True if save successful.

**Example:**
```python
# Save current state
success = digital_twin_adapter.save_state("production_state_backup.json")

if success:
    print("State saved successfully")
else:
    print("Failed to save state")
```

#### 3.1.5 load_state()

```python
def load_state(filename: str) -> bool
```

Loads a state from a file.

**Parameters:**
- `filename` (str): Filename to load the state from.

**Returns:**
- `bool`: True if load successful.

**Example:**
```python
# Load saved state
success = digital_twin_adapter.load_state("production_state_backup.json")

if success:
    print("State loaded successfully")
else:
    print("Failed to load state")
```

### 3.2 Simulation

#### 3.2.1 run_simulation()

```python
def run_simulation(
    steps: int = 10,
    parameters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

Runs a simulation using the digital twin.

**Parameters:**
- `steps` (int): Number of simulation steps to run.
- `parameters` (Dict[str, Any], optional): Optional parameter modifications for the simulation.

**Returns:**
- `List[Dict[str, Any]]`: List of simulated states.

**Example:**
```python
# Define parameter modifications
parameters = {
    "production_line": {
        "temperature": 24.0,
        "production_rate": 8.5
    }
}

# Run simulation for 20 steps
simulation_results = digital_twin_adapter.run_simulation(steps=20, parameters=parameters)

# Process simulation results
final_state = simulation_results[-1]
final_temperature = final_state.get("production_line", {}).get("temperature")
print(f"Final temperature: {final_temperature}°C")
```

### 3.3 Scenario Management

#### 3.3.1 save_scenario()

```python
def save_scenario(
    name: str,
    parameters: Dict[str, Any],
    description: str = ""
) -> bool
```

Saves a simulation scenario.

**Parameters:**
- `name` (str): Name for the scenario.
- `parameters` (Dict[str, Any]): Scenario parameters.
- `description` (str): Optional scenario description.

**Returns:**
- `bool`: True if scenario was saved.

**Example:**
```python
# Save a production scenario
success = digital_twin_adapter.save_scenario(
    name="high_efficiency",
    parameters={
        "production_line.temperature": 22.5,
        "production_line.production_rate": 9.0
    },
    description="Optimized for high efficiency production"
)

if success:
    print("Scenario saved successfully")
else:
    print("Failed to save scenario")
```

#### 3.3.2 run_scenario()

```python
def run_scenario(scenario_name: str) -> List[Dict[str, Any]]
```

Runs a saved simulation scenario.

**Parameters:**
- `scenario_name` (str): Name of the scenario to run.

**Returns:**
- `List[Dict[str, Any]]`: List of simulated states.

**Example:**
```python
# Run a saved scenario
results = digital_twin_adapter.run_scenario("high_efficiency")

# Process scenario results
if results:
    print(f"Scenario simulation completed with {len(results)} states")
    final_state = results[-1]
    efficiency = final_state.get("production_line", {}).get("efficiency")
    print(f"Final efficiency: {efficiency:.2f}")
else:
    print("Scenario simulation failed")
```

#### 3.3.3 get_all_scenarios()

```python
def get_all_scenarios() -> Dict[str, Dict[str, Any]]
```

Gets all saved scenarios.

**Returns:**
- `Dict[str, Dict[str, Any]]`: Dictionary of scenario data.

**Example:**
```python
# Get all saved scenarios
scenarios = digital_twin_adapter.get_all_scenarios()

# Display scenario information
print(f"Available scenarios: {len(scenarios)}")
for name, scenario in scenarios.items():
    description = scenario.get("description", "No description")
    params = scenario.get("parameters", {})
    print(f"- {name}: {description}")
    print(f"  Parameters: {len(params)} parameters defined")
```

#### 3.3.4 compare_scenarios()

```python
def compare_scenarios(scenario_names: List[str]) -> Dict[str, Dict[str, float]]
```

Compares multiple scenarios.

**Parameters:**
- `scenario_names` (List[str]): List of scenario names to compare.

**Returns:**
- `Dict[str, Dict[str, float]]`: Comparison results.

**Example:**
```python
# Compare two scenarios
comparison = digital_twin_adapter.compare_scenarios(["high_efficiency", "energy_saving"])

# Process comparison results
if comparison:
    print("Scenario comparison results:")
    for metric, values in comparison.items():
        print(f"Metric: {metric}")
        for scenario, value in values.items():
            print(f"  {scenario}: {value}")
else:
    print("Scenario comparison failed")
```

## 4. Dashboard Components

### 4.1 Dashboard Manager

```python
from circman5.manufacturing.human_interface.core.dashboard_manager import DashboardManager
```

#### 4.1.1 Constructor

```python
DashboardManager()
```

**Note:** Typically accessed through the interface manager rather than created directly.

#### 4.1.2 Methods

##### create_dashboard()

```python
def create_dashboard(dashboard_id: str, config: Dict[str, Any]) -> bool
```

Creates a new dashboard with the specified configuration.

**Parameters:**
- `dashboard_id` (str): Unique dashboard identifier.
- `config` (Dict[str, Any]): Dashboard configuration.

**Returns:**
- `bool`: Success status.

##### activate_dashboard()

```python
def activate_dashboard(dashboard_id: str) -> bool
```

Activates a specific dashboard.

**Parameters:**
- `dashboard_id` (str): Dashboard to activate.

**Returns:**
- `bool`: Success status.

##### update_dashboard()

```python
def update_dashboard(dashboard_id: str, config: Dict[str, Any]) -> bool
```

Updates an existing dashboard's configuration.

**Parameters:**
- `dashboard_id` (str): Dashboard to update.
- `config` (Dict[str, Any]): New dashboard configuration.

**Returns:**
- `bool`: Success status.

### 4.2 Base Panel

```python
from circman5.manufacturing.human_interface.components.dashboard.base_panel import BasePanel
```

#### 4.2.1 Constructor

```python
BasePanel(panel_id: str)
```

**Parameters:**
- `panel_id` (str): Unique panel identifier.

#### 4.2.2 Methods

##### initialize()

```python
def initialize() -> None
```

Initializes the panel.

##### update()

```python
def update(data: Dict[str, Any]) -> None
```

Updates the panel with new data.

**Parameters:**
- `data` (Dict[str, Any]): Update data.

##### handle_command()

```python
def handle_command(command: str, params: Dict[str, Any]) -> Dict[str, Any]
```

Handles a command sent to this panel.

**Parameters:**
- `command` (str): Command name.
- `params` (Dict[str, Any]): Command parameters.

**Returns:**
- `Dict[str, Any]`: Command result.

### 4.3 Specialized Panels

#### 4.3.1 Process Panel

```python
from circman5.manufacturing.human_interface.components.dashboard.process_panel import ProcessPanel
```

**Purpose:** Visualizes the manufacturing process in real-time.

**Key Methods:**
- `refresh()`: Refreshes the process visualization.
- `set_visualization_mode(mode)`: Sets visualization mode.
- `highlight_component(component_id)`: Highlights a specific component.

#### 4.3.2 KPI Panel

```python
from circman5.manufacturing.human_interface.components.dashboard.kpi_panel import KPIPanel
```

**Purpose:** Displays key performance indicators.

**Key Methods:**
- `add_kpi(kpi_id, config)`: Adds a KPI to the panel.
- `remove_kpi(kpi_id)`: Removes a KPI from the panel.
- `update_kpi_thresholds(thresholds)`: Updates KPI thresholds.

#### 4.3.3 Status Panel

```python
from circman5.manufacturing.human_interface.components.dashboard.status_panel import StatusPanel
```

**Purpose:** Shows system status information.

**Key Methods:**
- `set_status_display_mode(mode)`: Sets status display mode.
- `filter_status_items(filters)`: Filters status items.
- `acknowledge_status_item(item_id)`: Acknowledges a status item.

## 5. Control Components

### 5.1 Parameter Control

```python
from circman5.manufacturing.human_interface.components.controls.parameter_control import ParameterControl
```

#### 5.1.1 Constructor

```python
ParameterControl(control_id: str)
```

**Parameters:**
- `control_id` (str): Unique control identifier.

#### 5.1.2 Methods

##### set_parameter_value()

```python
def set_parameter_value(
    parameter: str,
    value: Union[float, str, bool]
) -> bool
```

Sets a parameter value.

**Parameters:**
- `parameter` (str): Parameter identifier.
- `value` (Union[float, str, bool]): Parameter value.

**Returns:**
- `bool`: Success status.

##### get_parameter_value()

```python
def get_parameter_value(parameter: str) -> Union[float, str, bool, None]
```

Gets a parameter value.

**Parameters:**
- `parameter` (str): Parameter identifier.

**Returns:**
- `Union[float, str, bool, None]`: Parameter value, or None if not found.

##### set_parameter_range()

```python
def set_parameter_range(
    parameter: str,
    min_value: float,
    max_value: float
) -> bool
```

Sets a parameter's valid range.

**Parameters:**
- `parameter` (str): Parameter identifier.
- `min_value` (float): Minimum value.
- `max_value` (float): Maximum value.

**Returns:**
- `bool`: Success status.

### 5.2 Process Control

```python
from circman5.manufacturing.human_interface.components.controls.process_control import ProcessControl
```

#### 5.2.1 Constructor

```python
ProcessControl(control_id: str)
```

**Parameters:**
- `control_id` (str): Unique control identifier.

#### 5.2.2 Methods

##### start_process()

```python
def start_process() -> bool
```

Starts the manufacturing process.

**Returns:**
- `bool`: Success status.

##### stop_process()

```python
def stop_process() -> bool
```

Stops the manufacturing process.

**Returns:**
- `bool`: Success status.

##### set_process_mode()

```python
def set_process_mode(mode: str) -> bool
```

Sets the process operation mode.

**Parameters:**
- `mode` (str): Operation mode ("normal", "maintenance", "high_efficiency", etc.).

**Returns:**
- `bool`: Success status.

### 5.3 Scenario Control

```python
from circman5.manufacturing.human_interface.components.controls.scenario_control import ScenarioControl
```

#### 5.3.1 Constructor

```python
ScenarioControl(control_id: str)
```

**Parameters:**
- `control_id` (str): Unique control identifier.

#### 5.3.2 Methods

##### create_scenario()

```python
def create_scenario(
    name: str,
    parameters: Dict[str, Any],
    description: str = ""
) -> bool
```

Creates a new scenario.

**Parameters:**
- `name` (str): Scenario name.
- `parameters` (Dict[str, Any]): Scenario parameters.
- `description` (str): Optional scenario description.

**Returns:**
- `bool`: Success status.

##### run_scenario()

```python
def run_scenario(scenario_name: str) -> Dict[str, Any]
```

Runs a scenario.

**Parameters:**
- `scenario_name` (str): Name of scenario to run.

**Returns:**
- `Dict[str, Any]`: Scenario run results.

##### load_scenario_parameters()

```python
def load_scenario_parameters(scenario_name: str) -> Dict[str, Any]
```

Loads scenario parameters without running.

**Parameters:**
- `scenario_name` (str): Name of scenario.

**Returns:**
- `Dict[str, Any]`: Scenario parameters.

## 6. Alert Components

### 6.1 Alert Panel

```python
from circman5.manufacturing.human_interface.components.alerts.alert_panel import AlertPanel
```

#### 6.1.1 Constructor

```python
AlertPanel(panel_id: str)
```

**Parameters:**
- `panel_id` (str): Unique panel identifier.

#### 6.1.2 Methods

##### refresh_alerts()

```python
def refresh_alerts() -> None
```

Refreshes the alerts display.

##### filter_alerts()

```python
def filter_alerts(filters: Dict[str, Any]) -> None
```

Filters displayed alerts.

**Parameters:**
- `filters` (Dict[str, Any]): Alert filters.

##### acknowledge_alert()

```python
def acknowledge_alert(alert_id: str) -> bool
```

Acknowledges an alert.

**Parameters:**
- `alert_id` (str): ID of alert to acknowledge.

**Returns:**
- `bool`: Success status.

### 6.2 Notification Manager

```python
from circman5.manufacturing.human_interface.components.alerts.notification_manager import NotificationManager
```

#### 6.2.1 Constructor

```python
NotificationManager()
```

#### 6.2.2 Methods

##### add_notification()

```python
def add_notification(
    message: str,
    severity: str,
    details: Optional[Dict[str, Any]] = None
) -> str
```

Adds a notification to the display queue.

**Parameters:**
- `message` (str): Notification message.
- `severity` (str): Severity level ("info", "warning", "error", "critical").
- `details` (Dict[str, Any], optional): Additional details.

**Returns:**
- `str`: Notification ID.

##### remove_notification()

```python
def remove_notification(notification_id: str) -> bool
```

Removes a notification from the display.

**Parameters:**
- `notification_id` (str): ID of notification to remove.

**Returns:**
- `bool`: Success status.

##### set_notification_expiration()

```python
def set_notification_expiration(
    notification_id: str,
    expiration_seconds: int
) -> bool
```

Sets a notification's expiration time.

**Parameters:**
- `notification_id` (str): Notification ID.
- `expiration_seconds` (int): Seconds until expiration.

**Returns:**
- `bool`: Success status.

### 6.3 Event Subscriber

```python
from circman5.manufacturing.human_interface.components.alerts.event_subscriber import EventSubscriber
```

#### 6.3.1 Constructor

```python
EventSubscriber()
```

#### 6.3.2 Methods

##### subscribe_to_events()

```python
def subscribe_to_events(
    categories: Optional[List[str]] = None,
    severities: Optional[List[str]] = None
) -> None
```

Subscribes to digital twin events.

**Parameters:**
- `categories` (List[str], optional): Event categories to subscribe to.
- `severities` (List[str], optional): Event severities to subscribe to.

##### unsubscribe_from_events()

```python
def unsubscribe_from_events() -> None
```

Unsubscribes from all events.

##### set_event_handler()

```python
def set_event_handler(handler: Callable[[Dict[str, Any]], None]) -> None
```

Sets the event handler function.

**Parameters:**
- `handler` (Callable): Function to handle events.

## 7. Usage Examples

### 7.1 Basic Interface Initialization

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.manufacturing.human_interface.components.dashboard.process_panel import ProcessPanel
from circman5.manufacturing.human_interface.components.dashboard.kpi_panel import KPIPanel
from circman5.manufacturing.human_interface.components.alerts.alert_panel import AlertPanel

# Create interface components
process_panel = ProcessPanel("process_panel")
kpi_panel = KPIPanel("kpi_panel")
alert_panel = AlertPanel("alert_panel")

# Register components with interface manager
interface_manager.register_component("process_panel", process_panel)
interface_manager.register_component("kpi_panel", kpi_panel)
interface_manager.register_component("alert_panel", alert_panel)

# Initialize interface
interface_manager.initialize()

# Change to the main dashboard view
interface_manager.change_view("main_dashboard")
```

### 7.2 Parameter Control

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.manufacturing.human_interface.components.controls.parameter_control import ParameterControl

# Create parameter control
param_control = ParameterControl("temperature_control")

# Register with interface manager
interface_manager.register_component("temperature_control", param_control)

# Set parameter ranges
param_control.set_parameter_range("temperature", 15.0, 30.0)
param_control.set_parameter_range("production_rate", 1.0, 10.0)

# Set parameter values
param_control.set_parameter_value("temperature", 23.5)
param_control.set_parameter_value("production_rate", 8.0)

# Read parameters
current_temp = param_control.get_parameter_value("temperature")
print(f"Current temperature setting: {current_temp}°C")
```

### 7.3 Scenario Management

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.manufacturing.human_interface.components.controls.scenario_control import ScenarioControl

# Create scenario control
scenario_control = ScenarioControl("scenario_control")

# Register with interface manager
interface_manager.register_component("scenario_control", scenario_control)

# Create a new scenario
scenario_control.create_scenario(
    name="energy_efficiency",
    parameters={
        "production_line.temperature": 22.0,
        "production_line.cycle_time": 35.0,
        "production_line.pressure": 5.0
    },
    description="Optimized for energy efficiency"
)

# Run the scenario
results = scenario_control.run_scenario("energy_efficiency")

# Process results
if results.get("success"):
    print("Scenario run successfully")
    energy_saved = results.get("energy_saved", 0)
    print(f"Energy saved: {energy_saved:.2f} kWh")
else:
    print(f"Scenario run failed: {results.get('error')}")
```

### 7.4 Alert Handling

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.manufacturing.human_interface.components.alerts.notification_manager import NotificationManager
from circman5.manufacturing.human_interface.components.alerts.event_subscriber import EventSubscriber

# Create alert components
notification_manager = NotificationManager()
event_subscriber = EventSubscriber()

# Register with interface manager
interface_manager.register_component("notification_manager", notification_manager)
interface_manager.register_component("event_subscriber", event_subscriber)

# Set up event handler
def handle_event(event_data):
    severity = event_data.get("severity", "info")
    message = event_data.get("message", "Unknown event")

    # Create user notification for events
    notification_manager.add_notification(
        message=message,
        severity=severity,
        details=event_data.get("details")
    )

    print(f"Event received: [{severity}] {message}")

# Set event handler and subscribe
event_subscriber.set_event_handler(handle_event)
event_subscriber.subscribe_to_events(
    categories=["threshold", "system"],
    severities=["warning", "error", "critical"]
)

# Add a manual notification
notification_id = notification_manager.add_notification(
    message="System maintenance scheduled",
    severity="info",
    details={"time": "2025-03-01T08:00:00", "duration": "2 hours"}
)

# Set expiration time
notification_manager.set_notification_expiration(notification_id, 3600)  # 1 hour
```

### 7.5 Digital Twin Interaction

```python
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter

# Get current state
state = digital_twin_adapter.get_current_state()
print(f"Current system status: {state.get('system_status')}")

# Get production line temperature
temperature = state.get("production_line", {}).get("temperature")
print(f"Current temperature: {temperature}°C")

# Update temperature setting
update_success = digital_twin_adapter.update_state({
    "production_line": {
        "temperature": temperature + 1.0
    }
})

if update_success:
    print("Temperature updated successfully")

    # Run a simulation to see effects
    simulation_results = digital_twin_adapter.run_simulation(steps=10)

    # Check final state
    final_state = simulation_results[-1]
    final_temp = final_state.get("production_line", {}).get("temperature")
    energy = final_state.get("production_line", {}).get("energy_consumption")

    print(f"Simulated temperature after 10 steps: {final_temp}°C")
    print(f"Simulated energy consumption: {energy} kWh")
else:
    print("Failed to update temperature")
```

### 7.6 Custom Dashboard Creation

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.manufacturing.human_interface.core.dashboard_manager import DashboardManager

# Get dashboard manager
dashboard_manager = interface_manager.get_component("dashboard_manager")

# Create custom dashboard
dashboard_config = {
    "title": "Production Monitoring Dashboard",
    "layout": "grid",
    "panels": [
        {
            "id": "process_panel",
            "position": {"x": 0, "y": 0, "width": 8, "height": 4},
            "config": {"visualization_mode": "detailed"}
        },
        {
            "id": "kpi_panel",
            "position": {"x": 8, "y": 0, "width": 4, "height": 4},
            "config": {"kpis": ["temperature", "energy", "production_rate"]}
        },
        {
            "id": "alert_panel",
            "position": {"x": 0, "y": 4, "width": 12, "height": 2},
            "config": {"severity_filter": ["warning", "error", "critical"]}
        }
    ],
    "refresh_rate": 5
}

# Create and activate dashboard
dashboard_manager.create_dashboard("production_dashboard", dashboard_config)
dashboard_manager.activate_dashboard("production_dashboard")

# Switch to the dashboard
interface_manager.change_view("production_dashboard")
```

## 8. Error Handling

The HMI API implements comprehensive error handling:

1. Components catch and handle exceptions internally
2. Commands return structured error responses
3. Event handlers are isolated to prevent cascading failures
4. UI updates are managed to prevent inconsistent state

Example error handling pattern:

```python
# Example of proper error handling
def handle_command(self, command, params):
    try:
        # Process command
        if command == "update_parameter":
            parameter = params.get("parameter")
            value = params.get("value")

            if not parameter:
                return {"success": False, "error": "Missing parameter name"}

            if value is None:
                return {"success": False, "error": "Missing parameter value"}

            # Execute the command
            success = self.set_parameter_value(parameter, value)

            return {"success": success}
        else:
            return {"success": False, "error": "Unknown command"}

    except ValueError as e:
        # Handle validation errors
        self.logger.warning(f"Validation error in command {command}: {str(e)}")
        return {"success": False, "error": str(e)}

    except Exception as e:
        # Handle unexpected errors
        self.logger.error(f"Error in command {command}: {str(e)}")
        return {"success": False, "error": "Internal error"}
```

## 9. Thread Safety

The HMI API is designed with thread safety in mind:

1. Interface manager operations are thread-safe
2. Component access is synchronized
3. Event handling runs in a controlled manner
4. State updates are coordinated to prevent race conditions

Example thread-safe implementation:

```python
# Example of thread-safe implementation
def trigger_event(self, event_type, event_data):
    """Thread-safe event triggering."""
    with self._lock:
        # Get handlers for this event type
        handlers = self.event_handlers.get(event_type, [])

    # Execute handlers outside the lock to prevent deadlocks
    for handler in handlers:
        try:
            handler(event_data)
        except Exception as e:
            self.logger.error(f"Error in event handler: {str(e)}")
```

## A10. Configuration Options

The HMI system is highly configurable through constants service:

```json
{
    "HUMAN_INTERFACE": {
        "REFRESH_RATE": 5,
        "DEFAULT_VIEW": "main_dashboard",
        "PANEL_DEFAULTS": {
            "process_panel": {
                "visualization_mode": "standard",
                "auto_refresh": true
            },
            "kpi_panel": {
                "display_mode": "gauge",
                "warning_threshold": 80,
                "critical_threshold": 90
            },
            "alert_panel": {
                "max_alerts": 50,
                "default_severity_filter": ["warning", "error", "critical"],
                "auto_refresh": true
            }
        },
        "CUSTOMIZATION": {
            "allow_custom_dashboards": true,
            "allow_custom_panels": true,
            "allow_layout_changes": true
        }
    }
}
```

## 11. Extending the HMI System

The HMI system is designed to be extensible:

### 11.1 Creating Custom Panels

```python
from circman5.manufacturing.human_interface.components.dashboard.base_panel import BasePanel

class CustomProductionPanel(BasePanel):
    """Custom panel for specialized production visualization."""

    def __init__(self, panel_id):
        super().__init__(panel_id)
        self.title = "Production Analysis"
        self.data = {}

    def initialize(self):
        """Initialize panel components."""
        # Set up panel structure
        self.components = {
            "efficiency_chart": {"type": "line_chart", "data": []},
            "quality_meter": {"type": "gauge", "value": 0},
            "alerts_summary": {"type": "alert_list", "items": []}
        }

    def update(self, data):
        """Update panel with new data."""
        self.data = data

        # Update components with new data
        if "efficiency" in data:
            self.components["efficiency_chart"]["data"].append(data["efficiency"])
            # Keep last 20 points
            self.components["efficiency_chart"]["data"] = self.components["efficiency_chart"]["data"][-20:]

        if "quality" in data:
            self.components["quality_meter"]["value"] = data["quality"]

        if "alerts" in data:
            self.components["alerts_summary"]["items"] = data["alerts"]

    def handle_command(self, command, params):
        """Handle panel-specific commands."""
        if command == "reset_chart":
            self.components["efficiency_chart"]["data"] = []
            return {"success": True}

        elif command == "change_view_mode":
            mode = params.get("mode")
            if mode in ["simple", "detailed", "analytical"]:
                self.view_mode = mode
                return {"success": True}
            else:
                return {"success": False, "error": "Invalid view mode"}

        return {"success": False, "handled": False}
```

### 11.2 Creating Custom Controls

```python
from circman5.manufacturing.human_interface.components.controls.base_control import BaseControl

class BatchProcessControl(BaseControl):
    """Custom control for batch processing operations."""

    def __init__(self, control_id):
        super().__init__(control_id)
        self.batches = {}

    def initialize(self):
        """Initialize control components."""
        # Set up digital twin adapter
        from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter
        self.dt_adapter = digital_twin_adapter

    def create_batch(self, batch_id, parameters):
        """Create a new production batch."""
        if batch_id in self.batches:
            return False

        self.batches[batch_id] = {
            "parameters": parameters,
            "status": "created",
            "start_time": None,
            "end_time": None
        }

        return True

    def start_batch(self, batch_id):
        """Start a production batch."""
        if batch_id not in self.batches:
            return False

        if self.batches[batch_id]["status"] != "created":
            return False

        # Update batch status
        self.batches[batch_id]["status"] = "running"
        self.batches[batch_id]["start_time"] = datetime.datetime.now().isoformat()

        # Apply batch parameters to digital twin
        batch_params = self.batches[batch_id]["parameters"]
        success = self.dt_adapter.update_state(batch_params)

        return success

    def complete_batch(self, batch_id):
        """Complete a production batch."""
        if batch_id not in self.batches:
            return False

        if self.batches[batch_id]["status"] != "running":
            return False

        # Update batch status
        self.batches[batch_id]["status"] = "completed"
        self.batches[batch_id]["end_time"] = datetime.datetime.now().isoformat()

        return True

    def get_batch_status(self, batch_id):
        """Get status of a batch."""
        if batch_id not in self.batches:
            return None

        return self.batches[batch_id]
```

### 11.3 Customizing Event Handling

```python
from circman5.manufacturing.human_interface.components.alerts.event_subscriber import EventSubscriber
import datetime

class AdvancedEventProcessor(EventSubscriber):
    """Advanced event processing with filtering and analytics."""

    def __init__(self):
        super().__init__()
        self.event_stats = {
            "total_events": 0,
            "by_category": {},
            "by_severity": {},
            "by_hour": {}
        }
        self.event_history = []
        self.max_history = 1000

    def process_event(self, event_data):
        """Process and analyze events."""
        # Update statistics
        self.event_stats["total_events"] += 1

        # Update category stats
        category = event_data.get("category", "unknown")
        self.event_stats["by_category"][category] = self.event_stats["by_category"].get(category, 0) + 1

        # Update severity stats
        severity = event_data.get("severity", "unknown")
        self.event_stats["by_severity"][severity] = self.event_stats["by_severity"].get(severity, 0) + 1

        # Update hourly stats
        timestamp = event_data.get("timestamp", datetime.datetime.now().isoformat())
        hour = timestamp.split("T")[1].split(":")[0]
        self.event_stats["by_hour"][hour] = self.event_stats["by_hour"].get(hour, 0) + 1

        # Add to history
        self.event_history.append(event_data)

        # Trim history if needed
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]

        # Call default handler if set
        if self.event_handler:
            self.event_handler(event_data)

    def get_event_statistics(self):
        """Get event statistics."""
        return self.event_stats

    def query_events(self, filters=None):
        """Query event history with filters."""
        if not filters:
            return self.event_history

        filtered_events = []
        for event in self.event_history:
            matches = True

            for key, value in filters.items():
                if key in event:
                    if isinstance(value, list):
                        if event[key] not in value:
                            matches = False
                            break
                    elif event[key] != value:
                        matches = False
                        break
                else:
                    matches = False
                    break

            if matches:
                filtered_events.append(event)

        return filtered_events
```

## 12. Integration with Web Frameworks

The HMI system can be integrated with web frameworks:

```python
# Example of Flask integration
from flask import Flask, jsonify, request
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager

app = Flask(__name__)

@app.route('/api/state', methods=['GET'])
def get_state():
    """API endpoint to get current state."""
    state = digital_twin_adapter.get_current_state()
    return jsonify(state)

@app.route('/api/state/history', methods=['GET'])
def get_history():
    """API endpoint to get state history."""
    limit = request.args.get('limit', default=10, type=int)
    history = digital_twin_adapter.get_state_history(limit=limit)
    return jsonify(history)

@app.route('/api/state', methods=['POST'])
def update_state():
    """API endpoint to update state."""
    updates = request.json
    success = digital_twin_adapter.update_state(updates)
    return jsonify({"success": success})

@app.route('/api/command', methods=['POST'])
def execute_command():
    """API endpoint to execute commands."""
    command_data = request.json
    command = command_data.get('command')
    params = command_data.get('params', {})

    result = interface_manager.handle_command(command, params)
    return jsonify(result)

@app.route('/api/simulation', methods=['POST'])
def run_simulation():
    """API endpoint to run simulation."""
    data = request.json
    steps = data.get('steps', 10)
    parameters = data.get('parameters')

    results = digital_twin_adapter.run_simulation(steps=steps, parameters=parameters)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
```
