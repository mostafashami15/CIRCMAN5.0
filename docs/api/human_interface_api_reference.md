# Human Interface API Reference

## Overview

This document provides a comprehensive API reference for the Human-Machine Interface (HMI) components of CIRCMAN5.0. These components enable operators and managers to interact with the Digital Twin system for monitoring and controlling PV manufacturing processes through an intuitive, real-time interface.

## Table of Contents

1. [Core Services](#core-services)
   - [InterfaceManager](#interfacemanager)
   - [InterfaceState](#interfacestate)
   - [DashboardManager](#dashboardmanager)
2. [Dashboard Components](#dashboard-components)
   - [MainDashboard](#maindashboard)
3. [Control Components](#control-components)
   - [ParameterControl](#parametercontrol)
4. [Alert Components](#alert-components)
   - [AlertPanel](#alertpanel)
5. [Adapters](#adapters)
   - [DigitalTwinAdapter](#digitaltwinadapter)
6. [Services](#services)
   - [CommandService](#commandservice)
7. [Integration Examples](#integration-examples)

## Core Services

### InterfaceManager

The `InterfaceManager` is the central coordinator for the HMI system. It manages component registration, event handling, and command routing.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the interface manager singleton instance
- **Parameters**: None

#### Methods

```python
def register_component(self, component_id: str, component: Any) -> None
```
- **Description**: Registers an interface component
- **Parameters**:
  - `component_id`: Unique identifier for the component
  - `component`: Component instance
- **Raises**:
  - `ValueError`: If component ID already exists

```python
def get_component(self, component_id: str) -> Any
```
- **Description**: Gets a registered component by ID
- **Parameters**:
  - `component_id`: Component identifier
- **Returns**: Component instance
- **Raises**:
  - `KeyError`: If component not found

```python
def register_event_handler(self, event_type: str, handler: Callable) -> None
```
- **Description**: Registers a handler for interface events
- **Parameters**:
  - `event_type`: Type of event to handle
  - `handler`: Handler function

```python
def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None
```
- **Description**: Triggers an interface event
- **Parameters**:
  - `event_type`: Type of event to trigger
  - `event_data`: Event data

```python
def change_view(self, view_name: str) -> None
```
- **Description**: Changes the active interface view
- **Parameters**:
  - `view_name`: Name of view to activate

```python
def handle_parameter_selection(self, parameter: str, selected: bool) -> None
```
- **Description**: Handles parameter selection/deselection
- **Parameters**:
  - `parameter`: Parameter name
  - `selected`: Whether parameter is selected

```python
def toggle_panel(self, panel_id: str) -> bool
```
- **Description**: Toggles a panel's expanded state
- **Parameters**:
  - `panel_id`: Panel identifier
- **Returns**: New expanded state

```python
def update_alert_settings(self, filters: Dict[str, Any]) -> None
```
- **Description**: Updates alert display settings
- **Parameters**:
  - `filters`: Alert filter settings

```python
def save_custom_view(self, name: str, config: Dict[str, Any]) -> None
```
- **Description**: Saves a custom dashboard view configuration
- **Parameters**:
  - `name`: View name
  - `config`: View configuration

```python
def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]
```
- **Description**: Handles a command from the interface
- **Parameters**:
  - `command`: Command name
  - `params`: Command parameters
- **Returns**: Command result

```python
def initialize(self) -> bool
```
- **Description**: Initializes all interface components
- **Returns**: True if initialization successful

```python
def shutdown(self) -> None
```
- **Description**: Cleans up resources and shuts down interface
- **Parameters**: None

### InterfaceState

The `InterfaceState` manages the state of the HMI, including active views, selected parameters, and panel configurations.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the interface state singleton instance
- **Parameters**: None

#### Methods

```python
def set_active_view(self, view_name: str) -> None
```
- **Description**: Sets the active view in the interface
- **Parameters**:
  - `view_name`: Name of the view to set as active

```python
def get_active_view(self) -> str
```
- **Description**: Gets the currently active view
- **Returns**: Name of the active view

```python
def add_selected_parameter(self, parameter: str) -> None
```
- **Description**: Adds a parameter to the selected parameters set
- **Parameters**:
  - `parameter`: Parameter to select

```python
def remove_selected_parameter(self, parameter: str) -> None
```
- **Description**: Removes a parameter from the selected parameters set
- **Parameters**:
  - `parameter`: Parameter to deselect

```python
def get_selected_parameters(self) -> Set[str]
```
- **Description**: Gets the set of currently selected parameters
- **Returns**: Set of selected parameter names

```python
def toggle_panel_expanded(self, panel_id: str) -> bool
```
- **Description**: Toggles the expanded state of a panel
- **Parameters**:
  - `panel_id`: ID of the panel to toggle
- **Returns**: New expanded state (True if expanded)

```python
def is_panel_expanded(self, panel_id: str) -> bool
```
- **Description**: Checks if a panel is expanded
- **Parameters**:
  - `panel_id`: ID of the panel to check
- **Returns**: True if panel is expanded

```python
def update_alert_filters(self, filters: Dict[str, Any]) -> None
```
- **Description**: Updates alert filtering criteria
- **Parameters**:
  - `filters`: Dictionary of filter criteria

```python
def get_alert_filters(self) -> Dict[str, Any]
```
- **Description**: Gets current alert filters
- **Returns**: Current alert filters

```python
def save_custom_view(self, view_name: str, view_config: Dict[str, Any]) -> None
```
- **Description**: Saves a custom view configuration
- **Parameters**:
  - `view_name`: Name of the custom view
  - `view_config`: View configuration

```python
def delete_custom_view(self, view_name: str) -> bool
```
- **Description**: Deletes a custom view
- **Parameters**:
  - `view_name`: Name of the custom view to delete
- **Returns**: True if view was deleted

```python
def get_custom_view(self, view_name: str) -> Optional[Dict[str, Any]]
```
- **Description**: Gets a custom view configuration
- **Parameters**:
  - `view_name`: Name of the custom view
- **Returns**: View configuration if exists

```python
def get_all_custom_views(self) -> Dict[str, Dict[str, Any]]
```
- **Description**: Gets all custom views
- **Returns**: All custom views

```python
def set_parameter_group(self, group_name: str) -> None
```
- **Description**: Sets the selected parameter group
- **Parameters**:
  - `group_name`: Name of the parameter group

```python
def get_parameter_group(self) -> str
```
- **Description**: Gets the selected parameter group
- **Returns**: Name of the selected parameter group

```python
def set_parameter_edit_mode(self, edit_mode: bool) -> None
```
- **Description**: Sets parameter edit mode
- **Parameters**:
  - `edit_mode`: Whether edit mode is enabled

```python
def is_parameter_edit_mode(self) -> bool
```
- **Description**: Checks if parameter edit mode is enabled
- **Returns**: True if edit mode is enabled

```python
def set_selected_process(self, process_name: str) -> None
```
- **Description**: Sets the selected process
- **Parameters**:
  - `process_name`: Name of the process

```python
def get_selected_process(self) -> str
```
- **Description**: Gets the selected process
- **Returns**: Name of the selected process

```python
def set_process_control_mode(self, mode: str) -> None
```
- **Description**: Sets the process control mode
- **Parameters**:
  - `mode`: Control mode (monitor, manual, automatic)
- **Raises**:
  - `ValueError`: If invalid mode

```python
def get_process_control_mode(self) -> str
```
- **Description**: Gets the process control mode
- **Returns**: Current process control mode

```python
def reset_to_defaults(self) -> None
```
- **Description**: Resets interface state to default values
- **Parameters**: None

### DashboardManager

The `DashboardManager` handles dashboard layouts and rendering for the HMI system.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the dashboard manager
- **Parameters**: None

#### Dashboard Layout

```python
class DashboardLayout:
    def __init__(
        self,
        name: str,
        description: str = "",
        panels: Optional[Dict[str, Any]] = None,
        layout_config: Optional[Dict[str, Any]] = None,
    )
```
- **Description**: Represents a dashboard layout configuration
- **Parameters**:
  - `name`: Layout name
  - `description`: Layout description
  - `panels`: Optional dictionary of panel configurations
  - `layout_config`: Optional grid layout configuration

```python
def to_dict(self) -> Dict[str, Any]
```
- **Description**: Converts layout to dictionary
- **Returns**: Dictionary representation

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "DashboardLayout"
```
- **Description**: Creates layout from dictionary
- **Parameters**:
  - `data`: Dictionary representation
- **Returns**: Created layout

#### DashboardManager Methods

```python
def register_component(self, component_id: str, component: Any) -> None
```
- **Description**: Registers a dashboard component
- **Parameters**:
  - `component_id`: Component identifier
  - `component`: Component instance
- **Raises**:
  - `ValueError`: If component ID already exists

```python
def create_layout(
    self,
    name: str,
    description: str = "",
    panels: Optional[Dict[str, Any]] = None,
    layout_config: Optional[Dict[str, Any]] = None,
) -> DashboardLayout
```
- **Description**: Creates a new dashboard layout
- **Parameters**:
  - `name`: Layout name
  - `description`: Layout description
  - `panels`: Optional panel configurations
  - `layout_config`: Optional layout configuration
- **Returns**: Created layout
- **Raises**:
  - `ValueError`: If layout name already exists

```python
def get_layout(self, name: str) -> Optional[DashboardLayout]
```
- **Description**: Gets a dashboard layout by name
- **Parameters**:
  - `name`: Layout name
- **Returns**: Layout if found, None otherwise

```python
def delete_layout(self, name: str) -> bool
```
- **Description**: Deletes a dashboard layout
- **Parameters**:
  - `name`: Layout name
- **Returns**: True if layout was deleted

```python
def set_active_layout(self, name: str) -> bool
```
- **Description**: Sets the active dashboard layout
- **Parameters**:
  - `name`: Layout name
- **Returns**: True if layout was activated

```python
def get_active_layout(self) -> Optional[DashboardLayout]
```
- **Description**: Gets the currently active layout
- **Returns**: Current layout or None

```python
def update_layout(self, layout: DashboardLayout) -> None
```
- **Description**: Updates a dashboard layout
- **Parameters**:
  - `layout`: Layout to update

```python
def get_all_layouts(self) -> Dict[str, DashboardLayout]
```
- **Description**: Gets all available layouts
- **Returns**: Dictionary of layouts

```python
def render_dashboard(self, layout_name: Optional[str] = None) -> Dict[str, Any]
```
- **Description**: Renders a dashboard layout
- **Parameters**:
  - `layout_name`: Optional layout name (uses active layout if None)
- **Returns**: Rendered dashboard data

```python
def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]
```
- **Description**: Handles dashboard-related commands
- **Parameters**:
  - `command`: Command name
  - `params`: Command parameters
- **Returns**: Command result

## Dashboard Components

### MainDashboard

The `MainDashboard` implements the primary dashboard layout and functionality.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the main dashboard
- **Parameters**: None

#### Methods

```python
def render_dashboard(self) -> Dict[str, Any]
```
- **Description**: Renders the main dashboard
- **Returns**: Dashboard data

```python
def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]
```
- **Description**: Renders a dashboard panel
- **Parameters**:
  - `config`: Panel configuration
- **Returns**: Panel data

```python
def update_layout(self, layout_config: Dict[str, Any]) -> bool
```
- **Description**: Updates dashboard layout configuration
- **Parameters**:
  - `layout_config`: New layout configuration
- **Returns**: True if layout was updated

```python
def update_panel_config(self, panel_id: str, config: Dict[str, Any]) -> bool
```
- **Description**: Updates configuration for a specific panel
- **Parameters**:
  - `panel_id`: Panel identifier
  - `config`: New panel configuration
- **Returns**: True if panel was updated

```python
def add_panel(self, panel_id: str, config: Dict[str, Any]) -> bool
```
- **Description**: Adds a new panel to the dashboard
- **Parameters**:
  - `panel_id`: Panel identifier
  - `config`: Panel configuration
- **Returns**: True if panel was added

```python
def remove_panel(self, panel_id: str) -> bool
```
- **Description**: Removes a panel from the dashboard
- **Parameters**:
  - `panel_id`: Panel identifier
- **Returns**: True if panel was removed

```python
def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]
```
- **Description**: Handles dashboard-related commands
- **Parameters**:
  - `command`: Command name
  - `params`: Command parameters
- **Returns**: Command result

## Control Components

### ParameterControl

The `ParameterControl` provides an interface for viewing, editing, and applying parameter changes.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the parameter control
- **Parameters**: None

#### Methods

```python
def get_parameter_groups(self) -> List[Dict[str, Any]]
```
- **Description**: Gets parameter groups
- **Returns**: Parameter groups data

```python
def get_parameters(self, group_name: Optional[str] = None) -> List[Dict[str, Any]]
```
- **Description**: Gets parameters, optionally filtered by group
- **Parameters**:
  - `group_name`: Optional group name to filter by
- **Returns**: Parameter data

```python
def get_parameter(self, param_name: str) -> Optional[Dict[str, Any]]
```
- **Description**: Gets a specific parameter by name
- **Parameters**:
  - `param_name`: Parameter name
- **Returns**: Parameter data or None if not found

```python
def set_parameter_value(self, param_name: str, value: Any) -> Dict[str, Any]
```
- **Description**: Sets a parameter value
- **Parameters**:
  - `param_name`: Parameter name
  - `value`: New parameter value
- **Returns**: Result with success and error message if applicable

```python
def reset_parameter(self, param_name: str) -> Dict[str, Any]
```
- **Description**: Resets a parameter to its default value
- **Parameters**:
  - `param_name`: Parameter name
- **Returns**: Result with success and error message if applicable

```python
def reset_all_parameters(self) -> Dict[str, Any]
```
- **Description**: Resets all parameters to their default values
- **Returns**: Result with success and error message if applicable

```python
def export_configuration(self) -> Dict[str, Any]
```
- **Description**: Exports current configuration as JSON
- **Returns**: Result with success, config JSON, and error message if applicable

```python
def import_configuration(self, config_json: str) -> Dict[str, Any]
```
- **Description**: Imports configuration from JSON
- **Parameters**:
  - `config_json`: Configuration JSON string
- **Returns**: Result with success and error message if applicable

```python
def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]
```
- **Description**: Handles parameter control commands
- **Parameters**:
  - `command`: Command name
  - `params`: Command parameters
- **Returns**: Command result

## Alert Components

### AlertPanel

The `AlertPanel` displays system alerts and notifications from the event notification system.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the alert panel
- **Parameters**: None

#### Methods

```python
def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]
```
- **Description**: Renders the alert panel
- **Parameters**:
  - `config`: Panel configuration
- **Returns**: Panel data

```python
def get_filtered_alerts(self, filter_settings: Dict[str, Any]) -> List[Dict[str, Any]]
```
- **Description**: Gets alerts filtered by settings
- **Parameters**:
  - `filter_settings`: Filter settings
- **Returns**: Filtered alerts

```python
def acknowledge_alert(self, alert_id: str) -> bool
```
- **Description**: Acknowledges an alert
- **Parameters**:
  - `alert_id`: Alert/Event ID to acknowledge
- **Returns**: True if alert was acknowledged

```python
def acknowledge_all_visible(self, filter_settings: Dict[str, Any]) -> int
```
- **Description**: Acknowledges all visible alerts based on current filters
- **Parameters**:
  - `filter_settings`: Filter settings
- **Returns**: Number of alerts acknowledged

```python
def update_filter_settings(self, filter_settings: Dict[str, Any]) -> None
```
- **Description**: Updates alert filter settings
- **Parameters**:
  - `filter_settings`: New filter settings

```python
def reset_new_alerts_count(self) -> None
```
- **Description**: Resets the new alerts count
- **Parameters**: None

```python
def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]
```
- **Description**: Handles alert panel commands
- **Parameters**:
  - `command`: Command name
  - `params`: Command parameters
- **Returns**: Command result

## Adapters

### DigitalTwinAdapter

The `DigitalTwinAdapter` provides a standardized interface for interacting with the digital twin.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the digital twin adapter
- **Parameters**: None

#### Methods

```python
def get_current_state(self) -> Dict[str, Any]
```
- **Description**: Gets the current state of the digital twin
- **Returns**: Current state

```python
def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]
```
- **Description**: Gets historical states of the digital twin
- **Parameters**:
  - `limit`: Optional limit on the number of historical states to retrieve
- **Returns**: List of historical states

```python
def update_state(self, updates: Dict[str, Any]) -> bool
```
- **Description**: Updates the digital twin state
- **Parameters**:
  - `updates`: State updates
- **Returns**: True if update was successful

```python
def run_simulation(
    self, steps: int = 10, parameters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```
- **Description**: Runs a simulation using the digital twin
- **Parameters**:
  - `steps`: Number of simulation steps to run
  - `parameters`: Optional parameter modifications for the simulation
- **Returns**: List of simulated states

```python
def save_state(self, filename: Optional[str] = None) -> bool
```
- **Description**: Saves the current state to a file
- **Parameters**:
  - `filename`: Optional filename to save the state
- **Returns**: True if save was successful

```python
def load_state(self, filename: str) -> bool
```
- **Description**: Loads a state from a file
- **Parameters**:
  - `filename`: Filename to load the state from
- **Returns**: True if load was successful

```python
def save_scenario(
    self, name: str, parameters: Dict[str, Any], description: str = ""
) -> bool
```
- **Description**: Saves a simulation scenario
- **Parameters**:
  - `name`: Name for the scenario
  - `parameters`: Scenario parameters
  - `description`: Optional scenario description
- **Returns**: True if scenario was saved

```python
def run_scenario(self, scenario_name: str) -> List[Dict[str, Any]]
```
- **Description**: Runs a saved simulation scenario
- **Parameters**:
  - `scenario_name`: Name of the scenario to run
- **Returns**: List of simulated states

```python
def get_all_scenarios(self) -> Dict[str, Dict[str, Any]]
```
- **Description**: Gets all saved scenarios
- **Returns**: Dictionary of scenario data

```python
def compare_scenarios(
    self, scenario_names: List[str]
) -> Dict[str, Dict[str, float]]
```
- **Description**: Compares multiple scenarios
- **Parameters**:
  - `scenario_names`: List of scenario names to compare
- **Returns**: Comparison results

## Services

### CommandService

The `CommandService` provides centralized command handling and routing for the HMI.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the command service
- **Parameters**: None

#### Methods

```python
def register_handler(self, command: str, handler: Callable) -> None
```
- **Description**: Registers a command handler
- **Parameters**:
  - `command`: Command name
  - `handler`: Handler function
- **Raises**:
  - `ValueError`: If handler already registered

```python
def unregister_handler(self, command: str) -> bool
```
- **Description**: Unregisters a command handler
- **Parameters**:
  - `command`: Command name
- **Returns**: True if handler was unregistered

```python
def execute_command(
    self, command: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```
- **Description**: Executes a command
- **Parameters**:
  - `command`: Command name
  - `params`: Command parameters
- **Returns**: Command result

```python
def get_command_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]
```
- **Description**: Gets command execution history
- **Parameters**:
  - `limit`: Optional limit on the number of history items to retrieve
- **Returns**: Command history

## Integration Examples

### Registering a Component

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager

# Create your component
class MyComponent:
    def __init__(self):
        # Initialize your component
        pass

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # Handle commands
        if command == "my_command":
            # Process command
            return {"handled": True, "success": True}
        return {"handled": False}

# Register your component
my_component = MyComponent()
interface_manager.register_component("my_component", my_component)
```

### Subscribing to Events

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager

# Define event handler
def handle_view_change(event_data):
    view_name = event_data.get("view_name")
    print(f"View changed to: {view_name}")

# Register event handler
interface_manager.register_event_handler("view_changed", handle_view_change)
```

### Executing Commands

```python
from circman5.manufacturing.human_interface.services.command_service import command_service

# Execute a command
result = command_service.execute_command(
    "set_parameter",
    {
        "param_name": "temperature",
        "value": 25.5
    }
)

# Check result
if result.get("success"):
    print("Parameter set successfully")
else:
    print(f"Error: {result.get('error')}")
```

### Using the Digital Twin Adapter

```python
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter

# Get current state
state = digital_twin_adapter.get_current_state()

# Run a simulation
simulation_results = digital_twin_adapter.run_simulation(
    steps=10,
    parameters={"production_line": {"temperature": 22.5}}
)

# Process results
for step, result in enumerate(simulation_results):
    print(f"Step {step}: {result.get('production_line', {}).get('production_rate')}")
```

### Creating a Custom Dashboard

```python
from circman5.manufacturing.human_interface.core.dashboard_manager import dashboard_manager

# Create a custom dashboard
dashboard = dashboard_manager.create_layout(
    name="custom_dashboard",
    description="My custom dashboard",
    panels={
        "temperature": {
            "type": "chart_panel",
            "title": "Temperature Trends",
            "position": {"row": 0, "col": 0},
            "size": {"rows": 1, "cols": 1},
            "chart_type": "line",
            "metrics": ["temperature"]
        },
        "production": {
            "type": "kpi_panel",
            "title": "Production Metrics",
            "position": {"row": 0, "col": 1},
            "size": {"rows": 1, "cols": 1},
            "metrics": ["production_rate", "yield_rate"]
        }
    },
    layout_config={"rows": 2, "columns": 2, "spacing": 10}
)

# Activate the dashboard
dashboard_manager.set_active_layout("custom_dashboard")
```

### Managing Alerts

```python
from circman5.manufacturing.human_interface.components.alerts.alert_panel import alert_panel

# Get alerts with specific filters
alerts = alert_panel.get_filtered_alerts({
    "severity_levels": ["critical", "error"],
    "categories": ["system", "process"],
    "show_acknowledged": False
})

# Acknowledge an alert
alert_panel.acknowledge_alert("alert-123")

# Acknowledge all visible alerts
count = alert_panel.acknowledge_all_visible({
    "severity_levels": ["critical"]
})
print(f"Acknowledged {count} critical alerts")
```
