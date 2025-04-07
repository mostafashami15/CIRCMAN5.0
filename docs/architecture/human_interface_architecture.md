# Human Interface Architecture for CIRCMAN5.0

## 1. Introduction

This document describes the architecture of the Human-Machine Interface (HMI) system within the CIRCMAN5.0 framework. The HMI system provides operators and managers with a real-time, interactive interface to monitor and control photovoltaic manufacturing processes through the Digital Twin. The architecture follows a modular, event-driven design that emphasizes separation of concerns, extensibility, and real-time responsiveness.

## 2. Architecture Overview

The HMI system implements a layered architecture with clearly defined responsibilities:

```
+--------------------------------------------------+
|                  User Interface                  |
+--------------------------------------------------+
                         ↑↓
+--------------------------------------------------+
|                 HMI Components                   |
|  +------------------+  +--------------------+    |
|  |     Dashboard    |  |     Controls       |    |
|  +------------------+  +--------------------+    |
|  +------------------+  +--------------------+    |
|  |     Alerts       |  |     Panels         |    |
|  +------------------+  +--------------------+    |
+--------------------------------------------------+
                         ↑↓
+--------------------------------------------------+
|                   Core Services                  |
|  +------------------+  +--------------------+    |
|  | Interface Manager|  | Dashboard Manager  |    |
|  +------------------+  +--------------------+    |
|  +------------------+  +--------------------+    |
|  | Interface State  |  | Update Service     |    |
|  +------------------+  +--------------------+    |
+--------------------------------------------------+
                         ↑↓
+--------------------------------------------------+
|                     Adapters                     |
|  +------------------+  +--------------------+    |
|  | Config Adapter   |  | Digital Twin       |    |
|  +------------------+  | Adapter            |    |
|  +------------------+  +--------------------+    |
|  | Event Adapter    |  |                    |    |
|  +------------------+  +--------------------+    |
+--------------------------------------------------+
                         ↑↓
+--------------------------------------------------+
|               Digital Twin System                |
+--------------------------------------------------+
```

### 2.1 Key Architectural Patterns

The HMI architecture implements several key design patterns:

1. **Singleton Pattern**: Core managers (InterfaceManager, DashboardManager, InterfaceState) use the singleton pattern to ensure a single, global instance is available throughout the application.

2. **Observer Pattern**: The event system uses an observer pattern, allowing components to subscribe to events and react accordingly.

3. **Adapter Pattern**: Adapters provide standardized interfaces to external systems like the Digital Twin and configuration services.

4. **Command Pattern**: User interactions are processed as commands, providing a uniform mechanism for handling user input.

5. **State Pattern**: The interface state manages the UI state independent of the underlying Digital Twin state.

6. **Composite Pattern**: Dashboard layouts consist of composite panel structures that can be nested and configured.

### 2.2 System Components

The HMI system consists of these major components:

1. **Core Services**:
   - Interface Manager
   - Dashboard Manager
   - Interface State
   - Service Providers (Command, Data, Update)

2. **Component Layers**:
   - Dashboard Components
   - Control Components
   - Alert Components
   - Panel Components

3. **Adapters**:
   - Configuration Adapter
   - Digital Twin Adapter
   - Event Adapter

4. **Utilities**:
   - UI Utilities
   - Validation Utilities

## 3. Core Components

### 3.1 Interface Manager

The InterfaceManager serves as the central coordinator for the HMI system, responsible for:

1. Component registration and lifecycle management
2. Event processing and distribution
3. Command routing and execution
4. System initialization and shutdown
5. Coordinating communication between components

#### Key Features:

- **Singleton Implementation**: Ensures a single instance manages the entire interface
- **Thread Safety**: Uses reentrant locks for thread-safe operation
- **Event Distribution**: Routes events to appropriate handlers
- **Command Processing**: Routes and executes user commands

```python
# Singleton pattern implementation
def __new__(cls):
    """Ensure singleton pattern."""
    if cls._instance is None:
        cls._instance = super(InterfaceManager, cls).__new__(cls)
        cls._instance._initialized = False
    return cls._instance
```

```python
# Command handling example
def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a command from the interface.
    """
    self.logger.debug(f"Handling command: {command}, params: {params}")

    # Trigger command event
    self.trigger_event("command_executed", {"command": command, "params": params})

    # Process standard commands
    if command == "change_view":
        self.change_view(params.get("view_name", "main_dashboard"))
        return {"success": True}

    # If no standard command handled it, delegate to registered components
    for component in self.components.values():
        if hasattr(component, "handle_command"):
            try:
                result = component.handle_command(command, params)
                if result.get("handled", False):
                    return result
            except Exception as e:
                self.logger.error(f"Error in component command handler: {str(e)}")

    return {"success": False, "error": "Unknown command"}
```

### 3.2 Interface State

The InterfaceState manages the state of the user interface, including:

1. Active view tracking
2. Selected parameters
3. Panel expansion states
4. Alert filters
5. Custom view configurations
6. Parameter and process control states

#### Key Features:

- **State Persistence**: State is persisted between sessions
- **Thread Safety**: Protected by locks for thread-safe operation
- **Default Values**: Provides sensible defaults for all state elements
- **State Reset**: Ability to reset to default state

```python
# State persistence example
def _save_state(self) -> None:
    """Save interface state to persistent storage."""
    try:
        # Convert state to serializable format
        state_dict = {
            "active_view": self.active_view,
            "selected_parameters": list(self.selected_parameters),
            "expanded_panels": list(self.expanded_panels),
            "alert_filters": self.alert_filters,
            "custom_views": self.custom_views,
            "selected_parameter_group": self.selected_parameter_group,
            "parameter_edit_mode": self.parameter_edit_mode,
            "selected_process": self.selected_process,
            "process_control_mode": self.process_control_mode,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Get the interface directory
        interface_dir = results_manager.get_path("digital_twin")
        state_file = interface_dir / "interface_state.json"

        # Save state to file
        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)

    except Exception as e:
        self.logger.error(f"Error saving interface state: {str(e)}")
```

### 3.3 Dashboard Manager

The DashboardManager handles dashboard layouts and rendering, responsible for:

1. Layout creation and management
2. Panel configuration and positioning
3. Dashboard rendering and updating
4. Layout persistence
5. View changes coordination

#### Key Features:

- **Layout Templates**: Provides standard dashboard layouts
- **Custom Layouts**: Supports user-defined custom layouts
- **Persistence**: Layouts are persisted between sessions
- **Dynamic Rendering**: Renders dashboards based on current Digital Twin state

```python
# Dashboard rendering example
def render_dashboard(self, layout_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Render a dashboard layout.
    """
    # Determine which layout to use
    if layout_name:
        layout = self.get_layout(layout_name)
        if not layout:
            self.logger.warning(f"Layout not found: {layout_name}")
            return {"error": f"Layout not found: {layout_name}"}
    else:
        layout = self.current_layout
        if not layout:
            # If no active layout, try to use the one from interface state
            state_view = self.state.get_active_view()
            layout = self.get_layout(state_view)

            # If still no layout, use the first available
            if not layout and self.layouts:
                layout = next(iter(self.layouts.values()))

            if not layout:
                self.logger.warning("No layouts available to render")
                return {"error": "No layouts available"}

    # Set this as the active layout
    self.current_layout = layout

    # Get current digital twin state for panel rendering
    digital_twin = DigitalTwin()
    digital_twin_state = digital_twin.get_current_state()

    # Prepare dashboard data structure
    dashboard_data = {
        "layout": layout.to_dict(),
        "panels": {},
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Render each panel in the layout
    for panel_id, panel_config in layout.panels.items():
        panel_type = panel_config.get("type", "unknown")
        dashboard_data["panels"][panel_id] = render_panel(
            panel_type, panel_config, digital_twin_state
        )

    return dashboard_data
```

## 4. Component Layers

### 4.1 Dashboard Components

Dashboard components provide visualization and interaction with system data:

1. **Main Dashboard**: Primary system overview
2. **Process Panels**: Manufacturing process visualization
3. **KPI Panels**: Key Performance Indicators display
4. **Status Panels**: System status information

#### Example: Main Dashboard

The MainDashboard component coordinates the top-level dashboard experience:

```python
class MainDashboard:
    """
    Main dashboard component for the Human-Machine Interface.
    """

    def __init__(self):
        """Initialize the main dashboard."""
        self.logger = setup_logger("main_dashboard")
        self.constants = ConstantsService()

        # Get references
        self.state = interface_state
        self.event_adapter = EventAdapter()  # Get instance

        # Thread safety
        self._lock = threading.RLock()

        # Register with interface manager and dashboard manager
        interface_manager.register_component("main_dashboard", self)
        dashboard_manager.register_component("main_dashboard", self)

        # Register event handlers
        interface_manager.register_event_handler(
            "panel_toggled", self._on_panel_toggled
        )
```

### 4.2 Control Components

Control components enable user interaction with the manufacturing system:

1. **Parameter Control**: Parameter adjustment interfaces
2. **Process Control**: Manufacturing process control
3. **Scenario Control**: Scenario management for what-if analysis

### 4.3 Alert Components

Alert components manage system notifications:

1. **Alert Panel**: Visualization of alerts and notifications
2. **Notification Manager**: Management of alerts and notifications
3. **Event Subscriber**: Subscription to system events

### 4.4 Panel Components

Panel components provide modular visualization units:

1. **KPI Panel**: Displays Key Performance Indicators
2. **Process Panel**: Visualizes manufacturing processes
3. **Status Panel**: Shows system status information
4. **Chart Panel**: Displays data charts and graphs

## 5. Adapter Layer

### 5.1 Digital Twin Adapter

The Digital Twin Adapter provides a simplified interface to Digital Twin functionality:

1. State retrieval and updates
2. Command forwarding
3. Event subscription
4. Data transformation

### 5.2 Configuration Adapter

The Configuration Adapter provides access to configuration data:

1. UI configuration loading
2. Parameter validation
3. Default values

### 5.3 Event Adapter

The Event Adapter connects to the Event Notification System:

1. Event subscription management
2. Event transformation for UI consumption
3. Event filtering

## 6. Communication Patterns

### 6.1 Event-Based Communication

The HMI system uses event-based communication for loosely coupled interactions:

```
Component A         Interface Manager         Component B
     |                     |                      |
     | 1. trigger_event()  |                      |
     |-------------------->|                      |
     |                     | 2. process_event()   |
     |                     |--------------------->|
     |                     |                      |
     |                     | 3. handle_event()    |
     |                     |<---------------------|
     |                     |                      |
     | 4. update UI        |                      |
     |<--------------------|                      |
     |                     |                      |
```

#### Key Event Types:

1. **View Changes**: When the active view changes
2. **State Updates**: When underlying system state changes
3. **Parameter Changes**: When parameters are updated
4. **Alerts**: System alerts and notifications
5. **User Actions**: User interaction events

### 6.2 Command Processing

User interactions are processed through a command pattern:

```
User Interface      Interface Manager         Command Handler
     |                     |                       |
     | 1. handle_command() |                       |
     |-------------------->|                       |
     |                     | 2. route_command()    |
     |                     |---------------------->|
     |                     |                       |
     |                     | 3. execute_command()  |
     |                     |                       |
     |                     | 4. return result      |
     |                     |<----------------------|
     | 5. update UI        |                       |
     |<--------------------|                       |
     |                     |                       |
```

#### Common Commands:

1. **change_view**: Change the active view
2. **update_panel**: Update panel configuration
3. **toggle_panel**: Expand/collapse panel
4. **select_parameter**: Select parameter for monitoring
5. **execute_process**: Execute manufacturing process

### 6.3 State Synchronization

The HMI maintains its own state separate from the Digital Twin:

```
User Interface    Interface State    Digital Twin Adapter    Digital Twin
     |                  |                    |                    |
     | 1. update state  |                    |                    |
     |----------------->|                    |                    |
     |                  | 2. get DT state    |                    |
     |                  |------------------->|                    |
     |                  |                    | 3. get_state()     |
     |                  |                    |------------------->|
     |                  |                    |                    |
     |                  |                    | 4. return state    |
     |                  |                    |<-------------------|
     |                  | 5. update UI state |                    |
     |                  |<-------------------|                    |
     | 6. render updated|                    |                    |
     |<-----------------|                    |                    |
     |                  |                    |                    |
```

## 7. Dashboard Rendering Process

The dashboard rendering process flows through multiple components:

```
User         Interface Manager    Dashboard Manager    Panel Components    Digital Twin
  |                 |                    |                    |                 |
  | request view    |                    |                    |                 |
  |---------------->|                    |                    |                 |
  |                 | render_dashboard() |                    |                 |
  |                 |------------------->|                    |                 |
  |                 |                    | get Digital Twin   |                 |
  |                 |                    | state              |                 |
  |                 |                    |--------------------------------->|
  |                 |                    |                    |                 |
  |                 |                    |                    |                 |
  |                 |                    | get current state  |                 |
  |                 |                    |<---------------------------------|
  |                 |                    |                    |                 |
  |                 |                    | render panels      |                 |
  |                 |                    |------------------->|                 |
  |                 |                    |                    |                 |
  |                 |                    | return rendered    |                 |
  |                 |                    | panels             |                 |
  |                 |                    |<-------------------|                 |
  |                 |                    |                    |                 |
  |                 | return dashboard   |                    |                 |
  |                 |<-------------------|                    |                 |
  |                 |                    |                    |                 |
  | render dashboard|                    |                    |                 |
  |<----------------|                    |                    |                 |
  |                 |                    |                    |                 |
```

### 7.1 Dashboard Layout Structure

Dashboard layouts are structured as JSON-compatible objects:

```json
{
  "name": "main_dashboard",
  "description": "Main system dashboard",
  "panels": {
    "status": {
      "type": "status_panel",
      "title": "System Status",
      "position": {"row": 0, "col": 0},
      "size": {"rows": 1, "cols": 1}
    },
    "kpi": {
      "type": "kpi_panel",
      "title": "Key Performance Indicators",
      "position": {"row": 0, "col": 1},
      "size": {"rows": 1, "cols": 1}
    },
    "process": {
      "type": "process_panel",
      "title": "Manufacturing Process",
      "position": {"row": 1, "col": 0},
      "size": {"rows": 1, "cols": 2}
    }
  },
  "layout_config": {"rows": 2, "columns": 2, "spacing": 10}
}
```

## 8. State Management

### 8.1 Interface State Structure

The interface state maintains several key elements:

```json
{
  "active_view": "main_dashboard",
  "selected_parameters": ["temperature", "pressure", "flow_rate"],
  "expanded_panels": ["status", "kpi"],
  "alert_filters": {
    "severity_levels": ["critical", "error", "warning", "info"],
    "categories": ["system", "process", "user"],
    "show_acknowledged": false
  },
  "custom_views": {
    "production_overview": {
      "panels": {
        "production_rate": {"type": "kpi_panel", "position": {"row": 0, "col": 0}}
      }
    }
  },
  "selected_parameter_group": "thermal",
  "parameter_edit_mode": false,
  "selected_process": "silicon_wafer_processing",
  "process_control_mode": "monitor"
}
```

### 8.2 State Persistence

The interface state is persisted to disk:

1. Saved to JSON file on state changes
2. Loaded on system initialization
3. Fallback to defaults if load fails
4. Thread-safe operations for all state changes

## 9. Threading and Concurrency

### 9.1 Thread Safety

The HMI system implements thread safety through:

1. Reentrant locks in singleton components
2. State copies to prevent race conditions
3. Thread-safe event distribution
4. Immutable data structures where possible

```python
# Thread-safe state update example
def add_selected_parameter(self, parameter: str) -> None:
    """
    Add a parameter to the selected parameters set.
    """
    with self._lock:
        self.selected_parameters.add(parameter)
        self.logger.debug(f"Parameter selected: {parameter}")
        self._save_state()
```

### 9.2 Event Processing

Events are processed in separate threads:

1. Event handlers execute in the publisher's thread
2. Long-running operations should spawn worker threads
3. UI updates must be synchronized to the UI thread

## 10. Error Handling

### 10.1 Component Error Handling

Components implement consistent error handling:

```python
try:
    # Operation that might fail
    result = component.operation()
except Exception as e:
    self.logger.error(f"Error in component operation: {str(e)}")
    # Handle gracefully with fallback behavior
    result = default_value
```

### 10.2 User Feedback

Errors are reported to users through:

1. Alert panel notifications
2. Status indicators
3. Toast messages for transient errors
4. Error details in logs for troubleshooting

## 11. Configuration

### 11.1 UI Configuration

UI configuration is managed through configuration files:

```json
{
  "UI_CONFIG": {
    "theme": "light",
    "refresh_rate": 1000,
    "animation_enabled": true,
    "default_view": "main_dashboard"
  },
  "PANEL_CONFIG": {
    "default_expanded": ["status", "kpi"],
    "max_history_points": 100,
    "chart_update_frequency": 5000
  },
  "ALERT_CONFIG": {
    "notification_duration": 5000,
    "max_visible_alerts": 5,
    "default_severity_filter": ["critical", "error", "warning"]
  }
}
```

### 11.2 Layout Templates

Standard layout templates are provided:

1. Main Dashboard
2. Production Dashboard
3. Quality Dashboard
4. Alerts Dashboard

## 12. Extensibility

### 12.1 Adding New Components

New components can be integrated through:

1. Component registration with the InterfaceManager
2. Event subscription for state updates
3. Command handling for user interaction

```python
# Component registration example
def register_new_component():
    # Create new component
    new_component = NewComponent()

    # Register with interface manager
    interface_manager.register_component("new_component", new_component)

    # Subscribe to events
    interface_manager.register_event_handler(
        "state_changed", new_component.handle_state_change
    )

    # Initialize if needed
    if hasattr(new_component, "initialize"):
        new_component.initialize()
```

### 12.2 Custom Panels

The system supports custom panel development:

1. Create a new panel class
2. Implement the panel rendering interface
3. Register with the panel registry

## 13. Security Considerations

### 13.1 Authentication Integration

The HMI integrates with authentication systems:

1. User session management
2. Role-based access control
3. Command authorization
4. Secure state persistence

### 13.2 Authorization Controls

Authorization is enforced at multiple levels:

1. Command validation before execution
2. Parameter value constraints
3. View access restrictions
4. Action auditing

## 14. Performance Considerations

### 14.1 Rendering Optimization

Dashboard rendering is optimized through:

1. Selective updates of changed components
2. Lazy loading of panel data
3. Throttling of high-frequency updates
4. Pagination of large datasets

### 14.2 Resource Management

The HMI manages resources efficiently:

1. Image and asset caching
2. Data buffering for trend displays
3. Cleanup of unused resources
4. Memory usage monitoring

## 15. Integration Examples

### 15.1 Digital Twin Integration

```python
# Example of Digital Twin integration
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter

# Get Digital Twin state through adapter
state = digital_twin_adapter.get_current_state()

# Process state for UI consumption
ui_data = process_for_ui(state)

# Update UI components
update_dashboard(ui_data)
```

### 15.2 Command Execution

```python
# Example of command execution
def execute_process_command():
    # Create command parameters
    command_params = {
        "process_id": "silicon_wafer_processing",
        "parameters": {
            "temperature": 220.5,
            "pressure": 1.2
        },
        "mode": "automatic"
    }

    # Send command through interface manager
    result = interface_manager.handle_command("execute_process", command_params)

    # Handle result
    if result.get("success"):
        show_success_notification("Process started successfully")
    else:
        show_error_notification(f"Process start failed: {result.get('error')}")
```

## 16. Future Enhancements

The HMI architecture supports several planned enhancements:

1. **Mobile Interface**: Responsive design for mobile access
2. **Voice Control**: Voice command integration
3. **Augmented Reality**: AR visualization of manufacturing processes
4. **Advanced Analytics**: Enhanced data visualization and analysis
5. **Collaborative Features**: Multi-user interaction and annotation

## 17. Conclusion

The Human-Machine Interface architecture for CIRCMAN5.0 provides a robust, extensible framework for interacting with the Digital Twin system. Its modular design, event-driven communication, and separation of concerns enable flexible customization while maintaining system integrity. The architecture balances the needs for real-time performance, usability, and integration with the underlying Digital Twin system.
