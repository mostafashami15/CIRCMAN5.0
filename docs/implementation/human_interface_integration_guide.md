# Human Interface Integration Guide

## 1. Introduction

This guide provides instructions and best practices for integrating with the CIRCMAN5.0 Human-Machine Interface (HMI) system. Whether you're extending the interface with new components, connecting external systems, or customizing the visualization, this document will walk you through the integration process.

## 2. Integration Overview

The HMI system is designed with extensibility in mind, providing several integration points:

1. **Component Integration**: Adding new dashboard panels, controls, or visualization components
2. **External Systems Integration**: Connecting additional data sources or control systems
3. **Event Integration**: Subscribing to and publishing events
4. **Command Integration**: Adding custom command handlers
5. **Dashboard Customization**: Creating custom dashboard layouts and panels

### 2.1 Key Integration Concepts

Before diving into specific integrations, understand these key concepts:

- **Component Registration**: All components must be registered with the InterfaceManager
- **Event-Based Communication**: Use the event system for loose coupling between components
- **Command Pattern**: User interactions are processed as commands
- **Adapters**: Custom systems should be integrated via adapters
- **Thread Safety**: All integrations must be thread-safe

## 3. Adding New Interface Components

### 3.1 Component Types

The HMI supports several component types:

1. **Dashboard Components**: For visualization and monitoring (status panels, charts, etc.)
2. **Control Components**: For user interaction and control (buttons, sliders, etc.)
3. **Alert Components**: For notifications and alerts
4. **Data Components**: For data processing and transformation

### 3.2 Basic Component Structure

Here's the basic structure for a custom component:

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.adapters.services.constants_service import ConstantsService
import threading

class CustomComponent:
    """Custom component for the Human-Machine Interface."""

    def __init__(self):
        """Initialize the custom component."""
        # Setup logging
        self.logger = setup_logger("custom_component")
        self.constants = ConstantsService()

        # Component state
        self._state = {}

        # Thread safety
        self._lock = threading.RLock()

        # Register with interface manager
        interface_manager.register_component("custom_component", self)

        # Register event handlers
        interface_manager.register_event_handler(
            "view_changed", self._on_view_changed
        )

        self.logger.info("Custom Component initialized")

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle component-specific commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict with handled flag and results
        """
        if command == "custom_command":
            # Process custom command
            result = self._process_custom_command(params)
            return {"handled": True, "success": True, "result": result}

        # Not a command for this component
        return {"handled": False}

    def _on_view_changed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle view change events.

        Args:
            event_data: Event data
        """
        view_name = event_data.get("view_name")
        # Handle view change if needed
```

### 3.3 Registering Your Component

Register your component with the InterfaceManager to integrate it into the HMI system:

```python
# Create component instance
custom_component = CustomComponent()

# Register with interface manager
interface_manager.register_component("custom_component", custom_component)
```

### 3.4 Creating a Dashboard Panel Component

To create a dashboard panel that can be added to layouts:

```python
class CustomPanelComponent:
    """Custom panel component for dashboards."""

    def __init__(self):
        """Initialize the custom panel component."""
        self.logger = setup_logger("custom_panel")

        # Register with dashboard manager
        from circman5.manufacturing.human_interface.core.dashboard_manager import dashboard_manager
        dashboard_manager.register_component("custom_panel", self)

    def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the panel based on configuration.

        Args:
            config: Panel configuration

        Returns:
            Rendered panel data
        """
        # Process configuration
        title = config.get("title", "Custom Panel")
        data_source = config.get("data_source")

        # Fetch data if needed
        panel_data = self._fetch_data(data_source)

        # Return panel structure
        return {
            "type": "custom_panel",
            "title": title,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": panel_data,
            "config": config
        }

    def _fetch_data(self, data_source: str) -> Dict[str, Any]:
        """
        Fetch data for the panel.

        Args:
            data_source: Data source identifier

        Returns:
            Panel data
        """
        # Implement data fetching logic
        # This could connect to Digital Twin, database, or other sources
        return {"example_metric": 123.45}
```

### 3.5 Component Lifecycle

Implement these methods if your component needs lifecycle management:

```python
def initialize(self) -> bool:
    """
    Initialize the component. Called by InterfaceManager.initialize().

    Returns:
        True if initialization successful
    """
    try:
        # Initialization logic
        return True
    except Exception as e:
        self.logger.error(f"Initialization error: {str(e)}")
        return False

def shutdown(self) -> None:
    """
    Clean up resources and shut down component.
    Called by InterfaceManager.shutdown().
    """
    try:
        # Cleanup logic
    except Exception as e:
        self.logger.error(f"Shutdown error: {str(e)}")
```

## 4. Integrating with the Event System

### 4.1 Subscribing to Events

Subscribe to events from the InterfaceManager:

```python
# Define event handler
def handle_state_change(event_data):
    """Handle state change events."""
    # Process event data
    print(f"State changed: {event_data}")

# Register with interface manager
interface_manager.register_event_handler("state_changed", handle_state_change)
```

### 4.2 Publishing Events

Trigger events through the InterfaceManager:

```python
# Create event data
event_data = {
    "source": "custom_component",
    "timestamp": datetime.datetime.now().isoformat(),
    "details": {
        "parameter": "temperature",
        "value": 25.5,
        "previous_value": 24.8
    }
}

# Trigger event
interface_manager.trigger_event("parameter_changed", event_data)
```

### 4.3 Custom Event Types

For complex integrations, define custom event types:

```python
# Define event type constants
CUSTOM_EVENT_TYPE = "custom_system_event"
CUSTOM_STATE_CHANGE = "custom_state_change"

# Register handlers for custom events
interface_manager.register_event_handler(CUSTOM_EVENT_TYPE, handle_custom_event)
interface_manager.register_event_handler(CUSTOM_STATE_CHANGE, handle_custom_state_change)

# Trigger custom events
interface_manager.trigger_event(CUSTOM_EVENT_TYPE, custom_event_data)
```

### 4.4 Digital Twin Event Integration

To integrate with Digital Twin events, use the EventAdapter:

```python
from circman5.manufacturing.human_interface.adapters.event_adapter import EventAdapter

# Create event adapter instance
event_adapter = EventAdapter()

# Define callback
def handle_digital_twin_event(event):
    """Handle events from Digital Twin."""
    # Process event
    print(f"Digital Twin event: {event.message}")

    # Convert to interface event if needed
    interface_manager.trigger_event(
        "digital_twin_event",
        {
            "message": event.message,
            "severity": event.severity.value,
            "category": event.category.value
        }
    )

# Register callback for all categories
event_adapter.register_callback(handle_digital_twin_event)

# Or for specific categories
from circman5.manufacturing.digital_twin.event_notification.event_types import EventCategory
event_adapter.register_callback(
    handle_threshold_events,
    category=EventCategory.THRESHOLD
)
```

## 5. Command System Integration

### 5.1 Handling Commands

Implement the `handle_command` method in your component:

```python
def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle component-specific commands.

    Args:
        command: Command name
        params: Command parameters

    Returns:
        Command result
    """
    if command == "custom_operation":
        # Process command
        try:
            result = self._perform_operation(params)
            return {
                "handled": True,
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "handled": True,
                "success": False,
                "error": str(e)
            }

    # Not handled by this component
    return {"handled": False}
```

### 5.2 Registering Custom Command Handlers

For more complex scenarios, register handlers with the CommandService:

```python
from circman5.manufacturing.human_interface.services.command_service import command_service

# Define command handler
def handle_custom_system_command(params):
    """Handle system-level command."""
    # Process command
    return {"success": True, "message": "Command processed"}

# Register handler
command_service.register_handler("custom_system_command", handle_custom_system_command)
```

### 5.3 Executing Commands

Execute commands through the CommandService:

```python
from circman5.manufacturing.human_interface.services.command_service import command_service

# Execute command
result = command_service.execute_command(
    "set_parameter",
    {
        "param_name": "custom_param",
        "value": "custom_value"
    }
)

# Check result
if result.get("success"):
    print("Command executed successfully")
else:
    print(f"Command failed: {result.get('error')}")
```

### 5.4 Command Chain

For complex operations, chain multiple commands:

```python
def execute_process_sequence():
    """Execute a sequence of operations."""
    # Start process
    result = command_service.execute_command("start_process", {"process_id": "p1"})
    if not result.get("success"):
        return result

    # Set parameters
    result = command_service.execute_command(
        "set_parameters",
        {
            "parameters": {
                "temperature": 25.5,
                "pressure": 1.2
            }
        }
    )
    if not result.get("success"):
        # Rollback if needed
        command_service.execute_command("stop_process", {"process_id": "p1"})
        return result

    # Complete sequence
    return {"success": True, "message": "Process sequence completed"}
```

## 6. Dashboard Integration

### 6.1 Creating Custom Dashboard Layouts

Create custom dashboards programmatically:

```python
from circman5.manufacturing.human_interface.core.dashboard_manager import dashboard_manager

# Create custom dashboard
custom_dashboard = dashboard_manager.create_layout(
    name="custom_view",
    description="Custom monitoring view",
    panels={
        "custom_panel": {
            "type": "custom_panel",
            "title": "Custom Data",
            "position": {"row": 0, "col": 0},
            "size": {"rows": 1, "cols": 1},
            "data_source": "custom_metrics"
        },
        "status": {
            "type": "status_panel",
            "title": "System Status",
            "position": {"row": 0, "col": 1},
            "size": {"rows": 1, "cols": 1}
        },
        "chart": {
            "type": "chart_panel",
            "title": "Custom Metrics Trend",
            "position": {"row": 1, "col": 0},
            "size": {"rows": 1, "cols": 2},
            "chart_type": "line",
            "metrics": ["custom_metric_1", "custom_metric_2"]
        }
    },
    layout_config={"rows": 2, "columns": 2, "spacing": 10}
)

# Activate the dashboard
dashboard_manager.set_active_layout("custom_view")
```

### 6.2 Adding Panels to Existing Dashboards

Add new panels to existing dashboards:

```python
from circman5.manufacturing.human_interface.components.dashboard.main_dashboard import main_dashboard

# Add panel to main dashboard
main_dashboard.add_panel(
    "custom_panel",
    {
        "type": "custom_panel",
        "title": "Custom Data",
        "position": {"row": 2, "col": 0},
        "size": {"rows": 1, "cols": 2},
        "data_source": "custom_metrics"
    }
)
```

### 6.3 Saving Custom User Views

Allow users to save their custom dashboard configurations:

```python
from circman5.manufacturing.human_interface.core.interface_state import interface_state

# Save custom view
interface_state.save_custom_view(
    "user_custom_view",
    {
        "panels": {
            "custom_panel": {
                "type": "custom_panel",
                "title": "User Custom View",
                "position": {"row": 0, "col": 0},
                "size": {"rows": 1, "cols": 1}
            }
        },
        "layout_config": {"rows": 1, "columns": 1, "spacing": 10}
    }
)

# Retrieve saved view
user_view = interface_state.get_custom_view("user_custom_view")
```

## 7. External System Integration

### 7.1 Creating a System Adapter

For integrating external systems, create an adapter:

```python
import threading
import datetime
from circman5.utils.logging_config import setup_logger

class ExternalSystemAdapter:
    """Adapter for external systems."""

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ExternalSystemAdapter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the adapter."""
        if self._initialized:
            return

        self.logger = setup_logger("external_system_adapter")

        # Connection state
        self.is_connected = False
        self.last_update = None

        # Thread safety
        self._lock = threading.RLock()

        # Initialize connection
        self._initialize_connection()

        self._initialized = True
        self.logger.info("External System Adapter initialized")

    def _initialize_connection(self):
        """Initialize connection to external system."""
        try:
            # Connection logic
            self.is_connected = True
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            self.is_connected = False

    def get_data(self, data_id: str) -> Dict[str, Any]:
        """
        Get data from external system.

        Args:
            data_id: Data identifier

        Returns:
            Data from external system
        """
        with self._lock:
            if not self.is_connected:
                self._initialize_connection()

            if not self.is_connected:
                return {"error": "Not connected"}

            try:
                # Data retrieval logic
                # This would connect to the actual external system

                # Example placeholder data
                data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "value": 123.45,
                    "status": "normal"
                }

                self.last_update = datetime.datetime.now()
                return data

            except Exception as e:
                self.logger.error(f"Data retrieval error: {str(e)}")
                return {"error": str(e)}

    def send_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send command to external system.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Command result
        """
        with self._lock:
            if not self.is_connected:
                self._initialize_connection()

            if not self.is_connected:
                return {"success": False, "error": "Not connected"}

            try:
                # Command sending logic
                # This would connect to the actual external system

                # Example placeholder response
                result = {
                    "success": True,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "response": "Command acknowledged"
                }

                return result

            except Exception as e:
                self.logger.error(f"Command error: {str(e)}")
                return {"success": False, "error": str(e)}
```

### 7.2 Integrating External Data with Digital Twin

Connect external system data to the Digital Twin:

```python
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter
from external_system_adapter import ExternalSystemAdapter

def integrate_external_data():
    """Integrate external system data with Digital Twin."""
    # Get adapters
    external_adapter = ExternalSystemAdapter()

    # Fetch external data
    external_data = external_adapter.get_data("production_metrics")

    if "error" in external_data:
        return {"success": False, "error": external_data["error"]}

    # Transform data for Digital Twin
    dt_updates = {
        "external_system": {
            "timestamp": external_data["timestamp"],
            "metrics": {
                "production_rate": external_data.get("value", 0),
                "status": external_data.get("status", "unknown")
            }
        }
    }

    # Update Digital Twin
    success = digital_twin_adapter.update_state(dt_updates)

    return {"success": success}
```

### 7.3 Creating a Data Service

For more complex integrations, create a dedicated data service:

```python
import threading
import time
from circman5.utils.logging_config import setup_logger
from external_system_adapter import ExternalSystemAdapter
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager

class ExternalDataService:
    """Service for integrating external data."""

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ExternalDataService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the data service."""
        if self._initialized:
            return

        self.logger = setup_logger("external_data_service")

        # Get adapters
        self.external_adapter = ExternalSystemAdapter()

        # Service state
        self.running = False
        self.update_interval = 60  # seconds
        self._update_thread = None

        # Thread safety
        self._lock = threading.RLock()

        # Register with interface manager
        interface_manager.register_component("external_data_service", self)

        self._initialized = True
        self.logger.info("External Data Service initialized")

    def start(self, interval: int = 60) -> bool:
        """
        Start the data service.

        Args:
            interval: Update interval in seconds

        Returns:
            True if started successfully
        """
        with self._lock:
            if self.running:
                return True

            self.update_interval = interval
            self.running = True

            # Start update thread
            self._update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True
            )
            self._update_thread.start()

            self.logger.info(f"External Data Service started (interval: {interval}s)")
            return True

    def stop(self) -> bool:
        """
        Stop the data service.

        Returns:
            True if stopped successfully
        """
        with self._lock:
            if not self.running:
                return True

            self.running = False

            # Thread will exit on next iteration
            if self._update_thread:
                self._update_thread.join(timeout=5)

            self.logger.info("External Data Service stopped")
            return True

    def _update_loop(self):
        """Update loop for fetching and integrating data."""
        while self.running:
            try:
                self._fetch_and_integrate_data()
            except Exception as e:
                self.logger.error(f"Update error: {str(e)}")

            # Wait for next interval
            sleep_time = self.update_interval
            while sleep_time > 0 and self.running:
                time.sleep(1)
                sleep_time -= 1

    def _fetch_and_integrate_data(self):
        """Fetch and integrate external data."""
        # Fetch data
        external_data = self.external_adapter.get_data("production_metrics")

        if "error" in external_data:
            self.logger.warning(f"Data fetch error: {external_data['error']}")
            return

        # Transform data for Digital Twin
        dt_updates = {
            "external_system": {
                "timestamp": external_data["timestamp"],
                "metrics": {
                    "production_rate": external_data.get("value", 0),
                    "status": external_data.get("status", "unknown")
                }
            }
        }

        # Update Digital Twin
        try:
            success = digital_twin_adapter.update_state(dt_updates)
            if success:
                self.logger.debug("Digital Twin updated with external data")
            else:
                self.logger.warning("Failed to update Digital Twin")
        except Exception as e:
            self.logger.error(f"Digital Twin update error: {str(e)}")

        # Trigger event for UI components
        interface_manager.trigger_event(
            "external_data_updated",
            {
                "source": "external_system",
                "timestamp": external_data["timestamp"],
                "data": external_data
            }
        )

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle service commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Command result
        """
        if command == "start_external_service":
            interval = params.get("interval", 60)
            success = self.start(interval)
            return {"handled": True, "success": success}

        elif command == "stop_external_service":
            success = self.stop()
            return {"handled": True, "success": success}

        elif command == "fetch_external_data":
            try:
                self._fetch_and_integrate_data()
                return {"handled": True, "success": True}
            except Exception as e:
                return {
                    "handled": True,
                    "success": False,
                    "error": str(e)
                }

        # Not handled by this service
        return {"handled": False}
```

## 8. Custom Data Visualization

### 8.1 Creating a Custom Chart Panel

Implement a custom chart panel component:

```python
import datetime
from circman5.utils.logging_config import setup_logger
from circman5.manufacturing.human_interface.core.dashboard_manager import dashboard_manager
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import digital_twin_adapter

class CustomChartPanel:
    """Custom chart panel component."""

    def __init__(self):
        """Initialize the custom chart panel."""
        self.logger = setup_logger("custom_chart_panel")

        # Register with dashboard manager
        dashboard_manager.register_component("custom_chart_panel", self)

        self.logger.info("Custom Chart Panel initialized")

    def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the chart panel.

        Args:
            config: Panel configuration

        Returns:
            Rendered panel data
        """
        # Get configuration
        title = config.get("title", "Custom Chart")
        metrics = config.get("metrics", [])
        time_range = config.get("time_range", "1h")
        chart_type = config.get("chart_type", "line")

        # Get historical data from digital twin
        history = self._get_history_data(metrics, time_range)

        # Process data for chart
        chart_data = self._process_chart_data(history, metrics)

        # Return panel structure
        return {
            "type": "custom_chart_panel",
            "title": title,
            "timestamp": datetime.datetime.now().isoformat(),
            "chart_type": chart_type,
            "metrics": metrics,
            "data": chart_data,
            "config": config
        }

    def _get_history_data(self, metrics: List[str], time_range: str) -> List[Dict[str, Any]]:
        """
        Get historical data from Digital Twin.

        Args:
            metrics: List of metrics to retrieve
            time_range: Time range (e.g., "1h", "24h", "7d")

        Returns:
            Historical data
        """
        # Convert time range to state history limit
        if time_range == "1h":
            limit = 60  # Assuming 1-minute intervals
        elif time_range == "24h":
            limit = 1440  # Assuming 1-minute intervals
        elif time_range == "7d":
            limit = 10080  # Assuming 1-minute intervals
        else:
            limit = 100  # Default

        # Get state history
        try:
            history = digital_twin_adapter.get_state_history(limit)
            return history
        except Exception as e:
            self.logger.error(f"Error getting history: {str(e)}")
            return []

    def _process_chart_data(
        self, history: List[Dict[str, Any]], metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Process historical data for chart visualization.

        Args:
            history: Historical state data
            metrics: Metrics to include

        Returns:
            Processed chart data
        """
        # Extract timestamps and metric values
        timestamps = []
        values = {metric: [] for metric in metrics}

        for state in history:
            # Add timestamp
            if "timestamp" in state:
                timestamps.append(state["timestamp"])
            else:
                # Skip states without timestamp
                continue

            # Extract metric values
            for metric in metrics:
                # Handle dotted path notation (e.g., "production_line.temperature")
                parts = metric.split(".")
                value = state

                try:
                    for part in parts:
                        value = value[part]

                    values[metric].append(value)
                except (KeyError, TypeError):
                    # Use None for missing values
                    values[metric].append(None)

        # Prepare chart data
        return {
            "timestamps": timestamps,
            "values": values
        }
```

### 8.2 Adding the Panel to a Dashboard

Add the custom chart panel to a dashboard:

```python
from circman5.manufacturing.human_interface.core.dashboard_manager import dashboard_manager

# Get main dashboard
layout = dashboard_manager.get_layout("main_dashboard")

# Add custom chart panel
if layout:
    layout.panels["custom_chart"] = {
        "type": "custom_chart_panel",
        "title": "Production Metrics",
        "position": {"row": 2, "col": 0},
        "size": {"rows": 1, "cols": 2},
        "metrics": [
            "production_line.temperature",
            "production_line.production_rate",
            "production_line.energy_consumption"
        ],
        "time_range": "24h",
        "chart_type": "line"
    }

    # Update the layout
    dashboard_manager.update_layout(layout)
```

## 9. Alert System Integration

### 9.1 Creating Custom Alerts

Generate custom alerts for the alert panel:

```python
from circman5.manufacturing.digital_twin.event_notification.event_types import (
    Event, EventCategory, EventSeverity
)
from circman5.manufacturing.digital_twin.event_notification.event_manager import event_manager

def generate_custom_alert(message, severity=EventSeverity.WARNING, details=None):
    """
    Generate a custom alert.

    Args:
        message: Alert message
        severity: Alert severity
        details: Optional alert details
    """
    # Create event
    event = Event(
        category=EventCategory.CUSTOM,
        severity=severity,
        source="custom_component",
        message=message,
        details=details or {}
    )

    # Publish event
    event_manager.publish(event)
```

### 9.2 Alert Filtering

Configure custom alert filtering:

```python
from circman5.manufacturing.human_interface.core.interface_state import interface_state

# Update alert filters
interface_state.update_alert_filters({
    "severity_levels": ["critical", "error", "warning"],
    "categories": ["system", "process", "custom"],
    "show_acknowledged": False
})
```

## 10. Parameter Control Integration

### 10.1 Integrating Custom Parameters

Add custom parameter control:

```python
from circman5.manufacturing.human_interface.components.controls.parameter_control import parameter_control
from circman5.manufacturing.digital_twin.configuration.parameter_definition import (
    ParameterDefinition, ParameterGroup, ParameterCategory, ParameterType
)

def add_custom_parameters():
    """Add custom parameters to the configuration system."""
    # Define custom parameter group
    custom_group = ParameterGroup(
        name="custom_system",
        display_name="Custom System",
        description="Parameters for custom system integration"
    )

    # Define parameters
    custom_parameters = [
        ParameterDefinition(
            name="custom_update_interval",
            display_name="Update Interval",
            description="Time between updates in seconds",
            parameter_type=ParameterType.INTEGER,
            default_value=60,
            min_value=10,
            max_value=3600,
            unit="seconds",
            category=ParameterCategory.PERFORMANCE
        ),
        ParameterDefinition(
            name="custom_threshold",
            display_name="Alert Threshold",
            description="Threshold for generating alerts",
            parameter_type=ParameterType.FLOAT,
            default_value=50.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
            category=ParameterCategory.ALERT
        ),
        ParameterDefinition(
            name="custom_mode",
            display_name="Operation Mode",
            description="System operation mode",
            parameter_type=ParameterType.STRING,
            default_value="normal",
            allowed_values=["normal", "economical", "performance"],
            category=ParameterCategory.OPERATION
        )
    ]

    # Register parameters with configuration system
    # Note: This is conceptual - actual parameter registration
    # depends on the specific implementation of the config system

    return custom_parameters
```

### 10.2 Handling Parameter Changes

Respond to parameter changes:

```python
# Define parameter change handler
def handle_parameter_change(event):
    """
    Handle parameter change events.

    Args:
        event: Parameter change event
    """
    # Extract details
    param_name = event.details.get("parameter_name")
    new_value = event.details.get("new_value")
    old_value = event.details.get("old_value")

    # Handle specific parameters
    if param_name == "custom_update_interval":
        # Adjust update interval
        service = ExternalDataService()
        if service.running:
            service.stop()
            service.start(interval=new_value)

    elif param_name == "custom_threshold":
        # Update threshold logic
        pass

    elif param_name == "custom_mode":
        # Switch operation mode
        if new_value == "normal":
            # Normal mode logic
            pass
        elif new_value == "economical":
            # Economical mode logic
            pass
        elif new_value == "performance":
            # Performance mode logic
            pass

# Register event handler
from circman5.manufacturing.human_interface.adapters.event_adapter import EventAdapter
event_adapter = EventAdapter()
event_adapter.register_callback(
    handle_parameter_change,
    event_type="parameter_change"
)
```

## 11. Security Integration

### 11.1 Command Authorization

Add authorization checks to command handling:

```python
from circman5.security.authorization import is_authorized  # Hypothetical module

def handle_secured_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle command with authorization checks.

    Args:
        command: Command name
        params: Command parameters

    Returns:
        Command result
    """
    # Check for command requiring authorization
    if command == "critical_operation":
        # Check authorization
        user_id = params.get("user_id")
        if not user_id or not is_authorized(user_id, "critical_operations"):
            return {
                "handled": True,
                "success": False,
                "error": "Unauthorized access"
            }

        # Execute authorized command
        return self._execute_critical_operation(params)

    # Other commands
    return {"handled": False}
```

### 11.2 Secure Event Handling

Apply security filtering to events:

```python
def handle_secured_event(event_data):
    """
    Handle events with security filtering.

    Args:
        event_data: Event data
    """
    # Extract user information
    user_id = event_data.get("user_id")
    user_role = event_data.get("user_role")

    # Security filtering
    if event_data.get("type") == "sensitive_data" and user_role != "admin":
        # Redact sensitive information
        redacted_data = redact_sensitive_info(event_data)
        # Process redacted data
    else:
        # Process complete data
        pass
```

## 12. Implementing Custom Views

### 12.1 Creating a Custom View

Implement a specialized view for custom functionality:

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
from circman5.manufacturing.human_interface.core.dashboard_manager import dashboard_manager
import threading

class CustomViewComponent:
    """Custom view for specialized functionality."""

    def __init__(self):
        """Initialize the custom view component."""
        self.logger = setup_logger("custom_view")

        # Component state
        self._state = {
            "active": False,
            "data": {},
            "settings": {}
        }

        # Thread safety
        self._lock = threading.RLock()

        # Register with interface manager
        interface_manager.register_component("custom_view", self)

        # Register event handlers
        interface_manager.register_event_handler(
            "view_changed", self._on_view_changed
        )

        # Create dashboard layout
        self._create_dashboard_layout()

        self.logger.info("Custom View Component initialized")

    def _create_dashboard_layout(self):
        """Create dashboard layout for custom view."""
        try:
            dashboard_manager.create_layout(
                name="custom_view",
                description="Specialized view for custom functionality",
                panels={
                    "main": {
                        "type": "custom_panel",
                        "title": "Custom Data View",
                        "position": {"row": 0, "col": 0},
                        "size": {"rows": 1, "cols": 2}
                    },
                    "controls": {
                        "type": "control_panel",
                        "title": "Custom Controls",
                        "position": {"row": 1, "col": 0},
                        "size": {"rows": 1, "cols": 1}
                    },
                    "details": {
                        "type": "detail_panel",
                        "title": "Details",
                        "position": {"row": 1, "col": 1},
                        "size": {"rows": 1, "cols": 1}
                    }
                },
                layout_config={"rows": 2, "columns": 2, "spacing": 10}
            )
        except ValueError:
            # Layout already exists
            self.logger.debug("Custom view layout already exists")

    def _on_view_changed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle view change events.

        Args:
            event_data: Event data
        """
        view_name = event_data.get("view_name")

        with self._lock:
            # Update active state based on view
            self._state["active"] = (view_name == "custom_view")

            if self._state["active"]:
                # View is becoming active
                self._on_view_activated()
            else:
                # View is being deactivated
                self._on_view_deactivated()

    def _on_view_activated(self):
        """Handle view activation."""
        # Load data, initialize view
        self.logger.debug("Custom view activated")

    def _on_view_deactivated(self):
        """Handle view deactivation."""
        # Clean up, save state
        self.logger.debug("Custom view deactivated")

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle custom view commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Command result
        """
        if command == "activate_custom_view":
            # Switch to custom view
            interface_manager.change_view("custom_view")
            return {"handled": True, "success": True}

        elif command == "custom_view_operation":
            # Only process if view is active
            if not self._state["active"]:
                return {
                    "handled": True,
                    "success": False,
                    "error": "View not active"
                }

            # Process custom operation
            try:
                result = self._execute_custom_operation(params)
                return {"handled": True, "success": True, "result": result}
            except Exception as e:
                return {
                    "handled": True,
                    "success": False,
                    "error": str(e)
                }

        # Not handled by this component
        return {"handled": False}

    def _execute_custom_operation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute custom operation.

        Args:
            params: Operation parameters

        Returns:
            Operation result
        """
        # Custom operation implementation
        return {"status": "completed"}
```

### 12.2 Activating the Custom View

Switch to the custom view:

```python
from circman5.manufacturing.human_interface.core.interface_manager import interface_manager

# Change to custom view
interface_manager.change_view("custom_view")
```

## 13. Best Practices

### 13.1 Concurrency and Thread Safety

Follow these best practices for thread safety:

1. **Use Locks**: Protect shared state with locks
   ```python
   with self._lock:
       # Modify shared state
       self._state["value"] = new_value
   ```

2. **Immutable Data**: Use immutable data structures where possible
   ```python
   # Use copy to avoid modifying original
   def get_state(self):
       with self._lock:
           return self._state.copy()
   ```

3. **Atomic Operations**: Keep critical sections small and focused
   ```python
   def increment_counter(self):
       with self._lock:
           self._counter += 1
           return self._counter
   ```

4. **Thread-safe Callbacks**: Be careful with callback execution
   ```python
   def trigger_callbacks(self, event):
       # Create a copy of callbacks to avoid modifying during iteration
       callbacks = list(self._callbacks)

       # Execute callbacks outside lock
       for callback in callbacks:
           try:
               callback(event)
           except Exception as e:
               self.logger.error(f"Callback error: {str(e)}")
   ```

### 13.2 Resource Management

Manage resources efficiently:

1. **Cleanup Resources**: Release resources properly
   ```python
   def shutdown(self):
       try:
           # Close connections
           if self._connection:
               self._connection.close()

           # Cancel timers
           if self._timer:
               self._timer.cancel()

           # Join threads
           if self._thread and self._thread.is_alive():
               self._thread.join(timeout=5)
       except Exception as e:
           self.logger.error(f"Shutdown error: {str(e)}")
   ```

2. **Lazy Initialization**: Initialize resources when needed
   ```python
   def get_connection(self):
       with self._lock:
           if self._connection is None:
               self._connection = self._create_connection()
           return self._connection
   ```

3. **Resource Pooling**: Reuse expensive resources
   ```python
   def get_connection_from_pool(self):
       with self._lock:
           if not self._connection_pool:
               self._initialize_connection_pool()

           connection = self._connection_pool.pop()
           return connection
   ```

### 13.3 Error Handling

Implement robust error handling:

1. **Graceful Degradation**: Degrade gracefully on errors
   ```python
   def get_data(self):
       try:
           return self._fetch_data()
       except ConnectionError:
           self.logger.warning("Connection error, using cached data")
           return self._get_cached_data()
       except Exception as e:
           self.logger.error(f"Unexpected error: {str(e)}")
           return None
   ```

2. **Meaningful Error Messages**: Provide clear error information
   ```python
   def process_command(self, command):
       if not self._validate_command(command):
           return {
               "success": False,
               "error": f"Invalid command format: {command}",
               "expected_format": self._command_format
           }
   ```

3. **Logging**: Log errors with context
   ```python
   def execute_operation(self, operation_id, params):
       try:
           result = self._operations[operation_id](**params)
           return result
       except KeyError:
           self.logger.error(f"Unknown operation: {operation_id}")
           return None
       except Exception as e:
           self.logger.error(
               f"Operation {operation_id} failed: {str(e)}",
               exc_info=True
           )
           return None
   ```

### 13.4 Performance Optimization

Optimize for performance:

1. **Caching**: Cache expensive results
   ```python
   def get_data(self, data_id):
       # Check cache first
       if data_id in self._cache and not self._is_cache_stale(data_id):
           return self._cache[data_id]

       # Fetch data
       data = self._fetch_data(data_id)

       # Update cache
       self._cache[data_id] = data
       self._cache_timestamps[data_id] = time.time()

       return data
   ```

2. **Lazy Loading**: Load data only when needed
   ```python
   def get_details(self, item_id):
       # Basic info already loaded
       basic_info = self._items[item_id]

       # Load details only when requested
       if "details" not in basic_info:
           basic_info["details"] = self._load_details(item_id)

       return basic_info
   ```

3. **Batching**: Batch operations where possible
   ```python
   def update_multiple_items(self, updates):
       # Batch updates instead of individual calls
       batch_update = []

       for item_id, new_value in updates.items():
           batch_update.append((item_id, new_value))

       return self._batch_update(batch_update)
   ```

## 14. Testing Your Integration

### 14.1 Unit Testing

Test your component in isolation:

```python
import unittest
from unittest.mock import MagicMock, patch

class TestCustomComponent(unittest.TestCase):
    def setUp(self):
        # Create mocks
        self.interface_manager_mock = MagicMock()
        self.digital_twin_mock = MagicMock()

        # Patch modules
        self.patches = [
            patch('circman5.manufacturing.human_interface.core.interface_manager.interface_manager',
                  self.interface_manager_mock),
            patch('circman5.manufacturing.human_interface.adapters.digital_twin_adapter.digital_twin_adapter',
                  self.digital_twin_mock)
        ]

        for p in self.patches:
            p.start()

        # Create component
        from custom_component import CustomComponent
        self.component = CustomComponent()

    def tearDown(self):
        # Remove patches
        for p in self.patches:
            p.stop()

    def test_initialization(self):
        """Test component initialization."""
        # Verify registration with interface manager
        self.interface_manager_mock.register_component.assert_called_with(
            "custom_component", self.component
        )

    def test_handle_command(self):
        """Test command handling."""
        # Setup mock
        self.component._process_custom_command = MagicMock(return_value={"result": "success"})

        # Test command handling
        result = self.component.handle_command("custom_command", {"param": "value"})

        # Verify result
        self.assertTrue(result["handled"])
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], {"result": "success"})

        # Verify processing call
        self.component._process_custom_command.assert_called_with({"param": "value"})
```

### 14.2 Integration Testing

Test your component's integration with the HMI system:

```python
import unittest

class TestCustomComponentIntegration(unittest.TestCase):
    def setUp(self):
        # Import actual interface manager
        from circman5.manufacturing.human_interface.core.interface_manager import interface_manager
        self.interface_manager = interface_manager

        # Create component
        from custom_component import CustomComponent
        self.component = CustomComponent()

    def test_event_handling(self):
        """Test event handling integration."""
        # Setup test event handler
        event_received = False
        event_data = None

        def test_handler(data):
            nonlocal event_received, event_data
            event_received = True
            event_data = data

        # Register handler
        self.interface_manager.register_event_handler("custom_event", test_handler)

        # Trigger event
        test_event_data = {"source": "test", "value": 123}
        self.interface_manager.trigger_event("custom_event", test_event_data)

        # Verify event was received
        self.assertTrue(event_received)
        self.assertEqual(event_data, test_event_data)

    def test_command_execution(self):
        """Test command execution integration."""
        # Import command service
        from circman5.manufacturing.human_interface.services.command_service import command_service

        # Execute command
        result = command_service.execute_command(
            "custom_command", {"param": "test_value"}
        )

        # Verify result
        self.assertTrue(result["handled"])
        self.assertTrue(result["success"])
```

## 15. Example Use Cases

### 15.1 Equipment Monitoring Integration

Integrate equipment monitoring systems:

```python
class EquipmentMonitoringComponent:
    """Component for integrating equipment monitoring systems."""

    def __init__(self):
        """Initialize the equipment monitoring component."""
        self.logger = setup_logger("equipment_monitoring")

        # Get services and adapters
        self.event_adapter = EventAdapter()

        # Setup monitoring integration
        self._setup_monitoring()

        # Register with interface manager
        interface_manager.register_component("equipment_monitoring", self)

        self.logger.info("Equipment Monitoring Component initialized")

    def _setup_monitoring(self):
        """Setup monitoring integration."""
        # Connect to monitoring systems
        # Define event handlers
        # Initialize data processing

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle monitoring commands."""
        if command == "equipment_status":
            equipment_id = params.get("equipment_id")
            return {
                "handled": True,
                "success": True,
                "status": self._get_equipment_status(equipment_id)
            }
        return {"handled": False}
```

### 15.2 Energy Management Integration

Integrate energy management systems:

```python
class EnergyManagementComponent:
    """Component for energy management integration."""

    def __init__(self):
        """Initialize the energy management component."""
        self.logger = setup_logger("energy_management")

        # Create energy efficiency dashboard
        self._create_energy_dashboard()

        # Register with interface manager
        interface_manager.register_component("energy_management", self)

        self.logger.info("Energy Management Component initialized")

    def _create_energy_dashboard(self):
        """Create energy efficiency dashboard."""
        # Create dashboard layout
        dashboard_manager.create_layout(
            name="energy_dashboard",
            description="Energy efficiency monitoring",
            panels={
                "consumption": {
                    "type": "chart_panel",
                    "title": "Energy Consumption",
                    "position": {"row": 0, "col": 0},
                    "size": {"rows": 1, "cols": 2},
                    "chart_type": "line",
                    "metrics": ["production_line.energy_consumption"]
                },
                "efficiency": {
                    "type": "kpi_panel",
                    "title": "Energy Efficiency",
                    "position": {"row": 1, "col": 0},
                    "size": {"rows": 1, "cols": 1},
                    "metrics": ["energy_efficiency"]
                },
                "controls": {
                    "type": "control_panel",
                    "title": "Energy Controls",
                    "position": {"row": 1, "col": 1},
                    "size": {"rows": 1, "cols": 1}
                }
            }
        )
```

## 16. Troubleshooting

### 16.1 Common Integration Issues

#### Component Registration Issues

**Problem**: Component not receiving events or commands

**Solution**:
```python
# Check if component is registered correctly
if "my_component" in interface_manager.components:
    print("Component registered correctly")
else:
    print("Component not registered")

    # Register manually
    interface_manager.register_component("my_component", my_component)
```

#### Event Handling Issues

**Problem**: Events not being received by component

**Solution**:
```python
# Check event registration
interface_manager.register_event_handler("view_changed", my_component._on_view_changed)

# Test event trigger
interface_manager.trigger_event("view_changed", {"view_name": "test_view"})
```

#### Command Execution Issues

**Problem**: Commands not being processed

**Solution**:
```python
# Debug command processing
result = interface_manager.handle_command("my_command", {"param": "value"})
print(f"Command result: {result}")

# Check if component's handle_command is implemented correctly
if hasattr(my_component, "handle_command"):
    # Test direct execution
    direct_result = my_component.handle_command("my_command", {"param": "value"})
    print(f"Direct result: {direct_result}")
```

### 16.2 Debugging Tools

#### Event Monitoring

```python
def monitor_all_events(event_type, event_data):
    """
    Monitor all events for debugging.

    Args:
        event_type: Event type
        event_data: Event data
    """
    print(f"EVENT: {event_type}")
    print(f"DATA: {event_data}")
    print("-" * 40)

# Register for all events that might be relevant
event_types = [
    "view_changed", "state_changed", "parameter_changed",
    "command_executed", "panel_toggled", "alert_settings_changed",
    "custom_view_saved"
]

for event_type in event_types:
    interface_manager.register_event_handler(event_type,
        lambda data, type=event_type: monitor_all_events(type, data)
    )
```

#### State Inspection

```python
def inspect_interface_state():
    """Inspect the current interface state."""
    print("INTERFACE STATE:")
    print(f"Active View: {interface_state.get_active_view()}")
    print(f"Selected Parameters: {interface_state.get_selected_parameters()}")
    print(f"Expanded Panels: {interface_state.expanded_panels}")
    print(f"Alert Filters: {interface_state.get_alert_filters()}")
    print(f"Parameter Group: {interface_state.get_parameter_group()}")
    print(f"Edit Mode: {interface_state.is_parameter_edit_mode()}")
    print(f"Selected Process: {interface_state.get_selected_process()}")
    print(f"Control Mode: {interface_state.get_process_control_mode()}")
    print("-" * 40)
```

#### Command Tracing

```python
def trace_commands():
    """Trace all command executions."""
    # Patch command service
    original_execute = command_service.execute_command

    def traced_execute(command, params=None):
        print(f"COMMAND: {command}")
        print(f"PARAMS: {params}")

        result = original_execute(command, params)

        print(f"RESULT: {result}")
        print("-" * 40)

        return result

    command_service.execute_command = traced_execute
```

## 17. Frequently Asked Questions

### 17.1 General Integration

**Q: How do I know if my component is properly integrated?**

A: You can verify integration by:
- Checking if events are being received
- Testing command handling
- Verifying your component appears in registered components
- Monitoring system logs for initialization messages

**Q: Can I integrate multiple components at once?**

A: Yes, you can create and register multiple components. Each should have a unique ID and focused responsibility.

### 17.2 Dashboard Integration

**Q: How do I make my panel appear in multiple dashboards?**

A: Create your panel component once, then add it to multiple dashboard layouts:

```python
# Register panel component once
custom_panel = CustomPanelComponent()
dashboard_manager.register_component("custom_panel", custom_panel)

# Add to multiple dashboards
for dashboard_name in ["main_dashboard", "production_dashboard", "custom_dashboard"]:
    layout = dashboard_manager.get_layout(dashboard_name)
    if layout:
        layout.panels[f"custom_panel_{dashboard_name}"] = {
            "type": "custom_panel",
            "title": f"Custom Panel - {dashboard_name}",
            "position": {"row": 1, "col": 0},
            "size": {"rows": 1, "cols": 1}
        }
        dashboard_manager.update_layout(layout)
```

**Q: Can I dynamically update panel content?**

A: Yes, panels are re-rendered when:
- The Digital Twin state changes
- The panel is explicitly refreshed
- The dashboard is refreshed

### 17.3 External System Integration

**Q: How often should I update data from external systems?**

A: Consider these factors:
- Data volatility (how quickly it changes)
- System performance impact
- User experience requirements
- Network and resource constraints

For most scenarios, update intervals between 5 seconds (real-time) and 5 minutes (background) are appropriate.

**Q: How should I handle connection failures?**

A: Implement a robust failure handling strategy:
- Retry with exponential backoff
- Cache last known good data
- Provide clear user feedback
- Log failures for diagnostics
- Automatically reconnect when possible

## 18. Conclusion

This guide has provided comprehensive instructions for integrating with the CIRCMAN5.0 Human-Machine Interface system. By following these patterns and best practices, you can successfully extend the interface with new components, integrate external systems, and customize the user experience.

Remember these key principles:
- Use the component registration system for proper integration
- Leverage the event system for loose coupling
- Follow command patterns for user interactions
- Implement proper thread safety
- Create adapters for external systems
- Test thoroughly before deployment

For additional assistance, refer to:
- The HMI API Reference for detailed method documentation
- The HMI Architecture document for system design understanding
- The Digital Twin API documentation for backend integration
