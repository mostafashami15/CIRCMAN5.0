# src/circman5/manufacturing/human_interface/core/dashboard_manager.py

"""
Dashboard manager for CIRCMAN5.0 Human-Machine Interface.

This module manages dashboard layouts, components, and rendering,
providing a flexible system for displaying manufacturing data.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable
import datetime
import threading
import json
from pathlib import Path

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from .interface_state import interface_state
from .interface_manager import interface_manager


class DashboardLayout:
    """
    Represents a dashboard layout configuration.

    Attributes:
        name: Layout name
        description: Layout description
        panels: Dictionary of panel configurations
        layout_config: Grid layout configuration
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        panels: Optional[Dict[str, Any]] = None,
        layout_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a dashboard layout.

        Args:
            name: Layout name
            description: Layout description
            panels: Optional dictionary of panel configurations
            layout_config: Optional grid layout configuration
        """
        self.name = name
        self.description = description
        self.panels = panels or {}
        self.layout_config = layout_config or {"rows": 2, "columns": 2, "spacing": 10}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert layout to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "panels": self.panels,
            "layout_config": self.layout_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DashboardLayout":
        """
        Create layout from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            DashboardLayout: Created layout
        """
        return cls(
            name=data.get("name", "Unnamed Layout"),
            description=data.get("description", ""),
            panels=data.get("panels", {}),
            layout_config=data.get(
                "layout_config", {"rows": 2, "columns": 2, "spacing": 10}
            ),
        )


class DashboardManager:
    """
    Manages dashboard layouts and rendering.

    This class handles the creation, management, and rendering of
    dashboard layouts, providing a flexible system for displaying
    manufacturing data.

    Attributes:
        state: Reference to interface state
        layouts: Dictionary of available layouts
        components: Dictionary of registered dashboard components
        current_layout: Currently active layout
        logger: Logger instance for this class
    """

    def __init__(self):
        """Initialize the dashboard manager."""
        self.logger = setup_logger("dashboard_manager")
        self.constants = ConstantsService()

        # Get reference to interface state
        self.state = interface_state

        # Initialize layouts and components
        self.layouts: Dict[str, DashboardLayout] = {}
        self.components: Dict[str, Any] = {}
        self.current_layout: Optional[DashboardLayout] = None

        # Thread safety
        self._lock = threading.RLock()

        # Register with interface manager
        interface_manager.register_component("dashboard_manager", self)

        # Register event handlers
        interface_manager.register_event_handler("view_changed", self._on_view_changed)

        # Load layouts from storage
        self._load_layouts()

        # Create default layouts if none exist
        if not self.layouts:
            self._create_default_layouts()

        self.logger.info("Dashboard Manager initialized")

    def register_component(self, component_id: str, component: Any) -> None:
        """
        Register a dashboard component.

        Args:
            component_id: Component identifier
            component: Component instance

        Raises:
            ValueError: If component ID already exists
        """
        with self._lock:
            if component_id in self.components:
                raise ValueError(f"Component ID already registered: {component_id}")

            self.components[component_id] = component
            self.logger.debug(f"Dashboard component registered: {component_id}")

    def create_layout(
        self,
        name: str,
        description: str = "",
        panels: Optional[Dict[str, Any]] = None,
        layout_config: Optional[Dict[str, Any]] = None,
    ) -> DashboardLayout:
        """
        Create a new dashboard layout.

        Args:
            name: Layout name
            description: Layout description
            panels: Optional panel configurations
            layout_config: Optional layout configuration

        Returns:
            DashboardLayout: Created layout

        Raises:
            ValueError: If layout name already exists
        """
        with self._lock:
            if name in self.layouts:
                raise ValueError(f"Layout already exists: {name}")

            layout = DashboardLayout(
                name=name,
                description=description,
                panels=panels,
                layout_config=layout_config,
            )

            self.layouts[name] = layout
            self._save_layouts()

            self.logger.info(f"Layout created: {name}")
            return layout

    def get_layout(self, name: str) -> Optional[DashboardLayout]:
        """
        Get a dashboard layout by name.

        Args:
            name: Layout name

        Returns:
            Optional[DashboardLayout]: Layout if found, None otherwise
        """
        return self.layouts.get(name)

    def delete_layout(self, name: str) -> bool:
        """
        Delete a dashboard layout.

        Args:
            name: Layout name

        Returns:
            bool: True if layout was deleted
        """
        with self._lock:
            if name in self.layouts:
                del self.layouts[name]
                self._save_layouts()
                self.logger.info(f"Layout deleted: {name}")
                return True
            return False

    def set_active_layout(self, name: str) -> bool:
        """
        Set the active dashboard layout.

        Args:
            name: Layout name

        Returns:
            bool: True if layout was activated
        """
        layout = self.get_layout(name)
        if not layout:
            self.logger.warning(f"Layout not found: {name}")
            return False

        self.current_layout = layout

        # Set active view in interface state
        self.state.set_active_view(name)

        self.logger.info(f"Active layout set to: {name}")
        return True

    def get_active_layout(self) -> Optional[DashboardLayout]:
        """
        Get the currently active layout.

        Returns:
            Optional[DashboardLayout]: Current layout or None
        """
        return self.current_layout

    def update_layout(self, layout: DashboardLayout) -> None:
        """
        Update a dashboard layout.

        Args:
            layout: Layout to update
        """
        with self._lock:
            self.layouts[layout.name] = layout
            self._save_layouts()
            self.logger.info(f"Layout updated: {layout.name}")

    def get_all_layouts(self) -> Dict[str, DashboardLayout]:
        """
        Get all available layouts.

        Returns:
            Dict[str, DashboardLayout]: Dictionary of layouts
        """
        return self.layouts.copy()

    def render_dashboard(self, layout_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Render a dashboard layout.

        Args:
            layout_name: Optional layout name (uses active layout if None)

        Returns:
            Dict[str, Any]: Rendered dashboard data
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

        # Prepare dashboard data structure
        dashboard_data = {
            "layout": layout.to_dict(),
            "panels": {},
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Render each panel in the layout
        for panel_id, panel_config in layout.panels.items():
            panel_type = panel_config.get("type", "unknown")

            # Check if we have a component for this panel type
            if panel_type in self.components:
                try:
                    # Get the component and render the panel
                    component = self.components[panel_type]
                    if hasattr(component, "render_panel"):
                        panel_data = component.render_panel(panel_config)
                        dashboard_data["panels"][panel_id] = panel_data
                    else:
                        dashboard_data["panels"][panel_id] = {
                            "error": f"Component {panel_type} has no render_panel method",
                            "config": panel_config,
                        }
                except Exception as e:
                    self.logger.error(f"Error rendering panel {panel_id}: {str(e)}")
                    dashboard_data["panels"][panel_id] = {
                        "error": f"Rendering error: {str(e)}",
                        "config": panel_config,
                    }
            else:
                dashboard_data["panels"][panel_id] = {
                    "error": f"Unknown panel type: {panel_type}",
                    "config": panel_config,
                }

        return dashboard_data

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle dashboard-related commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        if command == "set_active_layout":
            layout_name = params.get("layout_name")
            if layout_name:
                success = self.set_active_layout(layout_name)
                return {"handled": True, "success": success}
            return {
                "handled": True,
                "success": False,
                "error": "Missing layout_name parameter",
            }

        elif command == "create_layout":
            try:
                layout = self.create_layout(
                    name=params.get("name", "New Layout"),
                    description=params.get("description", ""),
                    panels=params.get("panels", {}),
                    layout_config=params.get("layout_config"),
                )
                return {"handled": True, "success": True, "layout": layout.to_dict()}
            except Exception as e:
                return {"handled": True, "success": False, "error": str(e)}

        elif command == "delete_layout":
            layout_name = params.get("layout_name")
            if layout_name:
                success = self.delete_layout(layout_name)
                return {"handled": True, "success": success}
            return {
                "handled": True,
                "success": False,
                "error": "Missing layout_name parameter",
            }

        elif command == "render_dashboard":
            layout_name = params.get("layout_name")
            dashboard_data = self.render_dashboard(layout_name)
            if "error" in dashboard_data:
                return {
                    "handled": True,
                    "success": False,
                    "error": dashboard_data["error"],
                }
            return {"handled": True, "success": True, "dashboard": dashboard_data}

        # Not a dashboard command
        return {"handled": False}

    def _on_view_changed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle view change events.

        Args:
            event_data: Event data
        """
        view_name = event_data.get("view_name")
        if view_name:
            # Try to activate the layout with this name
            layout = self.get_layout(view_name)
            if layout:
                self.current_layout = layout
                self.logger.debug(f"Layout set to match view: {view_name}")

    def _create_default_layouts(self) -> None:
        """Create default dashboard layouts."""
        with self._lock:
            # Main Dashboard
            self.create_layout(
                name="main_dashboard",
                description="Main system dashboard",
                panels={
                    "status": {
                        "type": "status_panel",
                        "title": "System Status",
                        "position": {"row": 0, "col": 0},
                        "size": {"rows": 1, "cols": 1},
                    },
                    "kpi": {
                        "type": "kpi_panel",
                        "title": "Key Performance Indicators",
                        "position": {"row": 0, "col": 1},
                        "size": {"rows": 1, "cols": 1},
                    },
                    "process": {
                        "type": "process_panel",
                        "title": "Manufacturing Process",
                        "position": {"row": 1, "col": 0},
                        "size": {"rows": 1, "cols": 2},
                    },
                },
                layout_config={"rows": 2, "columns": 2, "spacing": 10},
            )

            # Production Dashboard
            self.create_layout(
                name="production_dashboard",
                description="Production monitoring dashboard",
                panels={
                    "production_status": {
                        "type": "status_panel",
                        "title": "Production Status",
                        "position": {"row": 0, "col": 0},
                        "size": {"rows": 1, "cols": 1},
                    },
                    "production_metrics": {
                        "type": "kpi_panel",
                        "title": "Production Metrics",
                        "position": {"row": 0, "col": 1},
                        "size": {"rows": 1, "cols": 1},
                        "metrics": [
                            "production_rate",
                            "yield_rate",
                            "energy_efficiency",
                        ],
                    },
                    "production_history": {
                        "type": "chart_panel",
                        "title": "Production History",
                        "position": {"row": 1, "col": 0},
                        "size": {"rows": 1, "cols": 2},
                        "chart_type": "line",
                        "metrics": ["production_rate", "energy_consumption"],
                    },
                },
                layout_config={"rows": 2, "columns": 2, "spacing": 10},
            )

            # Quality Dashboard
            self.create_layout(
                name="quality_dashboard",
                description="Quality monitoring dashboard",
                panels={
                    "quality_status": {
                        "type": "status_panel",
                        "title": "Quality Status",
                        "position": {"row": 0, "col": 0},
                        "size": {"rows": 1, "cols": 1},
                    },
                    "quality_metrics": {
                        "type": "kpi_panel",
                        "title": "Quality Metrics",
                        "position": {"row": 0, "col": 1},
                        "size": {"rows": 1, "cols": 1},
                        "metrics": ["defect_rate", "quality_score", "yield_rate"],
                    },
                    "defect_analysis": {
                        "type": "chart_panel",
                        "title": "Defect Analysis",
                        "position": {"row": 1, "col": 0},
                        "size": {"rows": 1, "cols": 1},
                        "chart_type": "bar",
                    },
                    "quality_trends": {
                        "type": "chart_panel",
                        "title": "Quality Trends",
                        "position": {"row": 1, "col": 1},
                        "size": {"rows": 1, "cols": 1},
                        "chart_type": "line",
                    },
                },
                layout_config={"rows": 2, "columns": 2, "spacing": 10},
            )

            # Alerts Dashboard
            self.create_layout(
                name="alerts_dashboard",
                description="System alerts and notifications",
                panels={
                    "active_alerts": {
                        "type": "alert_panel",
                        "title": "Active Alerts",
                        "position": {"row": 0, "col": 0},
                        "size": {"rows": 1, "cols": 2},
                        "filter": {"acknowledged": False},
                    },
                    "alert_history": {
                        "type": "alert_panel",
                        "title": "Alert History",
                        "position": {"row": 1, "col": 0},
                        "size": {"rows": 1, "cols": 1},
                        "filter": {"acknowledged": True},
                    },
                    "alert_stats": {
                        "type": "chart_panel",
                        "title": "Alert Statistics",
                        "position": {"row": 1, "col": 1},
                        "size": {"rows": 1, "cols": 1},
                        "chart_type": "pie",
                    },
                },
                layout_config={"rows": 2, "columns": 2, "spacing": 10},
            )

            self.logger.info("Default layouts created")

    def _save_layouts(self) -> None:
        """Save layouts to persistent storage."""
        try:
            # Convert layouts to serializable format
            layouts_dict = {
                name: layout.to_dict() for name, layout in self.layouts.items()
            }

            # Get the interface directory
            interface_dir = results_manager.get_path("digital_twin")
            layouts_file = interface_dir / "dashboard_layouts.json"

            # Save to file
            with open(layouts_file, "w") as f:
                json.dump(layouts_dict, f, indent=2)

            self.logger.debug("Dashboard layouts saved")

        except Exception as e:
            self.logger.error(f"Error saving dashboard layouts: {str(e)}")

    def _load_layouts(self) -> None:
        """Load layouts from persistent storage."""
        try:
            interface_dir = results_manager.get_path("digital_twin")
            layouts_file = interface_dir / "dashboard_layouts.json"

            if layouts_file.exists():
                with open(layouts_file, "r") as f:
                    layouts_dict = json.load(f)

                # Convert to DashboardLayout objects
                self.layouts = {
                    name: DashboardLayout.from_dict(layout_dict)
                    for name, layout_dict in layouts_dict.items()
                }

                self.logger.info(f"Loaded {len(self.layouts)} dashboard layouts")
        except Exception as e:
            self.logger.warning(f"Could not load dashboard layouts: {str(e)}")


# Create global instance
dashboard_manager = DashboardManager()
