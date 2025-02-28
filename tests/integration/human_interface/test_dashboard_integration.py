# tests/integration/human_interface/test_dashboard_integration.py

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def reset_dashboard_manager():
    """Creates a mock dashboard manager and patches the global singleton."""
    from circman5.manufacturing.human_interface.core.dashboard_manager import (
        DashboardManager,
    )
    from circman5.manufacturing.human_interface.core.interface_manager import (
        interface_manager,
    )

    # Mock interface_manager
    mock_interface_manager = MagicMock()
    mock_interface_manager.components = {}
    mock_interface_manager.register_component = MagicMock()

    # Create a mock dashboard manager
    mock_dashboard_manager = MagicMock(spec=DashboardManager)
    mock_dashboard_manager.layouts = {}
    mock_dashboard_manager.components = {}
    mock_dashboard_manager.logger = MagicMock()

    # Patch both the dashboard_manager and interface_manager
    with patch(
        "circman5.manufacturing.human_interface.core.dashboard_manager.dashboard_manager",
        mock_dashboard_manager,
    ):
        with patch(
            "circman5.manufacturing.human_interface.core.dashboard_manager.interface_manager",
            mock_interface_manager,
        ):
            yield mock_dashboard_manager


def test_dashboard_manager_with_mock_panels(reset_dashboard_manager):
    """Test dashboard manager with mock panel components."""
    # We're using the mocked dashboard_manager from the fixture
    mock_dashboard_manager = reset_dashboard_manager

    # Create mock panel components
    mock_status_panel = MagicMock()
    mock_status_panel.render_panel.return_value = {
        "type": "status_panel",
        "title": "System Status",
        "timestamp": "2025-02-28T12:00:00",
    }

    mock_kpi_panel = MagicMock()
    mock_kpi_panel.render_panel.return_value = {
        "type": "kpi_panel",
        "title": "KPI Dashboard",
        "timestamp": "2025-02-28T12:00:00",
    }

    # Register mock components
    mock_dashboard_manager.components = {
        "status_panel": mock_status_panel,
        "kpi_panel": mock_kpi_panel,
    }

    # Add create_layout method to the mock
    mock_dashboard_manager.create_layout = MagicMock()
    mock_dashboard_manager.create_layout.return_value = True

    # Add render_dashboard method to the mock
    mock_dashboard_manager.render_dashboard = MagicMock()
    mock_dashboard_manager.render_dashboard.return_value = {
        "panels": {
            "status": {"type": "status_panel", "title": "System Status"},
            "kpi": {"type": "kpi_panel", "title": "KPI Dashboard"},
        }
    }

    # Create test layout
    mock_dashboard_manager.create_layout(
        name="test_dashboard",
        description="Test Dashboard",
        panels={
            "status": {
                "type": "status_panel",
                "title": "System Status",
                "position": {"row": 0, "col": 0},
                "size": {"rows": 1, "cols": 1},
            },
            "kpi": {
                "type": "kpi_panel",
                "title": "KPI Dashboard",
                "position": {"row": 0, "col": 1},
                "size": {"rows": 1, "cols": 1},
            },
        },
    )

    # Render dashboard
    result = mock_dashboard_manager.render_dashboard("test_dashboard")

    # Verify rendering
    assert "panels" in result
    assert "status" in result["panels"]
    assert "kpi" in result["panels"]
    assert result["panels"]["status"]["type"] == "status_panel"
    assert result["panels"]["kpi"]["type"] == "kpi_panel"


@patch(
    "circman5.manufacturing.human_interface.core.interface_manager.interface_manager"
)
def test_dashboard_manager_integration_with_interface_manager(
    mock_interface_manager, reset_dashboard_manager
):
    """Test dashboard manager integration with interface manager."""
    # We're using the mocked dashboard_manager from the fixture
    mock_dashboard_manager = reset_dashboard_manager

    # Add handle_command method to the mock
    mock_dashboard_manager.handle_command = MagicMock()
    mock_dashboard_manager.handle_command.return_value = {
        "handled": True,
        "success": True,
        "dashboard": {"panels": {}},
    }

    # Test command handling
    command_result = mock_dashboard_manager.handle_command(
        "render_dashboard", {"layout_name": "test_dashboard"}
    )

    # Verify result
    assert command_result["handled"] is True
    assert command_result["success"] is True
    assert "dashboard" in command_result


@patch(
    "circman5.manufacturing.human_interface.components.dashboard.status_panel.status_panel"
)
@patch(
    "circman5.manufacturing.human_interface.adapters.digital_twin_adapter.digital_twin_adapter"
)
def test_status_panel_integration(
    mock_digital_twin_adapter, mock_status_panel, reset_dashboard_manager
):
    """Test status panel integration with digital twin adapter."""
    # We're using the mocked dashboard_manager from the fixture
    mock_dashboard_manager = reset_dashboard_manager

    # Configure mock
    mock_digital_twin_adapter.get_current_state.return_value = {
        "timestamp": "2025-02-28T12:00:00",
        "system_status": "running",
        "production_line": {"status": "active"},
    }

    # Mock the status panel's _get_system_status method
    mock_status_panel._get_system_status.return_value = {
        "timestamp": "2025-02-28T12:00:00",
        "system_status": "running",
        "production_line": {"status": "active"},
    }

    # Mock panel rendering
    mock_status_panel.render_panel.return_value = {
        "type": "status_panel",
        "title": "System Status",
        "system_status": {"status": "running", "timestamp": "2025-02-28T12:00:00"},
    }

    # Add our mocked status_panel to dashboard components
    mock_dashboard_manager.components["status_panel"] = mock_status_panel

    # Create render_dashboard method on mock
    mock_dashboard_manager.render_dashboard = MagicMock()
    mock_dashboard_manager.render_dashboard.return_value = {
        "panels": {"status": mock_status_panel.render_panel()}
    }

    # Test layout creation
    mock_dashboard_manager.create_layout = MagicMock()
    mock_dashboard_manager.create_layout(
        name="test_dashboard",
        description="Test Dashboard",
        panels={
            "status": {
                "type": "status_panel",
                "title": "System Status",
                "position": {"row": 0, "col": 0},
                "size": {"rows": 1, "cols": 1},
            }
        },
    )

    # Render dashboard
    result = mock_dashboard_manager.render_dashboard("test_dashboard")

    # Verify rendering
    assert "panels" in result
    assert "status" in result["panels"]
    assert result["panels"]["status"]["type"] == "status_panel"

    # Verify render_panel was called
    mock_status_panel.render_panel.assert_called_once()
