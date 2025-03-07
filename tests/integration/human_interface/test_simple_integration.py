# tests/integration/human_interface/test_simple_integration.py

import pytest


def test_interface_component_registration(setup_test_environment):
    """Test that components are properly registered."""
    env = setup_test_environment
    interface_manager = env["interface_manager"]

    # Verify required components exist
    assert "process_control" in interface_manager.components
    assert "command_service" in interface_manager.components
    assert "dashboard_manager" in interface_manager.components

    # Verify we can retrieve components
    process_control = interface_manager.get_component("process_control")
    assert process_control is not None

    command_service = interface_manager.get_component("command_service")
    assert command_service is not None

    dashboard_manager = interface_manager.get_component("dashboard_manager")
    assert dashboard_manager is not None

    # Test basic functionality
    dashboard = dashboard_manager.render_dashboard("main_dashboard")
    assert "panels" in dashboard
    assert "status" in dashboard["panels"]
    assert "kpi" in dashboard["panels"]
    assert "process" in dashboard["panels"]
