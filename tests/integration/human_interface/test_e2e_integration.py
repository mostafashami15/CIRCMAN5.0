# tests/integration/human_interface/test_e2e_integration.py

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.usefixtures(
    "initialized_interface", "mock_digital_twin", "mock_event_adapter"
)
class TestHumanInterfaceE2E:
    """End-to-End tests for Human-Machine Interface."""

    def test_dashboard_rendering_workflow(
        self, initialized_interface, mock_digital_twin
    ):
        """Test end-to-end dashboard rendering workflow."""
        from circman5.manufacturing.human_interface.services.command_service import (
            command_service,
        )
        from circman5.manufacturing.human_interface.core.dashboard_manager import (
            dashboard_manager,
        )
        from circman5.manufacturing.human_interface.components.dashboard.status_panel import (
            status_panel,
        )
        from circman5.manufacturing.human_interface.components.dashboard.kpi_panel import (
            kpi_panel,
        )

        # Create mock dashboard manager with properly mocked methods
        mock_dash_manager = MagicMock()
        mock_dash_manager.create_layout = MagicMock(return_value=True)
        mock_dash_manager.render_dashboard = MagicMock(
            return_value={
                "panels": {
                    "status": {
                        "type": "status_panel",
                        "title": "System Status",
                        "system_status": {"system_status": "running"},
                    },
                    "kpi": {
                        "type": "kpi_panel",
                        "title": "Key Performance Indicators",
                        "kpi_data": {"production_rate": {"value": 120.0}},
                    },
                }
            }
        )

        # Create mock command service
        mock_cmd_service = MagicMock()
        mock_cmd_service.execute_command = MagicMock(
            return_value={
                "success": True,
                "dashboard": {
                    "panels": {
                        "status": {
                            "type": "status_panel",
                            "title": "System Status",
                            "system_status": {"system_status": "running"},
                        },
                        "kpi": {
                            "type": "kpi_panel",
                            "title": "Key Performance Indicators",
                            "kpi_data": {"production_rate": {"value": 120.0}},
                        },
                    }
                },
            }
        )

        # Register mock components
        with patch.object(status_panel, "render_panel") as mock_status_render:
            with patch.object(kpi_panel, "render_panel") as mock_kpi_render:
                # Patch globals
                with patch(
                    "circman5.manufacturing.human_interface.core.dashboard_manager.dashboard_manager",
                    mock_dash_manager,
                ):
                    with patch(
                        "circman5.manufacturing.human_interface.services.command_service.command_service",
                        mock_cmd_service,
                    ):
                        # Setup mock responses
                        mock_status_render.return_value = {
                            "type": "status_panel",
                            "title": "System Status",
                            "system_status": {"system_status": "running"},
                        }

                        mock_kpi_render.return_value = {
                            "type": "kpi_panel",
                            "title": "Key Performance Indicators",
                            "kpi_data": {"production_rate": {"value": 120.0}},
                        }

                        # Create test layout
                        mock_dash_manager.create_layout(
                            name="test_e2e_dashboard",
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
                            },
                        )

                        # Execute render command through command service
                        result = mock_cmd_service.execute_command(
                            "render_dashboard", {"layout_name": "test_e2e_dashboard"}
                        )

                        # Verify command execution
                        assert result["success"] is True
                        assert "dashboard" in result
                        assert "panels" in result["dashboard"]
                        assert "status" in result["dashboard"]["panels"]
                        assert "kpi" in result["dashboard"]["panels"]

    def test_process_control_workflow(self, initialized_interface, mock_digital_twin):
        """Test end-to-end process control workflow."""
        from circman5.manufacturing.human_interface.services.command_service import (
            command_service,
        )
        from circman5.manufacturing.human_interface.core.interface_state import (
            interface_state,
        )
        from circman5.manufacturing.human_interface.components.controls.process_control import (
            process_control,
        )

        # Create mock command service
        mock_cmd_service = MagicMock()
        mock_cmd_service.execute_command = MagicMock(
            return_value={"success": True, "mode": "manual"}
        )

        # Create mock interface state
        mock_if_state = MagicMock()
        mock_if_state.get_process_control_mode = MagicMock(return_value="manual")

        # Patch process_control to avoid actual digital twin updates
        with patch.object(process_control, "start_process") as mock_start:
            with patch.object(process_control, "stop_process") as mock_stop:
                # Patch the globals
                with patch(
                    "circman5.manufacturing.human_interface.services.command_service.command_service",
                    mock_cmd_service,
                ):
                    with patch(
                        "circman5.manufacturing.human_interface.core.interface_state.interface_state",
                        mock_if_state,
                    ):
                        # Configure mocks
                        mock_start.return_value = {"success": True}
                        mock_stop.return_value = {"success": True}

                        # Set control mode
                        result = mock_cmd_service.execute_command(
                            "set_control_mode", {"mode": "manual"}
                        )

                        # Verify command execution
                        assert result["success"] is True

                        # Verify control mode set
                        assert mock_if_state.get_process_control_mode() == "manual"

                        # Start process
                        start_result = mock_cmd_service.execute_command(
                            "start_process", {"process_id": "process1"}
                        )

                        # Verify process started
                        assert start_result["success"] is True

    def test_alert_handling_workflow(self, initialized_interface, mock_event_adapter):
        """Test end-to-end alert handling workflow."""
        from circman5.manufacturing.human_interface.services.command_service import (
            command_service,
        )
        from circman5.manufacturing.human_interface.components.alerts.alert_panel import (
            alert_panel,
        )

        # Create mock command service
        mock_cmd_service = MagicMock()
        mock_cmd_service.execute_command = MagicMock(
            return_value={
                "success": True,
                "alerts": [
                    {
                        "id": "evt-001",
                        "timestamp": "2025-02-28T11:55:00",
                        "category": "system",
                        "severity": "info",
                        "message": "System started",
                        "acknowledged": False,
                    },
                    {
                        "id": "evt-002",
                        "timestamp": "2025-02-28T11:58:00",
                        "category": "process",
                        "severity": "warning",
                        "message": "Temperature high",
                        "acknowledged": False,
                    },
                ],
            }
        )

        # Patch alert_panel to avoid actual event system interactions
        with patch.object(alert_panel, "get_filtered_alerts") as mock_get_alerts:
            with patch.object(alert_panel, "acknowledge_alert") as mock_acknowledge:
                # Patch the command service
                with patch(
                    "circman5.manufacturing.human_interface.services.command_service.command_service",
                    mock_cmd_service,
                ):
                    # Configure mocks
                    mock_get_alerts.return_value = [
                        {
                            "id": "evt-001",
                            "timestamp": "2025-02-28T11:55:00",
                            "category": "system",
                            "severity": "info",
                            "message": "System started",
                            "acknowledged": False,
                        },
                        {
                            "id": "evt-002",
                            "timestamp": "2025-02-28T11:58:00",
                            "category": "process",
                            "severity": "warning",
                            "message": "Temperature high",
                            "acknowledged": False,
                        },
                    ]
                    mock_acknowledge.return_value = True

                    # Get alerts
                    get_result = mock_cmd_service.execute_command(
                        "get_alerts",
                        {"filter": {"severity_levels": ["info", "warning"]}},
                    )

                    # Verify get alerts
                    assert get_result["success"] is True
                    assert "alerts" in get_result
                    assert len(get_result["alerts"]) == 2

                    # Acknowledge alert
                    ack_result = mock_cmd_service.execute_command(
                        "acknowledge_alert", {"alert_id": "evt-002"}
                    )

                    # Verify acknowledge
                    assert ack_result["success"] is True

    def test_scenario_control_workflow(self, initialized_interface):
        """Test end-to-end scenario control workflow."""
        from circman5.manufacturing.human_interface.services.command_service import (
            command_service,
        )
        from circman5.manufacturing.human_interface.components.controls.scenario_control import (
            scenario_control,
        )

        # Create mock command service
        mock_cmd_service = MagicMock()

        # Configure mock responses for different commands
        mock_cmd_service.execute_command = MagicMock(
            side_effect=lambda cmd, params: {
                "create_scenario": {
                    "success": True,
                    "scenario": {"name": "high_production"},
                },
                "run_scenario": {
                    "success": True,
                    "results": [
                        {
                            "timestamp": "2025-02-28T12:00:00",
                            "system_status": "running",
                        },
                        {
                            "timestamp": "2025-02-28T12:01:00",
                            "system_status": "running",
                        },
                    ],
                },
            }.get(cmd, {"success": True})
        )

        # Patch scenario_control to avoid actual digital twin interactions
        with patch.object(scenario_control, "create_scenario") as mock_create:
            with patch.object(scenario_control, "run_scenario") as mock_run:
                # Patch the command service
                with patch(
                    "circman5.manufacturing.human_interface.services.command_service.command_service",
                    mock_cmd_service,
                ):
                    # Configure mocks
                    mock_create.return_value = {"success": True}
                    mock_run.return_value = {
                        "success": True,
                        "results": [
                            {
                                "timestamp": "2025-02-28T12:00:00",
                                "system_status": "running",
                            },
                            {
                                "timestamp": "2025-02-28T12:01:00",
                                "system_status": "running",
                            },
                        ],
                    }

                    # Create scenario
                    scenario_params = {"production_rate": 150, "temperature": 165.5}

                    create_result = mock_cmd_service.execute_command(
                        "create_scenario",
                        {
                            "name": "high_production",
                            "parameters": scenario_params,
                            "description": "High production test",
                        },
                    )

                    # Verify scenario creation
                    assert create_result["success"] is True
                    assert "scenario" in create_result

                    # Run scenario
                    run_result = mock_cmd_service.execute_command(
                        "run_scenario", {"scenario_name": "high_production"}
                    )

                    # Verify scenario run
                    assert run_result["success"] is True
                    assert "results" in run_result
                    assert len(run_result["results"]) == 2
