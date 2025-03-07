# tests/validation/test_hmi_validation.py

import pytest
from .validation_framework import ValidationSuite, ValidationCase, ValidationResult

import pytest
from .validation_framework import ValidationSuite, ValidationCase, ValidationResult


@pytest.mark.xfail(
    reason="Mock environment missing dashboard_manager and alert_panel components"
)
def test_hmi_validation_suite(setup_test_environment):
    """Run the HMI validation suite to verify system requirements."""
    env = setup_test_environment

    # Create validation suite
    hmi_suite = ValidationSuite(
        suite_id="hmi_validation",
        description="Human-Machine Interface Validation Suite",
    )

    # Add dashboard validation test case
    def validate_dashboard_components(env):
        """Verify all required dashboard components are present and functional."""
        try:
            interface_manager = env["interface_manager"]
            dashboard_manager = interface_manager.get_component("dashboard_manager")

            # Check if required components are registered
            required_components = [
                "status_panel",
                "kpi_panel",
                "process_panel",
                "alert_panel",
                "parameter_control",
                "process_control",
            ]

            missing_components = []
            for component in required_components:
                try:
                    interface_manager.get_component(component)
                except KeyError:
                    missing_components.append(component)

            if missing_components:
                return (
                    ValidationResult.FAIL,
                    f"Missing required components: {', '.join(missing_components)}",
                )

            # Render dashboard and check panels
            dashboard_data = dashboard_manager.render_dashboard("main_dashboard")

            if "panels" not in dashboard_data:
                return ValidationResult.FAIL, "Dashboard data missing panels section"

            panels = dashboard_data["panels"]

            # Verify key panels exist
            required_panels = ["status", "kpi", "process"]
            missing_panels = [panel for panel in required_panels if panel not in panels]

            if missing_panels:
                return (
                    ValidationResult.FAIL,
                    f"Dashboard missing required panels: {', '.join(missing_panels)}",
                )

            return (
                ValidationResult.PASS,
                "All dashboard components present and functional",
            )

        except Exception as e:
            return ValidationResult.FAIL, f"Exception during validation: {str(e)}"

    dashboard_case = ValidationCase(
        case_id="dashboard_components",
        description="Verify all required dashboard components are present",
        test_function=validate_dashboard_components,
        category="UI_COMPONENTS",
    )
    hmi_suite.add_test_case(dashboard_case)

    # Add event notification validation test case
    def validate_event_notification(env):
        """Verify event notification system is properly integrated with HMI."""
        try:
            from circman5.manufacturing.digital_twin.event_notification.event_manager import (
                event_manager,
            )
            from circman5.manufacturing.digital_twin.event_notification.event_types import (
                Event,
                EventCategory,
                EventSeverity,
            )

            interface_manager = env["interface_manager"]
            alert_panel = interface_manager.get_component("alert_panel")

            # Create and publish test event
            test_event = Event(
                category=EventCategory.SYSTEM,
                severity=EventSeverity.INFO,
                message="Validation test event",
                source="validation_test",
                details={"validation": True},
            )

            # Get initial alert count
            initial_alerts = alert_panel.get_filtered_alerts({})
            initial_count = len(initial_alerts)

            # Publish event
            event_manager.publish(test_event)

            # Wait for event propagation
            import time

            time.sleep(0.5)

            # Get updated alerts
            updated_alerts = alert_panel.get_filtered_alerts({})

            if len(updated_alerts) <= initial_count:
                return (
                    ValidationResult.FAIL,
                    "Event notification not propagated to alert panel",
                )

            # Verify event content
            new_alerts = [
                a for a in updated_alerts if a.get("message") == "Validation test event"
            ]
            if not new_alerts:
                return (
                    ValidationResult.FAIL,
                    "Event received but message content incorrect",
                )

            return (
                ValidationResult.PASS,
                "Event notification system properly integrated",
            )

        except Exception as e:
            return ValidationResult.FAIL, f"Exception during validation: {str(e)}"

    event_case = ValidationCase(
        case_id="event_notification",
        description="Verify event notification system is properly integrated",
        test_function=validate_event_notification,
        category="EVENT_SYSTEM",
    )
    hmi_suite.add_test_case(event_case)

    # Add control validation test case
    def validate_process_control(env):
        """Verify process control components function correctly."""
        try:
            interface_manager = env["interface_manager"]
            digital_twin = env["digital_twin"]
            process_control = interface_manager.get_component("process_control")

            # Set initial state
            digital_twin.update(
                {"system_status": "idle", "production_line": {"status": "idle"}}
            )

            # Use process control to start process
            result = process_control.start_process()

            if not result.get("success", False):
                return (
                    ValidationResult.FAIL,
                    f"Process start command failed: {result.get('error', 'Unknown error')}",
                )

            # Check digital twin state
            current_state = digital_twin.get_current_state()

            if current_state.get("system_status") != "running":
                return (
                    ValidationResult.FAIL,
                    f"System status not updated to running, current: {current_state.get('system_status')}",
                )

            if not current_state.get("production_line", {}).get("status") == "running":
                return (
                    ValidationResult.FAIL,
                    f"Production line status not updated to running",
                )

            # Use process control to stop process
            result = process_control.stop_process()

            if not result.get("success", False):
                return (
                    ValidationResult.FAIL,
                    f"Process stop command failed: {result.get('error', 'Unknown error')}",
                )

            # Check digital twin state again
            current_state = digital_twin.get_current_state()

            if current_state.get("system_status") != "idle":
                return (
                    ValidationResult.FAIL,
                    f"System status not updated to idle, current: {current_state.get('system_status')}",
                )

            return (
                ValidationResult.PASS,
                "Process control components function correctly",
            )

        except Exception as e:
            return ValidationResult.FAIL, f"Exception during validation: {str(e)}"

    control_case = ValidationCase(
        case_id="process_control",
        description="Verify process control components function correctly",
        test_function=validate_process_control,
        category="CONTROLS",
    )
    hmi_suite.add_test_case(control_case)

    # Execute validation suite
    hmi_suite.execute_all(env)

    # Generate and save report
    report_path = hmi_suite.save_report()

    # Check if all tests passed
    report = hmi_suite.generate_report()
    assert (
        report["summary"]["failed"] == 0
    ), f"Validation suite has {report['summary']['failed']} failed tests"
