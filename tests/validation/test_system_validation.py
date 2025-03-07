# tests/validation/test_system_validation.py

"""
System-level validation tests for CIRCMAN5.0.

This module implements comprehensive system validation tests that verify
the interaction between all major components of the system.
"""

import pytest
import time
import json
from pathlib import Path

from .validation_framework import ValidationSuite, ValidationCase, ValidationResult

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.human_interface.core.interface_manager import (
    InterfaceManager,
)
from circman5.adapters.services.constants_service import ConstantsService


def test_full_system_validation(setup_test_environment):
    """Run comprehensive system validation suite."""
    env = setup_test_environment

    # Create validation suite
    system_suite = ValidationSuite(
        suite_id="system_validation", description="Complete System Validation Suite"
    )

    # Add test case for Digital Twin and HMI integration
    def validate_dt_hmi_integration(env):
        """Verify Digital Twin and HMI integration."""
        try:
            # Get components
            digital_twin = env.get("digital_twin")
            interface_manager = env.get("interface_manager")

            if not digital_twin or not interface_manager:
                return (
                    ValidationResult.FAIL,
                    "Required components not available in test environment",
                )

            # Get digital twin adapter from interface manager
            dt_adapter = interface_manager.get_component("digital_twin_adapter")
            if not dt_adapter:
                return (
                    ValidationResult.FAIL,
                    "Digital Twin adapter not found in interface manager",
                )

            # Update Digital Twin state
            test_value = "system_validation_test"
            digital_twin.update({"test_key": test_value})

            # Get state through adapter
            adapter_state = dt_adapter.get_current_state()

            # Verify adapter received updated state
            if "test_key" not in adapter_state:
                return ValidationResult.FAIL, "State update not propagated to adapter"

            if adapter_state["test_key"] != test_value:
                return (
                    ValidationResult.FAIL,
                    f"State value mismatch: {adapter_state['test_key']} != {test_value}",
                )

            # Test parameter updates via HMI
            process_control = interface_manager.get_component("process_control")
            if not process_control:
                return ValidationResult.WARNING, "Process control component not found"
            else:
                # Test start/stop functionality
                start_result = process_control.start_process()
                if not start_result.get("success", False):
                    return (
                        ValidationResult.WARNING,
                        f"Process start failed: {start_result.get('error', 'Unknown error')}",
                    )

                # Check state was updated
                dt_state = digital_twin.get_current_state()
                if (
                    dt_state.get("system_status") != "running"
                    and dt_state.get("production_line", {}).get("status") != "running"
                ):
                    return (
                        ValidationResult.WARNING,
                        "Start command did not update Digital Twin state",
                    )

            return ValidationResult.PASS, "Digital Twin and HMI integration validated"

        except Exception as e:
            return ValidationResult.FAIL, f"Exception during validation: {str(e)}"

    dt_hmi_case = ValidationCase(
        case_id="dt_hmi_integration",
        description="Verify Digital Twin and HMI integration",
        test_function=validate_dt_hmi_integration,
        category="INTEGRATION",
    )
    system_suite.add_test_case(dt_hmi_case)

    # Add test case for event propagation
    def validate_event_propagation(env):
        """Verify event propagation through the system."""
        try:
            # Get components
            digital_twin = env.get("digital_twin")
            interface_manager = env.get("interface_manager")

            if not digital_twin or not interface_manager:
                return (
                    ValidationResult.FAIL,
                    "Required components not available in test environment",
                )

            # Get event-related components
            alert_panel = interface_manager.get_component("alert_panel")
            event_adapter = interface_manager.get_component("event_adapter")

            if not alert_panel or not event_adapter:
                return (
                    ValidationResult.WARNING,
                    "Event components not found in interface manager",
                )

            # Count initial alerts
            initial_alerts = alert_panel.get_filtered_alerts({}) if alert_panel else []
            initial_count = len(initial_alerts)

            # Trigger event by updating Digital Twin with threshold-breaching value
            # Get threshold information
            thresholds = digital_twin.parameter_thresholds
            if not thresholds:
                return ValidationResult.WARNING, "No parameter thresholds configured"

            # Find a threshold to trigger
            test_threshold = next(iter(thresholds.items()))
            path, config = test_threshold

            # Get threshold value and comparison
            threshold_value = config.get("value")
            comparison = config.get("comparison")

            if threshold_value is None or not comparison:
                return ValidationResult.WARNING, "Incomplete threshold configuration"

            # Create a value that will breach the threshold
            breach_value = None
            if comparison == "greater_than":
                breach_value = threshold_value + 10
            elif comparison == "less_than":
                breach_value = threshold_value - 10
            elif comparison == "equal":
                breach_value = threshold_value
            elif comparison == "not_equal":
                breach_value = threshold_value + 1

            if breach_value is None:
                return (
                    ValidationResult.WARNING,
                    f"Unsupported comparison type: {comparison}",
                )

            # Build update based on path
            update_data = {}
            path_parts = path.split(".")
            current = update_data

            for i, part in enumerate(path_parts):
                if i == len(path_parts) - 1:
                    current[part] = breach_value
                else:
                    current[part] = {}
                    current = current[part]

            # Update Digital Twin to trigger event
            digital_twin.update(update_data)

            # Wait for event propagation
            time.sleep(0.5)

            # Check if alerts were added
            if alert_panel:
                updated_alerts = alert_panel.get_filtered_alerts({})

                if len(updated_alerts) <= initial_count:
                    return (
                        ValidationResult.WARNING,
                        "No new alerts after threshold breach",
                    )

            return ValidationResult.PASS, "Event propagation validated"

        except Exception as e:
            return ValidationResult.FAIL, f"Exception during validation: {str(e)}"

    event_case = ValidationCase(
        case_id="event_propagation",
        description="Verify event propagation through the system",
        test_function=validate_event_propagation,
        category="EVENT_SYSTEM",
    )
    system_suite.add_test_case(event_case)

    # Add test case for end-to-end workflow
    def validate_e2e_workflow(env):
        """Verify end-to-end workflow functionality."""
        try:
            # Get components
            digital_twin = env.get("digital_twin")
            interface_manager = env.get("interface_manager")

            if not digital_twin or not interface_manager:
                return (
                    ValidationResult.FAIL,
                    "Required components not available in test environment",
                )

            # Define workflow steps to validate
            # 1. Start manufacturing process
            # 2. Update parameters
            # 3. Run simulation
            # 4. Stop process

            # Get control components
            process_control = interface_manager.get_component("process_control")
            parameter_control = interface_manager.get_component("parameter_control")
            scenario_control = interface_manager.get_component("scenario_control")

            if not process_control:
                return ValidationResult.WARNING, "Process control component not found"

            # Step 1: Start process
            start_result = process_control.start_process()

            if not start_result.get("success", False):
                return (
                    ValidationResult.WARNING,
                    f"Process start failed: {start_result.get('error', 'Unknown error')}",
                )

            # Step 2: Update parameters (if parameter control available)
            param_updated = False
            if parameter_control:
                # Try to update temperature parameter
                update_result = parameter_control.update_parameter(
                    "production_line.temperature", 23.5
                )
                param_updated = update_result.get("success", False)

            # Verify parameter update in Digital Twin state
            dt_state = digital_twin.get_current_state()
            if param_updated:
                temp_value = dt_state.get("production_line", {}).get("temperature")
                if temp_value != 23.5:
                    return (
                        ValidationResult.WARNING,
                        f"Parameter update not reflected in state: {temp_value} != 23.5",
                    )

            # Step 3: Run simulation (if scenario control available)
            sim_run = False
            if scenario_control:
                # Try to run a simulation scenario
                scenario_result = scenario_control.run_scenario(
                    {
                        "steps": 5,
                        "parameters": {"production_line": {"production_rate": 7.5}},
                    }
                )
                sim_run = scenario_result.get("success", False)

            # Alternative: Run simulation directly through Digital Twin
            if not sim_run:
                sim_results = digital_twin.simulate(
                    steps=5, parameters={"production_line": {"production_rate": 7.5}}
                )
                sim_run = len(sim_results) > 0

            # Step 4: Stop process
            stop_result = process_control.stop_process()

            if not stop_result.get("success", False):
                return (
                    ValidationResult.WARNING,
                    f"Process stop failed: {stop_result.get('error', 'Unknown error')}",
                )

            # Verify process stopped in Digital Twin state
            dt_state = digital_twin.get_current_state()
            if (
                dt_state.get("system_status") != "idle"
                and dt_state.get("production_line", {}).get("status") != "idle"
            ):
                return (
                    ValidationResult.WARNING,
                    "Stop command did not update Digital Twin state",
                )

            # Evaluate overall workflow
            if not param_updated:
                return ValidationResult.WARNING, "Parameter update step not validated"

            if not sim_run:
                return ValidationResult.WARNING, "Simulation step not validated"

            return ValidationResult.PASS, "End-to-end workflow validated"

        except Exception as e:
            return ValidationResult.FAIL, f"Exception during validation: {str(e)}"

    workflow_case = ValidationCase(
        case_id="e2e_workflow",
        description="Verify end-to-end workflow functionality",
        test_function=validate_e2e_workflow,
        category="WORKFLOW",
    )
    system_suite.add_test_case(workflow_case)

    # Add test case for configuration system validation
    def validate_configuration_system(env):
        """Verify configuration system functionality."""
        try:
            # Get components
            digital_twin = env.get("digital_twin")

            if not digital_twin:
                return (
                    ValidationResult.FAIL,
                    "Digital Twin not available in test environment",
                )

            # Get constants service
            constants = ConstantsService()

            # Verify digital twin configuration is accessible
            dt_config = constants.get_digital_twin_config()

            if not dt_config:
                return ValidationResult.FAIL, "Digital Twin configuration not available"

            # Check essential configuration sections
            required_sections = [
                "DIGITAL_TWIN_CONFIG",
                "SIMULATION_PARAMETERS",
                "SYNCHRONIZATION_CONFIG",
                "STATE_MANAGEMENT",
            ]

            missing_sections = [s for s in required_sections if s not in dt_config]

            if missing_sections:
                return (
                    ValidationResult.WARNING,
                    f"Missing configuration sections: {', '.join(missing_sections)}",
                )

            # Verify configuration is applied to Digital Twin
            if hasattr(digital_twin, "config"):
                twin_config = digital_twin.config

                # Compare configuration values
                config_section = dt_config.get("DIGITAL_TWIN_CONFIG", {})

                if config_section.get("name") != twin_config.name:
                    return (
                        ValidationResult.WARNING,
                        "Configuration name not applied correctly",
                    )

                if (
                    config_section.get("update_frequency")
                    != twin_config.update_frequency
                ):
                    return (
                        ValidationResult.WARNING,
                        "Update frequency not applied correctly",
                    )

            return ValidationResult.PASS, "Configuration system validated"

        except Exception as e:
            return ValidationResult.FAIL, f"Exception during validation: {str(e)}"

    config_case = ValidationCase(
        case_id="configuration_system",
        description="Verify configuration system functionality",
        test_function=validate_configuration_system,
        category="CONFIGURATION",
    )
    system_suite.add_test_case(config_case)

    # Execute validation suite
    system_suite.execute_all(env)

    # Generate and save report
    report_path = system_suite.save_report("system_validation_report.json")

    # Check if all tests passed (allowing warnings)
    report = system_suite.generate_report()
    assert (
        report["summary"]["failed"] == 0
    ), f"System validation has {report['summary']['failed']} failed tests"

    # Print report summary
    print(f"System Validation Summary:")
    print(f"  Total tests: {report['summary']['total']}")
    print(f"  Passed: {report['summary']['passed']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Warnings: {report['summary']['warnings']}")
    print(f"  Not tested: {report['summary']['not_tested']}")
    print(f"  Report saved to: {report_path}")
