# tests/validation/validation_framework.py

from typing import Dict, Any, List, Optional, Callable
import datetime
import json
from pathlib import Path
from enum import Enum
from circman5.utils.results_manager import results_manager
from circman5.utils.logging_config import setup_logger


class ValidationResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_TESTED = "NOT_TESTED"


class ValidationCase:
    """Represents a validation test case for system verification."""

    def __init__(
        self,
        case_id: str,
        description: str,
        test_function: Callable,
        category: str,
        severity: str = "CRITICAL",
    ):
        self.case_id = case_id
        self.description = description
        self.test_function = test_function
        self.category = category
        self.severity = severity
        self.result = ValidationResult.NOT_TESTED
        self.message = ""
        self.execution_time = 0.0
        self.timestamp = None

    def execute(self, *args, **kwargs) -> ValidationResult:
        """Execute the validation test case."""
        start_time = datetime.datetime.now()

        try:
            # Run the test function
            result = self.test_function(*args, **kwargs)

            # Process result
            if isinstance(result, tuple) and len(result) >= 2:
                # If function returns (ValidationResult.XXXX, "message")
                self.result = result[0]
                self.message = result[1]
            elif isinstance(result, ValidationResult):
                # If function returns a ValidationResult directly
                self.result = result
                self.message = "Test executed successfully."
            else:
                # If function returns something else, assume success
                self.result = ValidationResult.PASS
                self.message = "Test executed successfully."

        except Exception as e:
            self.result = ValidationResult.FAIL
            self.message = f"Exception during test execution: {str(e)}"

        # Record execution time and timestamp
        end_time = datetime.datetime.now()
        self.execution_time = (end_time - start_time).total_seconds()
        self.timestamp = end_time.isoformat()

        return self.result

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation case to dictionary."""
        return {
            "case_id": self.case_id,
            "description": self.description,
            "category": self.category,
            "severity": self.severity,
            "result": self.result.value
            if isinstance(self.result, ValidationResult)
            else self.result,
            "message": self.message,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }


class ValidationSuite:
    """A collection of validation test cases."""

    def __init__(self, suite_id: str, description: str):
        self.suite_id = suite_id
        self.description = description
        self.test_cases: Dict[str, ValidationCase] = {}
        self.logger = setup_logger(f"validation_suite_{suite_id}")

    def add_test_case(self, test_case: ValidationCase) -> None:
        """Add a test case to the suite."""
        self.test_cases[test_case.case_id] = test_case

    def execute_all(self, *args, **kwargs) -> Dict[str, ValidationResult]:
        """Execute all test cases in the suite."""
        results = {}

        self.logger.info(f"Starting validation suite: {self.suite_id}")

        for case_id, test_case in self.test_cases.items():
            self.logger.info(f"Executing test case: {case_id}")
            result = test_case.execute(*args, **kwargs)
            results[case_id] = result

            self.logger.info(f"Test case {case_id} result: {result}")
            if result != ValidationResult.PASS:
                self.logger.warning(f"Test case {case_id} message: {test_case.message}")

        self.logger.info(f"Validation suite {self.suite_id} completed")
        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate a validation report."""
        report = {
            "suite_id": self.suite_id,
            "description": self.description,
            "timestamp": datetime.datetime.now().isoformat(),
            "test_cases": {
                case_id: case.to_dict() for case_id, case in self.test_cases.items()
            },
            "summary": {
                "total": len(self.test_cases),
                "passed": sum(
                    1
                    for case in self.test_cases.values()
                    if case.result == ValidationResult.PASS
                ),
                "failed": sum(
                    1
                    for case in self.test_cases.values()
                    if case.result == ValidationResult.FAIL
                ),
                "warnings": sum(
                    1
                    for case in self.test_cases.values()
                    if case.result == ValidationResult.WARNING
                ),
                "not_tested": sum(
                    1
                    for case in self.test_cases.values()
                    if case.result == ValidationResult.NOT_TESTED
                ),
            },
        }

        return report

    def save_report(self, filename: Optional[str] = None) -> Path:
        """Save validation report to file."""
        report = self.generate_report()

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{self.suite_id}_{timestamp}.json"

        # Save report using results_manager
        report_path = results_manager.get_path("reports") / filename

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Validation report saved to: {report_path}")
        return report_path
