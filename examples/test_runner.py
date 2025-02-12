#!/usr/bin/env python
"""
Full system test runner for CIRCMAN5.0
Executes all test suites and generates comprehensive reports
"""

import sys
import pytest
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from circman5.config.project_paths import project_paths
from circman5.utils.logging_config import setup_logger


class TestRunner:
    """Manages execution of all test suites and result collection."""

    def __init__(self):
        self.logger = setup_logger("test_runner")
        self.run_dir = project_paths.get_run_directory()
        self.results = {}

        # Create directories
        for subdir in ["test_results", "test_reports", "coverage"]:
            (self.run_dir / subdir).mkdir(exist_ok=True)

    def run_test_suite(self, test_path: str) -> dict:
        """Run a specific test suite and collect results."""
        start_time = time.time()

        # Run pytest with basic options
        result = pytest.main(
            [
                "-v",  # verbose output
                test_path,  # test path
                f"--junitxml={self.run_dir}/test_results/{Path(test_path).stem}_results.xml",
            ]
        )

        end_time = time.time()
        duration = end_time - start_time

        return {"result": result, "duration": duration, "timestamp": datetime.now()}

    def run_all_tests(self):
        """Execute all test suites in the correct order."""
        test_suites = [
            # Unit tests
            "tests/unit/test_lca_core.py",
            # Integration tests
            "tests/integration/test_data_pipeline.py",
            "tests/integration/test_manufacturing_optimization.py",
            "tests/integration/test_system_integration.py",
            # AI tests
            "tests/ai/test_optimization.py",
            # Performance tests
            "tests/performance/test_performance.py",
        ]

        for test_path in test_suites:
            self.logger.info(f"Running test suite: {test_path}")
            result = self.run_test_suite(test_path)
            self.results[test_path] = result

            status = "passed" if result["result"] == 0 else "failed"
            self.logger.info(f"Test suite {status}: {test_path}")

    def generate_test_summary(self):
        """Generate comprehensive test execution summary."""
        if not self.results:
            self.logger.warning("No test results to summarize")
            return

        summary_data = []
        for test_path, result in self.results.items():
            summary_data.append(
                {
                    "test_suite": Path(test_path).stem,
                    "status": "Passed" if result["result"] == 0 else "Failed",
                    "duration": result["duration"],
                    "timestamp": result["timestamp"],
                }
            )

        summary_df = pd.DataFrame(summary_data)

        # Save CSV summary
        summary_path = self.run_dir / "test_results/test_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Test summary saved to {summary_path}")

        # Create visualizations
        self._create_summary_plots(summary_df)

    def _create_summary_plots(self, summary_df: pd.DataFrame):
        """Create summary visualizations."""
        plt.figure(figsize=(12, 6))

        # Duration plot
        plt.subplot(1, 2, 1)
        sns.barplot(data=summary_df, x="test_suite", y="duration")
        plt.xticks(rotation=45, ha="right")
        plt.title("Test Suite Duration")

        # Status plot
        plt.subplot(1, 2, 2)
        status_counts = summary_df["status"].value_counts()
        # Convert values and labels to compatible types
        values = status_counts.to_numpy()
        labels = list(status_counts.index.astype(str))
        colors = ["#2ecc71" if status == "Passed" else "#e74c3c" for status in labels]

        plt.pie(x=values, labels=labels, autopct="%1.1f%%", colors=colors)
        plt.title("Test Results Distribution")

        plt.tight_layout()
        plot_path = self.run_dir / "test_results/test_summary.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        self.logger.info(f"Test summary plot saved to {plot_path}")

    def cleanup_old_results(self, keep_last: int = 5):
        """Clean up old test results."""
        try:
            project_paths.cleanup_old_runs(keep_last)
            self.logger.info(f"Cleaned up old results, keeping last {keep_last} runs")
        except Exception as e:
            self.logger.error(f"Error cleaning up old results: {e}")


def main():
    """Main execution function."""
    runner = TestRunner()

    try:
        # Run all test suites
        runner.run_all_tests()

        # Generate summary
        runner.generate_test_summary()

        # Cleanup old results
        runner.cleanup_old_results()

        # Check final status
        successful = all(r["result"] == 0 for r in runner.results.values())
        if successful:
            runner.logger.info("All test suites completed successfully!")
            sys.exit(0)
        else:
            runner.logger.error("Some test suites failed. Check results for details.")
            sys.exit(1)

    except Exception as e:
        runner.logger.error(f"Error during test execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
