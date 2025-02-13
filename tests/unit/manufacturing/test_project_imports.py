# tests/unit/manufacturing/test_project_imports.py

import pytest
import importlib
from typing import List, Tuple


def verify_imports() -> Tuple[List[str], List[str]]:
    """Verify all critical project imports."""
    modules_to_check = [
        # Core manufacturing modules
        "circman5.manufacturing.core",
        "circman5.manufacturing.data_loader",
        "circman5.manufacturing.schemas",
        # Manufacturing analyzers
        "circman5.manufacturing.analyzers.efficiency",
        "circman5.manufacturing.analyzers.quality",
        "circman5.manufacturing.analyzers.sustainability",
        # Manufacturing reporting
        "circman5.manufacturing.reporting.reports",
        "circman5.manufacturing.reporting.visualizations",
        # Manufacturing lifecycle
        "circman5.manufacturing.lifecycle.lca_analyzer",
        "circman5.manufacturing.lifecycle.impact_factors",
        "circman5.manufacturing.lifecycle.visualizer",
        # Manufacturing optimization
        "circman5.manufacturing.optimization.model",
        "circman5.manufacturing.optimization.optimizer",
        "circman5.manufacturing.optimization.types",
        # Configuration and utilities
        "circman5.config.project_paths",
        "circman5.utils.data_types",
        "circman5.utils.errors",
        "circman5.utils.logging_config",
        "circman5.utils.result_paths",
        # Root level modules
        "circman5.constants",
        "circman5.monitoring",
        "circman5.test_data_generator",
    ]

    successful = []
    failed = []

    for module in modules_to_check:
        try:
            importlib.import_module(module)
            successful.append(module)
        except ImportError as e:
            failed.append(f"{module}: {str(e)}")

    return successful, failed


def test_project_imports():
    """Test that all project imports are working correctly."""
    successful, failed = verify_imports()

    # Print results for visibility in test output
    if failed:
        print("\nFailed imports:")
        for failure in failed:
            print(f"- {failure}")

    # Print success summary
    print(f"\nSuccessfully imported {len(successful)} modules")

    # Assert no failures
    assert not failed, f"Failed to import {len(failed)} modules"
