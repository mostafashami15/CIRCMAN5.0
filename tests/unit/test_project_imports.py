"""Unit tests for verifying project imports."""

import pytest
import importlib
from typing import List, Tuple


def verify_imports() -> Tuple[List[str], List[str]]:
    """Verify all critical project imports."""
    modules_to_check = [
        # Core modules
        "circman5.solitek_manufacturing",
        "circman5.test_data_generator",
        # AI and Optimization
        "circman5.ai.optimization_base",
        "circman5.ai.optimization_core",
        "circman5.ai.optimization_training",
        "circman5.ai.optimization_prediction",
        "circman5.ai.optimization_types",
        # Analysis modules
        "circman5.analysis.efficiency",
        "circman5.analysis.quality",
        "circman5.analysis.sustainability",
        "circman5.analysis.lca.core",
        # Configuration
        "circman5.config.project_paths",
        # Visualization
        "circman5.visualization.lca_visualizer",
        "circman5.visualization.manufacturing_visualizer",
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
