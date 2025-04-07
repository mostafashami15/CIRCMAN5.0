# Developer Guide

## Introduction

This guide provides essential information for developers working on the CIRCMAN5.0 project. It covers development environment setup, testing practices, contribution workflows, and coding standards. Following these guidelines will help maintain code quality and ensure consistent development practices across the project.

## Project Structure

CIRCMAN5.0 follows a structured organization to maintain code clarity and separation of concerns:

```
circman5/
├── src/
│   └── circman5/              # Main source code
│       ├── adapters/          # Configuration adapters
│       ├── config/            # Configuration management
│       ├── manufacturing/     # Manufacturing core functionality
│       │   ├── analyzers/     # Analysis components
│       │   ├── digital_twin/  # Digital twin implementation
│       │   ├── human_interface/ # Human interface components
│       │   ├── lifecycle/     # Lifecycle assessment
│       │   ├── optimization/  # Optimization algorithms
│       │   └── reporting/     # Reporting and visualization
│       └── utils/             # Utility functions
├── tests/                     # Test suite
│   ├── integration/           # Integration tests
│   ├── performance/           # Performance tests
│   ├── unit/                  # Unit tests
│   └── validation/            # Validation tests
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   ├── architecture/          # Architecture documentation
│   ├── guides/                # User and developer guides
│   └── implementation/        # Implementation details
├── data/                      # Data files
│   ├── processed/             # Processed data files
│   ├── raw/                   # Raw data files
│   └── synthetic/             # Synthetic test data
├── scripts/                   # Utility scripts
│   ├── backup/                # Backup scripts
│   └── maintenance/           # Maintenance scripts
└── notebooks/                 # Analysis notebooks
```

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- Poetry package manager
- Git

### Setting Up the Development Environment

1. **Clone the repository**:

```bash
git clone https://github.com/your-organization/CIRCMAN5.0.git
cd CIRCMAN5.0
```

2. **Install dependencies using Poetry**:

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

3. **Activate the virtual environment**:

```bash
poetry shell
```

4. **Verify installation**:

```bash
# Run the test suite to verify setup
poetry run pytest tests/
```

### Alternative Setup with pip

If Poetry is not available, you can use pip:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

## Testing Framework

CIRCMAN5.0 uses pytest as its testing framework, with custom extensions for manufacturing-specific testing.

### Test Organization

Tests are organized into four categories:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **Performance Tests**: Test system performance characteristics
4. **Validation Tests**: Test system behavior against validation criteria

### Test Directory Structure

```
tests/
├── unit/                     # Unit tests
│   ├── adapters/             # Adapter tests
│   ├── manufacturing/        # Manufacturing component tests
│   └── utils/                # Utility function tests
├── integration/              # Integration tests
│   ├── human_interface/      # Human interface integration tests
│   └── system/               # System integration tests
├── performance/              # Performance tests
└── validation/               # Validation tests
    └── validation_framework.py  # Validation framework
```

### Running Tests

To run all tests:

```bash
pytest
```

To run specific test categories:

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/

# Run tests with a specific marker
pytest -m performance
```

### Test Markers

The project uses pytest markers to categorize tests:

```
markers =
    performance: marks tests as performance tests
```

To see all available markers:

```bash
pytest --markers
```

### Test Results Management

The `ResultsManager` class handles test results, organizing them by run:

```python
from circman5.utils.results_manager import results_manager

# Get paths for test outputs
results_dir = results_manager.get_path("metrics")
viz_dir = results_manager.get_path("visualizations")

# Save test results
results_manager.save_file(your_file_path, "metrics")
```

## The `test_framework.py` Module

The `test_framework.py` module provides a comprehensive testing framework for manufacturing analysis:

```python
from circman5.test_framework import test_framework

# Run the comprehensive test framework
test_framework()
```

Key features:
- Generates synthetic test data
- Tests data loading capabilities
- Tests analysis workflows
- Tests reporting and visualization
- Logs test results

## Development Workflow

### Branching Strategy

We recommend the following branching strategy:

1. `main`: The stable, release-ready branch
2. `develop`: Integration branch for new features
3. Feature branches: For individual features or bug fixes

For a new feature:

```bash
# Create a feature branch from develop
git checkout develop
git pull
git checkout -b feature/your-feature-name

# Make changes, commit, and push
git add .
git commit -m "Implement feature X"
git push -u origin feature/your-feature-name
```

### Code Review Process

1. Create a pull request from your feature branch to `develop`
2. Ensure all tests pass
3. Request review from at least one team member
4. Address review comments
5. Once approved, merge into `develop`

## Code Style and Standards

### Python Version

CIRCMAN5.0 requires Python 3.11 or higher.

### Code Formatting

- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length is 88 characters (Black default)
- Use 4 spaces for indentation (not tabs)

### Import Conventions

Imports should be organized in the following order:

1. Standard library imports
2. Third-party library imports
3. Local application imports

Example:

```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local application
from circman5.utils.results_manager import results_manager
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
```

If you encounter import issues, you can use the provided script:

```bash
python scripts/fix_imports.py
```

### Docstring Style

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of the function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When and why this exception is raised
    """
```

### Type Hints

Use type hints consistently for all function definitions:

```python
def process_data(data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """Process data with the given threshold."""
    # Implementation
```

## Project Maintenance Scripts

CIRCMAN5.0 includes several scripts for project maintenance:

### fix_imports.py

Fixes import statements throughout the project:

```bash
python scripts/fix_imports.py
```

This script corrects common import issues, such as:
- Relative imports that should be absolute
- Imports from src.circman5 that should be from circman5
- Missing or incorrect import paths

### fix_project_structure.py

Fixes project structure issues:

```bash
python scripts/fix_project_structure.py
```

This script:
- Updates core file imports to use absolute imports
- Cleans up backup files
- Creates required `__init__.py` files

### verify_structure.py

Verifies and creates the required directory structure:

```bash
python scripts/verify_structure.py
```

This script:
- Creates the required directory structure if it doesn't exist
- Adds missing `__init__.py` files
- Logs created directories and files

## Dependency Management

CIRCMAN5.0 uses Poetry for dependency management. The `pyproject.toml` file defines all project dependencies:

```toml
[tool.poetry.dependencies]
python = "^3.11"
numpy = "^2.2.2"
pandas = "^2.2.3"
matplotlib = "^3.5.0"
seaborn = "^0.12.0"
scikit-learn = "^1.6.1"
psutil = "^6.1.1"
openpyxl = "^3.1.5"

[tool.poetry.group.dev.dependencies]
pre-commit = "^4.1.0"
pytest = "^7.1.3"
pytest-html = "^4.1.1"
pytest-cov = "^6.0.0"
pytest-xdist = "^3.6.1"
pytest-mock = "^3.14.0"
```

To add a new dependency:

```bash
poetry add package-name
```

To add a development dependency:

```bash
poetry add --group dev package-name
```

## Testing Best Practices

### Unit Testing

1. **Test Individual Components**: Each class or function should have corresponding unit tests
2. **Use Fixtures**: Use pytest fixtures for test setup and teardown
3. **Mock Dependencies**: Use pytest-mock to mock external dependencies
4. **Test Edge Cases**: Include tests for edge cases and error conditions
5. **Parameterize Tests**: Use pytest's parameterize for testing with multiple inputs

Example unit test:

```python
# tests/unit/manufacturing/test_core.py
import pytest
from circman5.manufacturing.core import SoliTekManufacturingAnalysis

class TestManufacturingCore:
    """Test core manufacturing analysis functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.analyzer = SoliTekManufacturingAnalysis()

    def test_initialization(self):
        """Test proper analyzer initialization."""
        assert isinstance(self.analyzer.efficiency_analyzer, EfficiencyAnalyzer)
        assert isinstance(self.analyzer.quality_analyzer, QualityAnalyzer)
        assert isinstance(self.analyzer.sustainability_analyzer, SustainabilityAnalyzer)
        assert isinstance(self.analyzer.lca_analyzer, LCAAnalyzer)
        assert not self.analyzer.is_optimizer_trained
```

### Integration Testing

1. **Test Component Interactions**: Focus on testing how components work together
2. **Use Real Dependencies**: Use real dependencies when practical
3. **Test Complete Workflows**: Test end-to-end workflows
4. **Validate Data Flow**: Verify data flows correctly between components

Example integration test:

```python
# tests/integration/test_digital_twin_integration.py
def test_digital_twin_optimization():
    """Test optimization using the Digital Twin."""
    analyzer = SoliTekManufacturingAnalysis()

    # Define test parameters
    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    # Run optimization
    optimized = analyzer.optimize_using_digital_twin(
        test_params, simulation_steps=3
    )
    assert isinstance(optimized, dict)
    assert "input_amount" in optimized
```

### Validation Testing

CIRCMAN5.0 includes a validation framework for system verification:

```python
# tests/validation/test_system_validation.py
from tests.validation.validation_framework import ValidationSuite, ValidationCase, ValidationResult

def test_system_validation():
    """Test system validation."""
    # Create validation suite
    suite = ValidationSuite("system_validation", "System Validation Suite")

    # Add test cases
    suite.add_test_case(
        ValidationCase(
            "core_functionality",
            "Test core functionality",
            test_core_functionality,
            "critical"
        )
    )

    # Execute tests
    results = suite.execute_all()

    # Generate report
    report_path = suite.save_report()

    # Verify results
    assert all(result == ValidationResult.PASS for result in results.values())
```

## Debugging and Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'circman5'`
**Solution**:
1. Ensure you're running from the project root
2. Run `pip install -e .` to install the package in development mode
3. Use the `fix_imports.py` script to correct import statements

#### Test Failures

**Problem**: Test failures with path-related errors
**Solution**:
1. Check if you're running tests from the correct directory
2. Verify that ResultsManager is initialized properly
3. Check file paths in test fixtures

#### Visualization Errors

**Problem**: Visualization errors when running tests
**Solution**:
1. Ensure matplotlib is properly configured
2. Use non-interactive backend with `plt.switch_backend('Agg')`
3. Close figures after tests with `plt.close('all')`

### Debugging with pytest

To run tests in debug mode:

```bash
pytest --pdb
```

To get more verbose output:

```bash
pytest -v
```

To see print statements during tests:

```bash
pytest -s
```

## Continuous Integration

CIRCMAN5.0 uses GitHub Actions for continuous integration. The workflow:

1. Runs on each push to main and develop branches
2. Runs on pull requests to main and develop
3. Executes the test suite
4. Checks code formatting with Black
5. Builds and publishes documentation

## Documentation

### Documentation Structure

```
docs/
├── api/                  # API documentation
├── architecture/         # Architecture documentation
├── guides/               # User and developer guides
├── implementation/       # Implementation details
└── mathematical/         # Mathematical foundations
```

### Documentation Style

Use Markdown for documentation files. Follow these guidelines:

1. Use descriptive titles and headings
2. Include code examples when relevant
3. Use proper heading hierarchy (H1, H2, H3)
4. Include cross-references to other documentation
5. Use tables for structured data
6. Include diagrams when helpful

## Performance Optimization

### Profiling Code

Use the built-in Python profiler:

```python
import cProfile
import pstats

# Profile a function
cProfile.run('your_function()', 'profile_stats')

# Analyze results
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(10)
```

### Memory Optimization

1. Use generators for processing large datasets
2. Avoid unnecessary copies of large arrays or DataFrames
3. Close matplotlib figures after use
4. Use appropriate data types to minimize memory usage

## Contributing New Features

To contribute a new feature:

1. **Discuss**: Start by discussing the feature in an issue
2. **Plan**: Outline the implementation approach
3. **Implement**: Develop the feature with tests
4. **Document**: Add documentation for the feature
5. **Review**: Submit a pull request for review

### Feature Implementation Checklist

- [ ] Unit tests for all new functionality
- [ ] Integration tests for component interactions
- [ ] Documentation updates
- [ ] Type hints for all functions
- [ ] Docstrings for all public methods
- [ ] Performance tests if applicable
- [ ] Example usage in documentation

## Advanced Development Topics

### Extending the Digital Twin

The Digital Twin architecture is extensible. To add new functionality:

1. Identify the appropriate component (core, simulation, etc.)
2. Implement new functionality following existing patterns
3. Add appropriate tests
4. Update documentation

### Creating Custom Analyzers

To create a custom analyzer:

1. Create a new class in the appropriate directory
2. Implement the analyzer interface
3. Register with the appropriate manager
4. Add tests and documentation

Example:

```python
# src/circman5/manufacturing/analyzers/custom_analyzer.py
class CustomAnalyzer:
    """Custom analyzer for specific metrics."""

    def __init__(self):
        self.logger = setup_logger("custom_analyzer")

    def analyze_custom_metric(self, data):
        """Analyze a custom metric."""
        # Implementation
        return result
```

### Optimization Framework Integration

To extend the optimization framework:

1. Create a new optimizer in the optimization directory
2. Implement the optimizer interface
3. Register with the optimization manager
4. Add tests and documentation

## Conclusion

Following the guidelines in this Developer Guide will help maintain a consistent, high-quality codebase for CIRCMAN5.0. The project's structured approach to testing, documentation, and development ensures that all contributors can effectively collaborate.

For additional information, refer to:

- [Adapter API Reference](../api/adapter_api_reference.md)
- [Digital Twin Implementation Guide](../implementation/dt_implementation_guide.md)
- [Monitoring Guide](../guides/monitoring_guide.md)
- [Reporting Implementation Guide](../implementation/reporting_implementation_guide.md)

## Appendix: Quick Reference

### Common Commands

```bash
# Install dependencies
poetry install

# Run all tests
pytest

# Run specific test file
pytest tests/unit/path/to/test_file.py

# Run tests with specific marker
pytest -m performance

# Format code with Black
black src/ tests/

# Fix imports
python scripts/fix_imports.py

# Verify project structure
python scripts/verify_structure.py
```

### Useful pytest Options

```bash
# Verbose output
pytest -v

# Show stdout
pytest -s

# Stop on first failure
pytest -x

# Run tests in parallel
pytest -n auto

# Generate coverage report
pytest --cov=src/circman5

# Generate HTML report
pytest --html=report.html
```
