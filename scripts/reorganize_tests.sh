#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")"

echo "Creating new directory structure..."
# Create new directory structure
mkdir -p tests/unit/manufacturing/{analyzers,optimization,lifecycle,reporting}
mkdir -p tests/unit/utils

echo "Moving analyzer tests..."
# Move analyzer tests
mv tests/unit/test_efficiency_analyzer.py tests/unit/manufacturing/analyzers/test_efficiency.py 2>/dev/null || true
mv tests/unit/test_quality_analyzer.py tests/unit/manufacturing/analyzers/test_quality.py 2>/dev/null || true
mv tests/unit/test_sustainability_analyzer.py tests/unit/manufacturing/analyzers/test_sustainability.py 2>/dev/null || true

echo "Moving utils tests..."
# Move utils tests
mv tests/unit/test_logging_config.py tests/unit/utils/ 2>/dev/null || true
mv tests/unit/test_project_paths.py tests/unit/utils/ 2>/dev/null || true

echo "Creating new optimization tests..."
# Create new optimization tests
touch tests/unit/manufacturing/optimization/{test_model.py,test_optimizer.py,conftest.py}

echo "Adding __init__.py files..."
# Add __init__.py files
find tests -type d -exec touch {}/__init__.py \;

echo "Archiving old AI tests..."
# Archive old AI tests
mkdir -p tests/archive/ai
mv tests/ai/* tests/archive/ai/ 2>/dev/null || true

echo "Cleaning up empty directories..."
# Clean up empty directories
rm -rf tests/ai 2>/dev/null || true

echo "Done! Test directory has been reorganized."
