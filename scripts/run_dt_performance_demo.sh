#!/bin/bash
# Script to run the Digital Twin Performance Demo

# Ensure we're in the project root
echo "Ensuring we're in the project root directory..."
# Get the directory of this script and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Activate poetry environment if available
if command -v poetry &> /dev/null; then
    echo "Activating poetry environment..."
    eval "$(poetry env info -p)/bin/activate"
fi

# Create examples directory if it doesn't exist
if [ ! -d "examples" ]; then
    echo "Creating examples directory..."
    mkdir -p examples
fi

# Check if the script file exists
if [ ! -f "examples/digital_twin_performance_demo.py" ]; then
    echo "Demo script not found! Make sure it exists at examples/digital_twin_performance_demo.py"
    exit 1
fi

# Make the demo script executable
chmod +x examples/digital_twin_performance_demo.py

# Create performance_results directory if it doesn't exist
if [ ! -d "performance_results" ]; then
    echo "Creating performance_results directory..."
    mkdir -p performance_results
fi

# Run the demo script
echo "Running Digital Twin Performance Demo..."
python examples/digital_twin_performance_demo.py

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Demo completed successfully!"
    echo "Performance results and visualizations saved in performance_results directory"
    echo "You can use these visualizations in your thesis Section 4.2"

    # Count the generated files
    RESULT_COUNT=$(ls -1 performance_results/*.png 2>/dev/null | wc -l)
    echo "Generated $RESULT_COUNT visualization files."
    echo "Generated files:"
    ls -la performance_results/
else
    echo "Error running demo script. Check for errors above."
fi

# Deactivate poetry environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating poetry environment..."
    deactivate
fi
