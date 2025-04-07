# Utilities API Reference

## Overview

This document provides a comprehensive reference for the utility functions and classes provided in the CIRCMAN5.0 system's `utils` package. These utilities provide core functionality used throughout the system, including error handling, logging, results management, and data type definitions.

## Error Handling Utilities

The error handling utilities provide specialized exception classes for different types of errors in the CIRCMAN5.0 system.

**Module**: `src.circman5.utils.errors`

### ManufacturingError

Base class for all manufacturing-related errors in the system.

```python
class ManufacturingError(Exception):
    """Base class for manufacturing-related errors"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
```

**Parameters**:
- `message` (str): Error message describing the issue
- `error_code` (Optional[str]): Optional error code for categorization

**Attributes**:
- `message` (str): Error message
- `error_code` (Optional[str]): Error code

**Example Usage**:
```python
try:
    # Some manufacturing operation
    if not valid_operation:
        raise ManufacturingError("Invalid manufacturing operation", "MFG_ERR_001")
except ManufacturingError as e:
    logger.error(f"Manufacturing error: {e.message} (Code: {e.error_code})")
```

### ValidationError

Error raised for data validation issues.

```python
class ValidationError(ManufacturingError):
    """Error raised for data validation issues"""

    def __init__(self, message: str, invalid_data: Optional[Dict[Any, Any]] = None):
        super().__init__(message, error_code="VAL_ERR")
        self.invalid_data = invalid_data
```

**Parameters**:
- `message` (str): Error message describing the validation issue
- `invalid_data` (Optional[Dict[Any, Any]]): Dictionary containing the invalid data

**Attributes**:
- `message` (str): Error message
- `error_code` (str): Error code (fixed as "VAL_ERR")
- `invalid_data` (Optional[Dict[Any, Any]]): The invalid data that failed validation

**Example Usage**:
```python
def validate_input_data(data):
    """Validate input data."""
    if "quantity" not in data:
        raise ValidationError("Missing required field: quantity", data)
    if data.get("quantity", 0) <= 0:
        raise ValidationError("Quantity must be positive", {"quantity": data.get("quantity")})
```

### ProcessError

Error raised for manufacturing process issues.

```python
class ProcessError(ManufacturingError):
    """Error raised for manufacturing process issues"""

    def __init__(self, message: str, process_name: Optional[str] = None):
        super().__init__(message, error_code="PROC_ERR")
        self.process_name = process_name
```

**Parameters**:
- `message` (str): Error message describing the process issue
- `process_name` (Optional[str]): Name of the process that encountered the error

**Attributes**:
- `message` (str): Error message
- `error_code` (str): Error code (fixed as "PROC_ERR")
- `process_name` (Optional[str]): Name of the affected process

**Example Usage**:
```python
try:
    result = run_manufacturing_process("cutting")
    if not result.success:
        raise ProcessError("Process failed to complete", "cutting")
except Exception as e:
    raise ProcessError(f"Unexpected error in process", "cutting") from e
```

### DataError

Error raised for data handling issues.

```python
class DataError(ManufacturingError):
    """Error raised for data handling issues"""

    def __init__(self, message: str, data_source: Optional[str] = None):
        super().__init__(message, error_code="DATA_ERR")
        self.data_source = data_source
```

**Parameters**:
- `message` (str): Error message describing the data issue
- `data_source` (Optional[str]): Source of the data that encountered the error

**Attributes**:
- `message` (str): Error message
- `error_code` (str): Error code (fixed as "DATA_ERR")
- `data_source` (Optional[str]): Source of the problematic data

**Example Usage**:
```python
def load_production_data(file_path):
    """Load production data from file."""
    try:
        # Attempt to load data
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise DataError(f"Production data file not found", file_path)
    except pd.errors.EmptyDataError:
        raise DataError(f"Production data file is empty", file_path)
    except Exception as e:
        raise DataError(f"Error loading production data: {str(e)}", file_path)
```

### ResourceError

Error raised for resource-related issues.

```python
class ResourceError(ManufacturingError):
    """Error raised for resource-related issues"""

    def __init__(self, message: str, resource_type: Optional[str] = None):
        super().__init__(message, error_code="RES_ERR")
        self.resource_type = resource_type
```

**Parameters**:
- `message` (str): Error message describing the resource issue
- `resource_type` (Optional[str]): Type of resource that encountered the error

**Attributes**:
- `message` (str): Error message
- `error_code` (str): Error code (fixed as "RES_ERR")
- `resource_type` (Optional[str]): Type of the problematic resource

**Example Usage**:
```python
def allocate_resources(resource_type, quantity):
    """Allocate resources for manufacturing."""
    available = get_available_resources(resource_type)
    if available < quantity:
        raise ResourceError(
            f"Insufficient {resource_type} resources. Requested: {quantity}, Available: {available}",
            resource_type
        )
```

## Logging Utilities

The logging utilities provide a standardized approach to configuring and using logging throughout the CIRCMAN5.0 system.

**Module**: `src.circman5.utils.logging_config`

### setup_logger

Configures a logger with file and console handlers.

```python
def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Configure logging system with file and console handlers.

    Args:
        name: Name of the logger
        log_dir: Optional custom log directory path
        file_level: Logging level for file output
        console_level: Logging level for console output

    Returns:
        logging.Logger: Configured logger instance
    """
```

**Parameters**:
- `name` (str): Name of the logger
- `log_dir` (Optional[str]): Optional custom log directory path
- `file_level` (int): Logging level for file output (default: logging.DEBUG)
- `console_level` (int): Logging level for console output (default: logging.INFO)

**Returns**:
- `logging.Logger`: Configured logger instance

**Behavior**:
- Creates a timestamped log file
- Adds a file handler at the specified level
- Adds a console handler at the specified level
- Falls back to basic console logging if setup fails

**Example Usage**:
```python
from circman5.utils.logging_config import setup_logger

# Create a logger for a component
logger = setup_logger("manufacturing_analyzer")

# Use the logger
logger.debug("Detailed debug information")
logger.info("Processing batch XYZ")
logger.warning("Resource levels running low")
logger.error("Process failed to complete")
```

## Results Management Utilities

The results management utilities provide a centralized approach to managing paths and saving results in the CIRCMAN5.0 system.

**Module**: `src.circman5.utils.results_manager`

### ResultsManager

Centralized results and path management.

```python
class ResultsManager:
    """Centralized results and path management."""

    def __init__(self):
        """Initialize path manager with project directories."""
```

**Singleton Pattern**: This class implements the singleton pattern, ensuring only one instance exists.

**Key Methods**:

#### get_run_dir()

Gets the current run directory.

```python
def get_run_dir(self) -> Path:
    """Get current run directory."""
```

**Returns**:
- `Path`: Path to the current run directory

**Example Usage**:
```python
from circman5.utils.results_manager import results_manager

# Get current run directory
run_dir = results_manager.get_run_dir()
output_file = run_dir / "analysis_output.csv"
```

#### get_path(key)

Gets a path by key.

```python
def get_path(self, key: str) -> Path:
    """Get path by key."""
```

**Parameters**:
- `key` (str): Key identifying the path

**Returns**:
- `Path`: Path corresponding to the key

**Raises**:
- `KeyError`: If the key is invalid

**Available Path Keys**:
- `DATA_DIR`: Base data directory
- `SYNTHETIC_DATA`: Directory for synthetic data
- `PROCESSED_DATA`: Directory for processed data
- `RAW_DATA`: Directory for raw data
- `RESULTS_BASE`: Base results directory
- `RESULTS_ARCHIVE`: Directory for archived results
- `RESULTS_RUNS`: Directory for run results
- `LOGS_DIR`: Directory for logs
- `LOGS_ARCHIVE`: Directory for archived logs
- Run-specific directories (e.g., `reports`, `visualizations`, `metrics`)

**Example Usage**:
```python
from circman5.utils.results_manager import results_manager

# Get path to specific directory
data_dir = results_manager.get_path("SYNTHETIC_DATA")
vis_dir = results_manager.get_path("visualizations")
```

#### save_file(file_path, target_dir)

Saves a file to the specified target directory.

```python
def save_file(self, file_path: Union[str, Path], target_dir: str) -> Path:
    """Save file to specified target directory."""
```

**Parameters**:
- `file_path` (Union[str, Path]): Path to the file to save
- `target_dir` (str): Target directory key (e.g., "reports", "visualizations")

**Returns**:
- `Path`: Path to the saved file

**Raises**:
- `ValueError`: If the target directory is invalid

**Example Usage**:
```python
from circman5.utils.results_manager import results_manager

# Save a file to the visualizations directory
vis_path = results_manager.save_file("my_plot.png", "visualizations")

# Save a file to the reports directory
report_path = results_manager.save_file("analysis_report.xlsx", "reports")
```

#### cleanup_old_runs(keep_last)

Archives old run directories.

```python
def cleanup_old_runs(self, keep_last: int = 5) -> None:
    """Archive old run directories."""
```

**Parameters**:
- `keep_last` (int): Number of recent runs to keep (default: 5)

**Example Usage**:
```python
from circman5.utils.results_manager import results_manager

# Keep only the 3 most recent runs
results_manager.cleanup_old_runs(keep_last=3)
```

#### save_to_path(file_path, target_path_key)

Saves a file to a specified path.

```python
def save_to_path(self, file_path: Union[str, Path], target_path_key: str) -> Path:
    """Save file to a specified path from self.paths."""
```

**Parameters**:
- `file_path` (Union[str, Path]): Path to the file to save
- `target_path_key` (str): Key of the target path from self.paths

**Returns**:
- `Path`: Path to the saved file

**Raises**:
- `ValueError`: If the target path key is invalid

**Example Usage**:
```python
from circman5.utils.results_manager import results_manager

# Save a file to the processed data directory
processed_path = results_manager.save_to_path("cleaned_data.csv", "PROCESSED_DATA")
```

### results_manager

Global instance of ResultsManager.

```python
# Global instance for import
results_manager = ResultsManager()
```

**Example Usage**:
```python
from circman5.utils.results_manager import results_manager

# Use the global instance
data_dir = results_manager.get_path("DATA_DIR")
results_dir = results_manager.get_path("visualizations")
```

## Cleanup Utilities

The cleanup utilities provide functions for managing test results and logs.

**Module**: `src.circman5.utils.cleanup`

### cleanup_test_results

Cleans up old test results and log files.

```python
def cleanup_test_results(keep_last: int = 5, max_log_age_days: int = 7) -> None:
    """Clean up old test results and log files.

    Args:
        keep_last: Number of recent test runs to keep
        max_log_age_days: Maximum age of log files in days
    """
```

**Parameters**:
- `keep_last` (int): Number of recent test runs to keep (default: 5)
- `max_log_age_days` (int): Maximum age of log files in days (default: 7)

**Behavior**:
- Archives old test run directories
- Archives old log files

**Example Usage**:
```python
from circman5.utils.cleanup import cleanup_test_results

# Clean up old test results and logs
cleanup_test_results(keep_last=3, max_log_age_days=5)
```

### cleanup_old_runs

Cleans up old test runs, keeping only the specified number of most recent runs.

```python
def cleanup_old_runs(runs_dir: Optional[Path] = None, keep_last: int = 5) -> None:
    """Clean up old test runs, keeping only the specified number of most recent runs.

    Args:
        runs_dir: Optional directory containing test runs. If None, uses results_manager
        keep_last: Number of most recent runs to keep
    """
```

**Parameters**:
- `runs_dir` (Optional[Path]): Optional directory containing test runs
- `keep_last` (int): Number of most recent runs to keep (default: 5)

**Behavior**:
- Archives old run directories from the specified directory

**Example Usage**:
```python
from circman5.utils.cleanup import cleanup_old_runs
from pathlib import Path

# Clean up old runs in a specific directory
custom_dir = Path("/path/to/custom/runs")
cleanup_old_runs(runs_dir=custom_dir, keep_last=3)

# Clean up using results_manager
cleanup_old_runs(keep_last=3)
```

## Data Type Definitions

The data type definitions provide standardized dataclasses for common data entities in the CIRCMAN5.0 system.

**Module**: `src.circman5.utils.data_types`

### BatchData

Manufacturing batch information.

```python
@dataclass
class BatchData:
    """Manufacturing batch information."""

    batch_id: str
    start_time: datetime
    stage: str
    status: str
    input_material: str
    input_amount: float
    output_amount: float = 0.0
    yield_rate: float = 0.0
    energy_used: float = 0.0
    completion_time: Optional[datetime] = None
```

**Fields**:
- `batch_id` (str): Unique identifier for the batch
- `start_time` (datetime): Start time of the batch
- `stage` (str): Current stage of the batch
- `status` (str): Current status of the batch
- `input_material` (str): Type of input material
- `input_amount` (float): Amount of input material
- `output_amount` (float): Amount of output product (default: 0.0)
- `yield_rate` (float): Yield rate of the batch (default: 0.0)
- `energy_used` (float): Energy used in the batch (default: 0.0)
- `completion_time` (Optional[datetime]): Completion time of the batch (default: None)

**Example Usage**:
```python
from datetime import datetime
from circman5.utils.data_types import BatchData

# Create a batch
batch = BatchData(
    batch_id="BATCH-2025-001",
    start_time=datetime.now(),
    stage="Initialization",
    status="Active",
    input_material="Silicon",
    input_amount=1000.0
)

# Update batch information
batch.output_amount = 950.0
batch.yield_rate = 95.0
batch.energy_used = 500.0
```

### QualityData

Quality control measurements.

```python
@dataclass
class QualityData:
    """Quality control measurements."""

    batch_id: str
    test_time: datetime
    efficiency: float
    defect_rate: float
    thickness_uniformity: float
    contamination_level: float
```

**Fields**:
- `batch_id` (str): Batch identifier
- `test_time` (datetime): Time of quality test
- `efficiency` (float): Efficiency measurement
- `defect_rate` (float): Rate of defects
- `thickness_uniformity` (float): Uniformity of thickness
- `contamination_level` (float): Level of contamination

**Example Usage**:
```python
from datetime import datetime
from circman5.utils.data_types import QualityData

# Create quality data
quality = QualityData(
    batch_id="BATCH-2025-001",
    test_time=datetime.now(),
    efficiency=21.5,
    defect_rate=1.2,
    thickness_uniformity=98.5,
    contamination_level=0.3
)
```

### CircularMetrics

Circular economy metrics.

```python
@dataclass
class CircularMetrics:
    """Circular economy metrics."""

    batch_id: str
    recycled_content: float
    recyclable_output: float
    water_reused: float
    material_efficiency: float
    waste_recyclability: float = 95.0
```

**Fields**:
- `batch_id` (str): Batch identifier
- `recycled_content` (float): Percentage of recycled content in input
- `recyclable_output` (float): Percentage of output that is recyclable
- `water_reused` (float): Percentage of water reused
- `material_efficiency` (float): Material efficiency percentage
- `waste_recyclability` (float): Percentage of waste that is recyclable (default: 95.0)

**Example Usage**:
```python
from circman5.utils.data_types import CircularMetrics

# Create circular metrics
metrics = CircularMetrics(
    batch_id="BATCH-2025-001",
    recycled_content=30.0,
    recyclable_output=98.5,
    water_reused=85.0,
    material_efficiency=92.5
)
```

## Usage Examples

### Setting Up Logging for a Component

```python
from circman5.utils.logging_config import setup_logger

class ManufacturingComponent:
    """Example manufacturing component."""

    def __init__(self):
        """Initialize component with logging."""
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Component initialized")

    def process(self, data):
        """Process manufacturing data."""
        try:
            self.logger.debug(f"Processing data: {data}")
            # Processing logic
            result = self._calculate_result(data)
            self.logger.info(f"Processing completed with result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def _calculate_result(self, data):
        """Calculate result from data."""
        # Implementation
        return {"status": "success", "value": 42}
```

### Managing Results and Paths

```python
from pathlib import Path
from circman5.utils.results_manager import results_manager
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_visualize(data_file):
    """Analyze data and create visualizations."""
    # Load data
    data = pd.read_csv(data_file)

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(data["timestamp"], data["value"])
    plt.title("Value Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")

    # Save visualization
    vis_dir = results_manager.get_path("visualizations")
    vis_path = vis_dir / "value_over_time.png"
    plt.savefig(vis_path)
    plt.close()

    # Generate report
    report = data.describe()
    report_dir = results_manager.get_path("reports")
    report_path = report_dir / "data_summary.csv"
    report.to_csv(report_path)

    return {
        "visualization": vis_path,
        "report": report_path
    }
```

### Error Handling and Validation

```python
from circman5.utils.errors import ValidationError, ProcessError, DataError
import pandas as pd

def process_manufacturing_data(data_file):
    """Process manufacturing data with proper error handling."""
    try:
        # Attempt to load data
        try:
            data = pd.read_csv(data_file)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            raise DataError(f"Error loading data: {str(e)}", data_file)

        # Validate data
        required_columns = ["timestamp", "batch_id", "value"]
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValidationError(
                f"Missing required columns: {missing_columns}",
                {"columns": list(data.columns)}
            )

        # Process data
        try:
            result = perform_calculation(data)
            return result
        except Exception as e:
            raise ProcessError(f"Error in calculation: {str(e)}", "data_processing")

    except ManufacturingError as e:
        # Handle all manufacturing errors
        logger.error(f"Manufacturing error: {str(e)}")
        # Appropriate error handling (e.g., return default value, raise, etc.)
        raise
```

### Cleanup After Testing

```python
from circman5.utils.cleanup import cleanup_test_results

def run_test_suite():
    """Run test suite and clean up afterward."""
    try:
        # Run tests
        execute_tests()

        # Generate reports
        generate_test_reports()

        # Clean up old test results
        cleanup_test_results(keep_last=5, max_log_age_days=7)

    except Exception as e:
        print(f"Error running tests: {str(e)}")
        raise
```

## Best Practices

### Logging Best Practices

1. **Use setup_logger**: Always use the `setup_logger` function to create loggers
2. **Appropriate Log Levels**: Use appropriate log levels for different types of messages
   - `DEBUG`: Detailed information for debugging
   - `INFO`: General information about program execution
   - `WARNING`: Indication of potential issues
   - `ERROR`: Error conditions that should be investigated
   - `CRITICAL`: Critical errors that require immediate attention
3. **Contextual Information**: Include contextual information in log messages
4. **Exception Logging**: Log exceptions with traceback information
5. **Performance Consideration**: Avoid expensive operations in debug-level logs

### Error Handling Best Practices

1. **Use Specific Exceptions**: Use the most specific exception class for the situation
2. **Include Context**: Include contextual information when raising exceptions
3. **Clean Error Messages**: Provide clear, actionable error messages
4. **Graceful Degradation**: When possible, handle errors gracefully
5. **Consistent Pattern**: Follow a consistent error handling pattern throughout the codebase

### Results Management Best Practices

1. **Centralized Management**: Use the `results_manager` singleton for all path management
2. **Standard Directories**: Store results in standard directories based on type
3. **Unique Filenames**: Use unique filenames to avoid overwriting results
4. **Regular Cleanup**: Regularly clean up old results to manage disk space
5. **Path Objects**: Use Path objects (not strings) for file path manipulation

### Data Type Best Practices

1. **Type Consistency**: Use consistent types for similar fields across dataclasses
2. **Default Values**: Provide sensible default values for optional fields
3. **Documentation**: Document the purpose and units of each field
4. **Validation**: Validate input data before creating dataclass instances
5. **Immutability**: Consider making dataclasses immutable (`frozen=True`) when appropriate
