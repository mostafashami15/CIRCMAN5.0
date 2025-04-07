# Monitoring API Reference

## Overview

This document provides a comprehensive API reference for the CIRCMAN5.0 Monitoring System. The monitoring system is responsible for tracking key performance indicators (KPIs) in PV manufacturing processes, including efficiency metrics, quality metrics, and resource utilization.

## Core Components

### ManufacturingMonitor

`ManufacturingMonitor` is the main class responsible for monitoring and tracking manufacturing performance metrics.

**Module**: `src.circman5.monitoring`

```python
class ManufacturingMonitor:
    """Monitors and tracks manufacturing performance metrics."""

    def __init__(self):
        """Initialize monitoring system with empty metrics storage."""
```

The `ManufacturingMonitor` class stores metrics in three pandas DataFrames:
- `metrics_history["efficiency"]`: Efficiency-related metrics
- `metrics_history["quality"]`: Quality-related metrics
- `metrics_history["resources"]`: Resource utilization metrics

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `metrics_history` | Dict[str, pd.DataFrame] | Dictionary of DataFrames containing recorded metrics by category |
| `current_batch` | Optional[str] | ID of the batch currently being monitored |
| `logger` | Logger | Logger instance for monitoring-related logging |
| `constants` | ConstantsService | Service for accessing configuration constants |

#### Methods

##### `start_batch_monitoring(batch_id: str) -> None`

Starts monitoring a new manufacturing batch.

**Parameters**:
- `batch_id` (str): Unique identifier for the batch

**Returns**:
- None

**Description**:
Initializes monitoring for a new batch. This method should be called at the start of each production batch to ensure metrics are correctly associated with the batch.

**Example**:
```python
monitor = ManufacturingMonitor()
monitor.start_batch_monitoring("BATCH_2025_001")
```

---

##### `record_efficiency_metrics(output_quantity: float, cycle_time: float, energy_consumption: float) -> Dict`

Records efficiency-related metrics for the current batch.

**Parameters**:
- `output_quantity` (float): Amount of product produced
- `cycle_time` (float): Production cycle duration
- `energy_consumption` (float): Energy used in production

**Returns**:
- Dict: Dictionary containing calculated efficiency metrics

**Raises**:
- ValueError: If no active batch is being monitored

**Description**:
Records and calculates efficiency metrics for the current manufacturing batch. Calculates derived metrics like production rate and energy efficiency.

**Example**:
```python
efficiency_metrics = monitor.record_efficiency_metrics(
    output_quantity=100.0,
    cycle_time=60.0,
    energy_consumption=500.0
)
print(f"Production rate: {efficiency_metrics['production_rate']}")
```

**Output Dictionary Fields**:
- `batch_id`: Current batch identifier
- `timestamp`: Time when metrics were recorded
- `output_quantity`: Amount of product produced
- `cycle_time`: Production cycle duration
- `energy_consumption`: Energy used in production
- `production_rate`: Calculated as output_quantity / cycle_time
- `energy_efficiency`: Calculated as output_quantity / energy_consumption

---

##### `record_quality_metrics(defect_rate: float, yield_rate: float, uniformity_score: float) -> Dict`

Records quality-related metrics for the current batch.

**Parameters**:
- `defect_rate` (float): Percentage of defective products
- `yield_rate` (float): Production yield percentage
- `uniformity_score` (float): Product uniformity measure

**Returns**:
- Dict: Dictionary containing calculated quality metrics

**Description**:
Records and calculates quality metrics for the current manufacturing batch. Calculates a composite quality score based on configured weights.

**Example**:
```python
quality_metrics = monitor.record_quality_metrics(
    defect_rate=2.5,
    yield_rate=97.5,
    uniformity_score=95.0
)
print(f"Quality score: {quality_metrics['quality_score']}")
```

**Output Dictionary Fields**:
- `batch_id`: Current batch identifier
- `timestamp`: Time when metrics were recorded
- `defect_rate`: Percentage of defective products
- `yield_rate`: Production yield percentage
- `uniformity_score`: Product uniformity measure
- `quality_score`: Weighted composite quality score

---

##### `record_resource_metrics(material_consumption: float, water_usage: float, waste_generated: float) -> Dict`

Records resource utilization metrics for the current batch.

**Parameters**:
- `material_consumption` (float): Amount of raw materials used
- `water_usage` (float): Volume of water consumed
- `waste_generated` (float): Amount of waste produced

**Returns**:
- Dict: Dictionary containing calculated resource metrics

**Description**:
Records and calculates resource utilization metrics for the current manufacturing batch. Calculates resource efficiency as a measure of material utilization.

**Example**:
```python
resource_metrics = monitor.record_resource_metrics(
    material_consumption=1000.0,
    water_usage=500.0,
    waste_generated=50.0
)
print(f"Resource efficiency: {resource_metrics['resource_efficiency']}")
```

**Output Dictionary Fields**:
- `batch_id`: Current batch identifier
- `timestamp`: Time when metrics were recorded
- `material_consumption`: Amount of raw materials used
- `water_usage`: Volume of water consumed
- `waste_generated`: Amount of waste produced
- `resource_efficiency`: Calculated as (material_consumption - waste_generated) / material_consumption

---

##### `save_metrics(metric_type: str, save_path: Optional[Path] = None) -> None`

Saves metrics to appropriate location.

**Parameters**:
- `metric_type` (str): Type of metrics to save ('efficiency', 'quality', or 'resources')
- `save_path` (Optional[Path]): Optional explicit save path

**Returns**:
- None

**Raises**:
- ValueError: If invalid metric type is specified

**Description**:
Saves the specified type of metrics to a CSV file. If no explicit path is provided, uses the ResultsManager to save to the default location.

**Example**:
```python
# Save to default location
monitor.save_metrics("efficiency")

# Save to specific path
from pathlib import Path
monitor.save_metrics("quality", Path("/path/to/quality_data.csv"))
```

---

##### `get_batch_summary(batch_id: str) -> Dict`

Generates batch summary and saves to reports directory.

**Parameters**:
- `batch_id` (str): Batch identifier to summarize

**Returns**:
- Dict: Nested dictionary containing batch summary metrics

**Description**:
Generates a comprehensive summary of all metrics for the specified batch. The summary is saved as an Excel file in the reports directory and also returned as a dictionary.

**Example**:
```python
batch_summary = monitor.get_batch_summary("BATCH_2025_001")
print(f"Average production rate: {batch_summary['efficiency']['avg_production_rate']}")
```

**Output Dictionary Structure**:
```
{
    "efficiency": {
        "avg_production_rate": float,
        "total_energy_consumption": float,
        "avg_energy_efficiency": float
    },
    "quality": {
        "avg_defect_rate": float,
        "final_yield_rate": float,
        "avg_quality_score": float
    },
    "resources": {
        "total_material_consumption": float,
        "total_waste_generated": float,
        "avg_resource_efficiency": float
    }
}
```

---

##### `_record_batch_start(batch_id: str) -> None`

Records the start of a new batch monitoring session.

**Parameters**:
- `batch_id` (str): Batch identifier

**Returns**:
- None

**Description**:
Internal method that records the start time and initial conditions for a new batch monitoring session.

---

##### `_calculate_quality_score(defect_rate: float, yield_rate: float, uniformity_score: float) -> float`

Calculates composite quality score.

**Parameters**:
- `defect_rate` (float): Percentage of defective products
- `yield_rate` (float): Production yield percentage
- `uniformity_score` (float): Product uniformity measure

**Returns**:
- float: Calculated quality score

**Description**:
Internal method that calculates a weighted quality score based on defect rate, yield rate, and uniformity score. Weights are obtained from the configuration.

---

##### `_calculate_resource_efficiency(material_input: float, waste_output: float) -> float`

Calculates resource utilization efficiency.

**Parameters**:
- `material_input` (float): Amount of material input
- `waste_output` (float): Amount of waste output

**Returns**:
- float: Calculated resource efficiency

**Description**:
Internal method that calculates resource utilization efficiency as (material_input - waste_output) / material_input.

---

##### `_summarize_efficiency(batch_id: str) -> Dict`

Generates efficiency metrics summary for a batch.

**Parameters**:
- `batch_id` (str): Batch identifier

**Returns**:
- Dict: Summary of efficiency metrics

**Description**:
Internal method that generates a summary of efficiency metrics for the specified batch.

---

##### `_summarize_quality(batch_id: str) -> Dict`

Generates quality metrics summary for a batch.

**Parameters**:
- `batch_id` (str): Batch identifier

**Returns**:
- Dict: Summary of quality metrics

**Description**:
Internal method that generates a summary of quality metrics for the specified batch.

---

##### `_summarize_resources(batch_id: str) -> Dict`

Generates resource utilization summary for a batch.

**Parameters**:
- `batch_id` (str): Batch identifier

**Returns**:
- Dict: Summary of resource metrics

**Description**:
Internal method that generates a summary of resource utilization metrics for the specified batch.

## MonitoringAdapter

`MonitoringAdapter` is an adapter for loading monitoring system configuration.

**Module**: `src.circman5.adapters.config.monitoring`

```python
class MonitoringAdapter(ConfigAdapterBase):
    """Adapter for monitoring system configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize monitoring adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "monitoring.json"
        )
```

#### Methods

##### `load_config() -> Dict[str, Any]`

Loads monitoring configuration.

**Returns**:
- Dict[str, Any]: Monitoring configuration

**Raises**:
- FileNotFoundError: If config file not found
- ValueError: If config is invalid

**Description**:
Loads the monitoring configuration from the JSON file, falling back to defaults if not found.

---

##### `validate_config(config: Dict[str, Any]) -> bool`

Validates monitoring configuration.

**Parameters**:
- `config` (Dict[str, Any]): Configuration to validate

**Returns**:
- bool: True if configuration is valid

**Description**:
Validates the monitoring configuration structure by checking for required keys and validating weight values.

---

##### `get_defaults() -> Dict[str, Any]`

Gets default monitoring configuration.

**Returns**:
- Dict[str, Any]: Default configuration

**Description**:
Provides default values for the monitoring configuration when the actual configuration is missing or invalid.

**Default Configuration**:
```python
{
    "MONITORING_WEIGHTS": {
        "defect": 0.4,
        "yield": 0.4,
        "uniformity": 0.2,
    },
}
```

## Constants and Data Structures

### Metric Types

The monitoring system manages three types of metrics:

- `"efficiency"`: Efficiency-related metrics
- `"quality"`: Quality-related metrics
- `"resources"`: Resource utilization metrics

### Metrics DataFrames

Each metrics category is stored in a pandas DataFrame with the following structure:

**Efficiency Metrics DataFrame**:

| Column | Type | Description |
|--------|------|-------------|
| batch_id | str | Batch identifier |
| timestamp | datetime | Timestamp when metric was recorded |
| output_quantity | float | Amount of product produced |
| cycle_time | float | Production cycle duration |
| energy_consumption | float | Energy used in production |
| production_rate | float | Output quantity per unit time |
| energy_efficiency | float | Output quantity per unit energy |

**Quality Metrics DataFrame**:

| Column | Type | Description |
|--------|------|-------------|
| batch_id | str | Batch identifier |
| timestamp | datetime | Timestamp when metric was recorded |
| defect_rate | float | Percentage of defective products |
| yield_rate | float | Production yield percentage |
| uniformity_score | float | Product uniformity measure |
| quality_score | float | Composite quality score |

**Resource Metrics DataFrame**:

| Column | Type | Description |
|--------|------|-------------|
| batch_id | str | Batch identifier |
| timestamp | datetime | Timestamp when metric was recorded |
| material_consumption | float | Amount of raw materials used |
| water_usage | float | Volume of water consumed |
| waste_generated | float | Amount of waste produced |
| resource_efficiency | float | Resource utilization efficiency |

### Configuration Structure

The monitoring system's configuration has the following structure:

```json
{
    "MONITORING_WEIGHTS": {
        "defect": 0.4,
        "yield": 0.4,
        "uniformity": 0.2
    }
}
```

## Integration with Other Components

### Results Manager Integration

The monitoring system uses the ResultsManager for file and path management:

```python
from circman5.utils.results_manager import results_manager

def save_metrics(metric_type: str, save_path: Optional[Path] = None) -> None:
    """Save metrics to appropriate location."""
    # Implementation using results_manager to handle paths and saving
```

### Constants Service Integration

The monitoring system accesses configuration through the ConstantsService:

```python
from circman5.adapters.services.constants_service import ConstantsService

def __init__(self):
    """Initialize monitoring system with empty metrics storage."""
    # ...
    self.constants = ConstantsService()
    # Later used to access configuration
    weights = self.constants.get_constant("impact_factors", "MONITORING_WEIGHTS")
```

## Usage Examples

### Basic Usage

```python
from circman5.monitoring import ManufacturingMonitor

# Create monitor instance
monitor = ManufacturingMonitor()

# Start monitoring a batch
batch_id = "BATCH_2025_001"
monitor.start_batch_monitoring(batch_id)

# Record metrics
efficiency_metrics = monitor.record_efficiency_metrics(
    output_quantity=100.0,
    cycle_time=60.0,
    energy_consumption=500.0
)

quality_metrics = monitor.record_quality_metrics(
    defect_rate=2.5,
    yield_rate=97.5,
    uniformity_score=95.0
)

resource_metrics = monitor.record_resource_metrics(
    material_consumption=1000.0,
    water_usage=500.0,
    waste_generated=50.0
)

# Generate batch summary
summary = monitor.get_batch_summary(batch_id)

# Save metrics
monitor.save_metrics("efficiency")
monitor.save_metrics("quality")
monitor.save_metrics("resources")
```

### Advanced Usage: Continuous Monitoring

```python
import time
from circman5.monitoring import ManufacturingMonitor

def continuous_monitoring(batch_id, duration_minutes, interval_seconds):
    """Continuously monitor a production batch."""
    monitor = ManufacturingMonitor()
    monitor.start_batch_monitoring(batch_id)

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    while time.time() < end_time:
        # Simulate getting production data
        current_output = get_current_output()  # External function
        current_cycle = get_current_cycle()    # External function
        current_energy = get_current_energy()  # External function

        # Record efficiency metrics
        monitor.record_efficiency_metrics(
            output_quantity=current_output,
            cycle_time=current_cycle,
            energy_consumption=current_energy
        )

        # Record other metrics as needed
        # ...

        # Save at regular intervals
        if int(time.time()) % 300 == 0:  # Every 5 minutes
            monitor.save_metrics("efficiency")

        # Wait for next interval
        time.sleep(interval_seconds)

    # Generate final summary
    summary = monitor.get_batch_summary(batch_id)
    return summary
```

### Analyzing Batch Data

```python
import pandas as pd
import matplotlib.pyplot as plt
from circman5.monitoring import ManufacturingMonitor

def analyze_batch_efficiency(monitor, batch_id):
    """Analyze efficiency metrics for a specific batch."""
    # Get efficiency data for the batch
    efficiency_data = monitor.metrics_history["efficiency"]
    batch_data = efficiency_data[efficiency_data["batch_id"] == batch_id]

    # Calculate key statistics
    avg_production_rate = batch_data["production_rate"].mean()
    max_production_rate = batch_data["production_rate"].max()
    total_energy = batch_data["energy_consumption"].sum()

    # Create time series visualization
    plt.figure(figsize=(10, 6))
    plt.plot(batch_data["timestamp"], batch_data["production_rate"], marker='o')
    plt.axhline(y=avg_production_rate, color='r', linestyle='--', label=f'Avg: {avg_production_rate:.2f}')
    plt.title(f"Production Rate - Batch {batch_id}")
    plt.xlabel("Time")
    plt.ylabel("Production Rate")
    plt.legend()
    plt.grid(True)

    # Save the visualization
    plt.savefig(f"batch_{batch_id}_production_rate.png")

    # Return analysis results
    return {
        "avg_production_rate": avg_production_rate,
        "max_production_rate": max_production_rate,
        "total_energy": total_energy,
        "efficiency_variance": batch_data["production_rate"].var()
    }
```

## Exception Handling

The monitoring system implements proper exception handling:

```python
try:
    # Start batch monitoring
    monitor.start_batch_monitoring(batch_id)

    # Record metrics
    monitor.record_efficiency_metrics(
        output_quantity=100.0,
        cycle_time=60.0,
        energy_consumption=500.0
    )

except ValueError as e:
    logger.error(f"Monitoring error: {e}")
    # Handle the error appropriately

except Exception as e:
    logger.error(f"Unexpected error in monitoring: {e}")
    # Perform appropriate error recovery
```

## Threading and Concurrency

The `ManufacturingMonitor` class is not thread-safe by default. When using it in a multi-threaded environment, implement appropriate synchronization:

```python
import threading

class ThreadSafeMonitor:
    """Thread-safe wrapper for ManufacturingMonitor."""

    def __init__(self):
        self.monitor = ManufacturingMonitor()
        self.lock = threading.Lock()

    def start_batch_monitoring(self, batch_id):
        with self.lock:
            return self.monitor.start_batch_monitoring(batch_id)

    def record_efficiency_metrics(self, output_quantity, cycle_time, energy_consumption):
        with self.lock:
            return self.monitor.record_efficiency_metrics(
                output_quantity, cycle_time, energy_consumption
            )

    # Implement other methods similarly
```

## Best Practices

1. **Batch Lifecycle**: Always call `start_batch_monitoring()` before recording metrics for a new batch.

2. **Regular Recording**: Record metrics at consistent intervals for reliable time-series analysis.

3. **Complete Data**: Record all three types of metrics (efficiency, quality, resources) for comprehensive monitoring.

4. **Error Handling**: Implement proper exception handling when using the monitoring system.

5. **Save Regularly**: Save metrics periodically to prevent data loss during long-running processes.

6. **Timestamp Analysis**: Use the automatically recorded timestamps for time-series analysis.

7. **Custom Metrics**: For advanced use cases, consider extending the monitoring system with custom metric types.

## Related Documentation

- [Monitoring Guide](../guides/monitoring_guide.md) - For usage instructions and examples
- [Adapter API Reference](../api/adapter_api_reference.md) - For information on configuration adapters
- [Results Manager Documentation](../api/utilities_api_reference.md) - For information on file and path management
