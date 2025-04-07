# Monitoring Guide

## Introduction

The CIRCMAN5.0 Monitoring System provides comprehensive tools for tracking, analyzing, and reporting on manufacturing performance metrics in photovoltaic (PV) production. This guide explains how to use and configure the monitoring system to track efficiency, quality, and resource utilization throughout the manufacturing process.

## Overview

The Monitoring System is designed to:

1. Track key performance indicators (KPIs) in real-time
2. Organize metrics by manufacturing batch
3. Calculate derived performance metrics
4. Generate batch summaries and reports
5. Persist metrics for historical analysis
6. Support quality control and process optimization

## Architecture

The Monitoring System consists of the following components:

```
+---------------------+      +------------------------+
|                     |      |                        |
| Manufacturing       |      | Configuration System   |
| Components          +----->+ (Constants Service)    |
|                     |      |                        |
+----------+----------+      +------------------------+
           |
           v
+----------+----------+      +------------------------+
|                     |      |                        |
| Manufacturing       +----->+ Results Manager        |
| Monitor             |      | (File and Path Mgmt)   |
|                     |      |                        |
+---------------------+      +------------------------+
```

## Getting Started

### Basic Usage Pattern

The core usage pattern for the monitoring system follows these steps:

1. Create a monitor instance
2. Begin monitoring a batch
3. Record metrics throughout the manufacturing process
4. Generate summaries and save metrics

Here's a simple example:

```python
from circman5.monitoring import ManufacturingMonitor

# Create monitor instance
monitor = ManufacturingMonitor()

# Start monitoring a batch
batch_id = "BATCH_2025_03_15_001"
monitor.start_batch_monitoring(batch_id)

# Record efficiency metrics
efficiency_metrics = monitor.record_efficiency_metrics(
    output_quantity=1500.0,  # Units produced
    cycle_time=120.0,        # Minutes
    energy_consumption=750.0  # kWh
)

# Record quality metrics
quality_metrics = monitor.record_quality_metrics(
    defect_rate=2.5,          # Percentage defective
    yield_rate=97.5,          # Percentage yield
    uniformity_score=95.0     # Uniformity measure (0-100)
)

# Record resource utilization
resource_metrics = monitor.record_resource_metrics(
    material_consumption=1000.0,  # kg of material
    water_usage=500.0,            # liters
    waste_generated=50.0          # kg of waste
)

# Generate batch summary
summary = monitor.get_batch_summary(batch_id)

# Save metrics for later analysis
monitor.save_metrics("efficiency")
monitor.save_metrics("quality")
monitor.save_metrics("resources")
```

## Metric Types

The monitoring system tracks three main categories of metrics:

### 1. Efficiency Metrics

Efficiency metrics track production throughput and energy efficiency:

- **Output Quantity**: Amount of product produced (units)
- **Cycle Time**: Production cycle duration (minutes)
- **Energy Consumption**: Energy used in production (kWh)
- **Production Rate**: Calculated as Output Quantity / Cycle Time
- **Energy Efficiency**: Calculated as Output Quantity / Energy Consumption

### 2. Quality Metrics

Quality metrics track product quality and production yield:

- **Defect Rate**: Percentage of defective products
- **Yield Rate**: Percentage of input material successfully converted to product
- **Uniformity Score**: Measure of product consistency (0-100)
- **Quality Score**: Composite score calculated from the above metrics

### 3. Resource Metrics

Resource metrics track material and resource usage:

- **Material Consumption**: Amount of raw materials used (kg)
- **Water Usage**: Volume of water consumed (liters)
- **Waste Generated**: Amount of waste produced (kg)
- **Resource Efficiency**: Calculated as (Material Consumption - Waste Generated) / Material Consumption

## Configuration

The monitoring system's behavior can be customized through configuration settings using the Adapter System.

### Quality Score Weights

The quality score calculation weights can be configured:

```json
{
    "MONITORING_WEIGHTS": {
        "defect": 0.4,
        "yield": 0.4,
        "uniformity": 0.2
    }
}
```

This configuration is stored in `monitoring.json` and accessed through the `MonitoringAdapter`.

### Custom Configuration

To customize the monitoring system configuration:

1. Modify the `monitoring.json` file with your desired weights
2. Alternatively, create a custom adapter to load configuration from other sources
3. Update the `MONITORING_WEIGHTS` section to adjust how the quality score is calculated

## Detailed Usage Guide

### Monitoring Batch Production

When a new production batch starts:

```python
# Start monitoring when batch begins
batch_id = "BATCH_2025_Q1_001"
monitor.start_batch_monitoring(batch_id)
```

This initializes tracking for the specified batch ID. Metrics recorded after this call will be associated with this batch.

### Recording Metrics During Production

Throughout the production process, record metrics at appropriate intervals:

```python
# Record metrics at key production stages
def record_production_stage(stage_name, output, cycle_time, energy):
    """Record metrics for a production stage."""
    print(f"Recording metrics for stage: {stage_name}")

    # Record efficiency metrics
    monitor.record_efficiency_metrics(
        output_quantity=output,
        cycle_time=cycle_time,
        energy_consumption=energy
    )

# Record at different production stages
record_production_stage("Cutting", 1200, 45.0, 250.0)
record_production_stage("Assembly", 1150, 60.0, 300.0)
record_production_stage("Testing", 1100, 30.0, 150.0)
```

### Performing Quality Checks

Record quality metrics during quality control procedures:

```python
# Record quality after inspection
def record_quality_inspection(inspection_point, defect_rate, yield_rate, uniformity):
    """Record quality metrics after inspection."""
    print(f"Recording quality at: {inspection_point}")

    monitor.record_quality_metrics(
        defect_rate=defect_rate,
        yield_rate=yield_rate,
        uniformity_score=uniformity
    )

# Record at quality inspection points
record_quality_inspection("Initial", 3.5, 96.5, 92.0)
record_quality_inspection("Mid-process", 2.8, 97.2, 94.0)
record_quality_inspection("Final", 2.2, 97.8, 95.5)
```

### Tracking Resource Usage

Monitor resource consumption throughout production:

```python
# Track resource usage
def track_resources(station, material_used, water_used, waste_produced):
    """Track resource usage at a production station."""
    print(f"Tracking resources at: {station}")

    monitor.record_resource_metrics(
        material_consumption=material_used,
        water_usage=water_used,
        waste_generated=waste_produced
    )

# Track resources at different stations
track_resources("Raw Materials", 500.0, 200.0, 25.0)
track_resources("Processing", 300.0, 150.0, 15.0)
track_resources("Finishing", 200.0, 100.0, 10.0)
```

### Generating Batch Reports

At the end of the production batch, generate summary reports:

```python
# Generate comprehensive batch summary
summary = monitor.get_batch_summary(batch_id)

# Print summary information
print(f"Batch {batch_id} Summary:")
print(f"Average Production Rate: {summary['efficiency']['avg_production_rate']:.2f} units/min")
print(f"Final Yield Rate: {summary['quality']['final_yield_rate']:.1f}%")
print(f"Resource Efficiency: {summary['resources']['avg_resource_efficiency']:.2f}")
```

### Saving Metrics Data

Save metrics for long-term storage and analysis:

```python
# Save all metric types
for metric_type in ["efficiency", "quality", "resources"]:
    monitor.save_metrics(metric_type)
    print(f"Saved {metric_type} metrics.")
```

## Integration with Other Systems

### Digital Twin Integration

The monitoring system can provide performance data to the Digital Twin system:

```python
from circman5.monitoring import ManufacturingMonitor
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwinCore

# Create instances
monitor = ManufacturingMonitor()
digital_twin = DigitalTwinCore()

# Register monitor with digital twin
digital_twin.register_data_source("performance_monitor", monitor)

# Update digital twin with monitoring data
def update_twin_with_monitoring_data():
    """Send latest monitoring data to digital twin."""
    latest_efficiency = monitor.metrics_history["efficiency"].iloc[-1]
    latest_quality = monitor.metrics_history["quality"].iloc[-1]

    # Update twin state with monitoring data
    digital_twin.update_state({
        "efficiency": {
            "production_rate": latest_efficiency["production_rate"],
            "energy_efficiency": latest_efficiency["energy_efficiency"]
        },
        "quality": {
            "defect_rate": latest_quality["defect_rate"],
            "quality_score": latest_quality["quality_score"]
        }
    })
```

### Visualization Integration

Integrate monitoring data with visualization tools:

```python
import matplotlib.pyplot as plt
import pandas as pd

def visualize_production_metrics(monitor, batch_id):
    """Create visualizations of production metrics."""
    # Filter metrics for specific batch
    efficiency_data = monitor.metrics_history["efficiency"]
    batch_data = efficiency_data[efficiency_data["batch_id"] == batch_id]

    # Create time series plot
    plt.figure(figsize=(10, 6))
    plt.plot(batch_data["timestamp"], batch_data["production_rate"], marker='o')
    plt.title(f"Production Rate Over Time - Batch {batch_id}")
    plt.xlabel("Time")
    plt.ylabel("Production Rate (units/min)")
    plt.grid(True)

    # Save visualization
    plt.savefig(f"batch_{batch_id}_production_rate.png")
```

## Advanced Usage

### Comparative Analysis

Compare metrics across multiple batches:

```python
def compare_batches(monitor, batch_ids):
    """Compare efficiency across multiple batches."""
    comparison = {}

    for batch_id in batch_ids:
        # Get batch summary
        summary = monitor.get_batch_summary(batch_id)

        # Store key metrics
        comparison[batch_id] = {
            "production_rate": summary["efficiency"]["avg_production_rate"],
            "energy_efficiency": summary["efficiency"]["avg_energy_efficiency"],
            "defect_rate": summary["quality"]["avg_defect_rate"],
            "resource_efficiency": summary["resources"]["avg_resource_efficiency"]
        }

    # Convert to DataFrame for easy comparison
    comparison_df = pd.DataFrame(comparison).T

    return comparison_df
```

### Trend Analysis

Analyze trends in monitoring data:

```python
def analyze_trends(monitor, metric_type, column, last_n_batches=10):
    """Analyze trends in a specific metric."""
    # Get data for the metric type
    data = monitor.metrics_history[metric_type]

    # Get unique batches, sorted by timestamp
    batches = data.sort_values("timestamp")["batch_id"].unique()

    # Take the last n batches
    recent_batches = batches[-last_n_batches:]

    # Create a list to store trend data
    trend_data = []

    # Calculate average for each batch
    for batch in recent_batches:
        batch_data = data[data["batch_id"] == batch]
        avg_value = batch_data[column].mean()

        trend_data.append({
            "batch_id": batch,
            "avg_value": avg_value
        })

    return pd.DataFrame(trend_data)
```

## Troubleshooting

### Common Issues

**Issue:** Metrics not being recorded correctly.
**Solution:** Ensure you've called `start_batch_monitoring()` with a valid batch ID before recording metrics.

**Issue:** Quality score calculation seems incorrect.
**Solution:** Check the monitoring configuration to ensure weights are properly set in `monitoring.json`.

**Issue:** Error when saving metrics.
**Solution:** Ensure the results_manager is properly configured and has write access to the target directories.

**Issue:** Metrics history is empty.
**Solution:** Verify that metrics are being recorded and that the batch ID is consistent across recordings.

## Best Practices

1. **Consistent Batch IDs**: Use a consistent naming convention for batch IDs to facilitate easy tracking and comparison.

2. **Regular Metrics Recording**: Record metrics at regular intervals throughout the manufacturing process.

3. **Multiple Metric Types**: Always record all three types of metrics (efficiency, quality, and resources) for complete monitoring.

4. **Saving Frequency**: Save metrics at logical breakpoints in the manufacturing process, not just at the end.

5. **Batch Summaries**: Generate batch summaries for each completed batch to maintain historical performance records.

6. **Configuration Review**: Periodically review monitoring configuration to ensure weights and thresholds reflect current priorities.

7. **Integration with Dashboard**: Integrate monitoring with real-time dashboards for immediate visibility.

## Related Documentation

- [Adapter API Reference](../api/adapter_api_reference.md) - For details on configuration adapters
- [Results Manager Documentation](../api/utilities_api_reference.md) - For details on file and path management
- [Monitoring API Reference](../api/monitoring_api_reference.md) - For complete API details
- [Digital Twin Integration Guide](../implementation/dt_integration_guide.md) - For integration with Digital Twin
