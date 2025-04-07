# Data Schemas Reference

## Introduction

This reference document provides a comprehensive overview of the data schemas and types used throughout the CIRCMAN5.0 system. These schemas define the structure, data types, and relationships for various manufacturing data entities, ensuring data consistency and enabling validation across the system.

## Schema Types

CIRCMAN5.0 uses two main approaches for defining data schemas:

1. **Pandas DataFrame Schemas**: Used for tabular data processing, defined in `schemas.py`
2. **Python Dataclasses**: Used for structured object representations, defined in `data_types.py`

Both schema types serve distinct purposes in the system and are used in different contexts.

## Pandas DataFrame Schemas

DataFrame schemas define the expected column names and data types for tabular data used in manufacturing analytics. These schemas are defined in `src/circman5/manufacturing/schemas.py`.

### Schema Definition Pattern

Each schema is defined as a dictionary mapping column names to pandas data types:

```python
SchemaType = Dict[str, PandasDtype]

EXAMPLE_SCHEMA: SchemaType = {
    "column_name": pandas_dtype,
    # ...
}
```

Where `PandasDtype` is a union type representing valid pandas data types:

```python
PandasDtype = Union[
    ExtensionDtype,
    Type[str],
    Type[float],
    Literal["datetime64[ns]", "string", "float64"],
]
```

### Production Data Schema

The `PRODUCTION_SCHEMA` defines the structure for production process data:

```python
PRODUCTION_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "batch_id": str,
    "product_type": str,
    "production_line": str,
    "output_amount": float,
    "cycle_time": float,
    "yield_rate": float,
    "input_amount": float,
    "output_amount": float,
    "energy_used": float,
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime64[ns] | Time when data was recorded |
| `batch_id` | str | Unique identifier for production batch |
| `product_type` | str | Type of product being manufactured |
| `production_line` | str | Identifier for production line |
| `output_amount` | float | Quantity of product produced |
| `cycle_time` | float | Duration of production cycle |
| `yield_rate` | float | Percentage of input converted to output |
| `input_amount` | float | Quantity of input material |
| `energy_used` | float | Energy consumption for production |

### Energy Data Schema

The `ENERGY_SCHEMA` defines the structure for energy consumption data:

```python
ENERGY_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "production_line": str,
    "energy_consumption": float,
    "energy_source": str,
    "efficiency_rate": float,
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime64[ns] | Time when data was recorded |
| `production_line` | str | Identifier for production line |
| `energy_consumption` | float | Amount of energy consumed |
| `energy_source` | str | Source of energy (e.g., grid, solar) |
| `efficiency_rate` | float | Energy efficiency rate |

### Quality Data Schema

The `QUALITY_SCHEMA` defines the structure for quality control measurements:

```python
QUALITY_SCHEMA: SchemaType = {
    "batch_id": str,
    "timestamp": "datetime64[ns]",
    "efficiency": float,
    "defect_rate": float,
    "thickness_uniformity": float,
    "visual_inspection": str,
}
```

| Field | Type | Description |
|-------|------|-------------|
| `batch_id` | str | Unique identifier for production batch |
| `timestamp` | datetime64[ns] | Time of quality inspection |
| `efficiency` | float | Efficiency rating (%) |
| `defect_rate` | float | Percentage of defective products |
| `thickness_uniformity` | float | Uniformity measure for PV panels |
| `visual_inspection` | str | Result of visual inspection |

### Material Data Schema

The `MATERIAL_SCHEMA` defines the structure for material flow data:

```python
MATERIAL_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "material_type": str,
    "quantity_used": float,
    "waste_generated": float,
    "recycled_amount": float,
    "batch_id": str,
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime64[ns] | Time when data was recorded |
| `material_type` | str | Type of material |
| `quantity_used` | float | Amount of material used |
| `waste_generated` | float | Amount of waste generated |
| `recycled_amount` | float | Amount of material recycled |
| `batch_id` | str | Batch identifier |

### LCA-specific Schemas

These schemas are specific to Life Cycle Assessment (LCA) analysis:

#### LCA Material Schema

```python
LCA_MATERIAL_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "material_type": str,
    "quantity": float,
    "impact_factor": float,
    "batch_id": str,
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime64[ns] | Time when data was recorded |
| `material_type` | str | Type of material |
| `quantity` | float | Amount of material |
| `impact_factor` | float | Environmental impact factor |
| `batch_id` | str | Batch identifier |

#### LCA Energy Schema

```python
LCA_ENERGY_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "source": str,
    "consumption": float,
    "carbon_intensity": float,
    "batch_id": str,
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime64[ns] | Time when data was recorded |
| `source` | str | Energy source |
| `consumption` | float | Energy consumption |
| `carbon_intensity` | float | Carbon intensity factor |
| `batch_id` | str | Batch identifier |

#### LCA Process Schema

```python
LCA_PROCESS_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "process_step": str,
    "duration": float,
    "impact_factor": float,
    "batch_id": str,
}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime64[ns] | Time when data was recorded |
| `process_step` | str | Manufacturing process step |
| `duration` | float | Duration of process step |
| `impact_factor` | float | Environmental impact factor |
| `batch_id` | str | Batch identifier |

## Python Dataclasses

CIRCMAN5.0 uses Python dataclasses to define structured data objects for various manufacturing entities. These are defined in `src/circman5/utils/data_types.py`.

### BatchData

The `BatchData` class represents information about a manufacturing batch:

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

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_id` | str | (required) | Unique identifier for batch |
| `start_time` | datetime | (required) | Batch start time |
| `stage` | str | (required) | Current production stage |
| `status` | str | (required) | Current batch status |
| `input_material` | str | (required) | Type of input material |
| `input_amount` | float | (required) | Amount of input material |
| `output_amount` | float | 0.0 | Current output amount |
| `yield_rate` | float | 0.0 | Current yield rate |
| `energy_used` | float | 0.0 | Energy consumption |
| `completion_time` | Optional[datetime] | None | Batch completion time |

### QualityData

The `QualityData` class represents quality control measurements:

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

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_id` | str | (required) | Batch identifier |
| `test_time` | datetime | (required) | Time of quality test |
| `efficiency` | float | (required) | Panel efficiency measurement |
| `defect_rate` | float | (required) | Rate of defects |
| `thickness_uniformity` | float | (required) | Panel thickness uniformity |
| `contamination_level` | float | (required) | Contamination level |

### CircularMetrics

The `CircularMetrics` class represents circular economy metrics:

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

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_id` | str | (required) | Batch identifier |
| `recycled_content` | float | (required) | Percentage of recycled content in input |
| `recyclable_output` | float | (required) | Percentage of output that is recyclable |
| `water_reused` | float | (required) | Percentage of water reused |
| `material_efficiency` | float | (required) | Material utilization efficiency |
| `waste_recyclability` | float | 95.0 | Percentage of waste that is recyclable |

## Schema Usage

### DataFrame Schema Validation

The schemas defined in `schemas.py` are primarily used for validating and ensuring the structure of pandas DataFrames:

```python
import pandas as pd
from circman5.manufacturing.schemas import PRODUCTION_SCHEMA

def validate_production_data(data: pd.DataFrame) -> bool:
    """Validate production data against schema."""

    # Check if all required columns exist
    required_columns = set(PRODUCTION_SCHEMA.keys())
    if not required_columns.issubset(set(data.columns)):
        missing = required_columns - set(data.columns)
        print(f"Missing columns: {missing}")
        return False

    # Check column types
    for column, expected_type in PRODUCTION_SCHEMA.items():
        # For columns present in the data
        if column in data.columns:
            # Check type compatibility
            if not pd.api.types.is_dtype_equal(data[column].dtype, expected_type):
                print(f"Column {column} has wrong type: {data[column].dtype} (expected {expected_type})")
                return False

    return True
```

### Applying Schema During Data Loading

Schemas are typically used when loading data to ensure proper structure:

```python
def load_production_data(file_path: str) -> pd.DataFrame:
    """Load production data from CSV file."""

    # Load raw data
    raw_data = pd.read_csv(file_path)

    # Apply schema
    for column, dtype in PRODUCTION_SCHEMA.items():
        if column in raw_data:
            # Convert timestamp columns
            if dtype == "datetime64[ns]" and column in raw_data:
                raw_data[column] = pd.to_datetime(raw_data[column])
            # Convert other columns
            else:
                raw_data[column] = raw_data[column].astype(dtype)

    # Validate against schema
    if not validate_production_data(raw_data):
        raise ValueError("Data does not match production schema")

    return raw_data
```

### Creating and Using Dataclass Instances

Dataclasses provide a structured way to represent manufacturing data entities:

```python
from datetime import datetime
from circman5.utils.data_types import BatchData, QualityData

# Create a new batch
batch = BatchData(
    batch_id="BATCH-2025-003",
    start_time=datetime.now(),
    stage="Initialization",
    status="Active",
    input_material="Silicon Wafer",
    input_amount=1000.0
)

# Update batch information as processing continues
def update_batch_progress(batch: BatchData, output_amount: float, energy_used: float):
    batch.output_amount = output_amount
    batch.energy_used = energy_used
    batch.yield_rate = (output_amount / batch.input_amount) * 100.0

    # Return updated batch
    return batch

# Create quality data for the batch
quality = QualityData(
    batch_id=batch.batch_id,
    test_time=datetime.now(),
    efficiency=21.5,
    defect_rate=1.2,
    thickness_uniformity=98.5,
    contamination_level=0.3
)
```

## Schema Relationships

The various schemas in CIRCMAN5.0 are interrelated, sharing common keys for data integration:

### Common Identifiers

- `batch_id`: Primary key that links data across schemas
- `timestamp`: Temporal key for time-series analysis
- `production_line`: Spatial key for location-specific analysis

### Schema Relationship Diagram

```
                     ┌─────────────────┐
                     │  BATCH_ID       │
                     └───────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
┌──────────▼──────────┐ ┌────▼─────────────┐ ┌─▼──────────────────┐
│  PRODUCTION_SCHEMA  │ │  QUALITY_SCHEMA  │ │  MATERIAL_SCHEMA   │
└──────────┬──────────┘ └────┬─────────────┘ └─┬──────────────────┘
           │                 │                 │
           │                 │                 │
┌──────────▼──────────┐ ┌────▼─────────────┐ ┌─▼──────────────────┐
│   BatchData         │ │   QualityData    │ │   CircularMetrics  │
└─────────────────────┘ └──────────────────┘ └────────────────────┘
```

## Data Validation Patterns

### Basic Validation Functions

Use these patterns to validate data against schemas:

```python
def validate_schema_compliance(data: pd.DataFrame, schema: Dict[str, Any]) -> bool:
    """Validate that a DataFrame complies with a schema."""
    # Check for required columns
    missing_columns = set(schema.keys()) - set(data.columns)
    if missing_columns:
        return False

    # Validate data types
    for column, expected_type in schema.items():
        if not pd.api.types.is_dtype_equal(data[column].dtype, expected_type):
            return False

    return True
```

### Advanced Data Validation

For more complex validation beyond schema structure:

```python
def validate_production_data_values(data: pd.DataFrame) -> bool:
    """Validate production data values for logical consistency."""

    # Ensure no negative values for physical quantities
    for column in ["input_amount", "output_amount", "energy_used"]:
        if (data[column] < 0).any():
            return False

    # Ensure yield rate is between 0 and 100
    if ((data["yield_rate"] < 0) | (data["yield_rate"] > 100)).any():
        return False

    # Ensure output <= input (conservation of mass)
    if (data["output_amount"] > data["input_amount"]).any():
        return False

    return True
```

## Working with Time Series Data

Many schemas include timestamp fields for time series analysis:

```python
def resample_production_data(data: pd.DataFrame, interval: str = "1H") -> pd.DataFrame:
    """Resample production data to a specific time interval."""

    # Ensure data has datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        if "timestamp" in data.columns:
            data = data.set_index("timestamp")
        else:
            raise ValueError("DataFrame has no timestamp column or datetime index")

    # Resample and aggregate
    resampled = data.resample(interval).agg({
        "input_amount": "sum",
        "output_amount": "sum",
        "energy_used": "sum",
        "cycle_time": "mean",
        "yield_rate": "mean"
    })

    return resampled
```

## Schema Extension Patterns

### Extending Existing Schemas

To extend an existing schema with additional fields:

```python
from circman5.manufacturing.schemas import PRODUCTION_SCHEMA

# Extend production schema with additional fields
EXTENDED_PRODUCTION_SCHEMA = {
    **PRODUCTION_SCHEMA,  # Include all original fields
    "operator_id": str,   # Add operator identifier
    "machine_id": str,    # Add machine identifier
    "temperature": float, # Add process temperature
    "pressure": float,    # Add process pressure
}
```

### Creating Custom Schemas

To create a new schema for a specific use case:

```python
# Define a new schema for maintenance data
MAINTENANCE_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "machine_id": str,
    "maintenance_type": str,     # Preventive, Corrective, etc.
    "duration": float,           # Duration in hours
    "parts_replaced": str,       # Comma-separated list of parts
    "technician_id": str,
    "notes": str,
}
```

### Extending Dataclasses

To extend existing dataclasses with additional fields:

```python
from dataclasses import dataclass
from circman5.utils.data_types import BatchData

@dataclass
class EnhancedBatchData(BatchData):
    """Enhanced batch data with additional fields."""

    operator_id: str = ""
    machine_id: str = ""
    temperature: float = 0.0
    pressure: float = 0.0
    humidity: float = 0.0
```

## Best Practices

### Schema Definition

1. **Consistency**: Use consistent naming conventions for related fields across schemas
2. **Documentation**: Document the purpose and units of each field
3. **Types**: Use appropriate data types that match the expected values
4. **Defaults**: Provide sensible defaults for optional fields in dataclasses

### Schema Usage

1. **Validation First**: Always validate data against schemas before processing
2. **Error Handling**: Provide clear error messages when validation fails
3. **Transformation**: Use schemas to guide data transformation and cleaning
4. **Extension**: Extend schemas rather than modifying them to maintain compatibility

### Data Processing

1. **Type Safety**: Use type hints and schema validation to ensure type safety
2. **Error Identification**: Identify and log specific schema violations
3. **Graceful Degradation**: When possible, handle partial schema compliance
4. **Performance**: Consider schema validation performance for large datasets

## Example: Complete Data Loading and Validation

```python
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from circman5.manufacturing.schemas import PRODUCTION_SCHEMA, QUALITY_SCHEMA

def load_and_validate_data(
    production_file: Path,
    quality_file: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load and validate production and quality data files.

    Args:
        production_file: Path to production data CSV
        quality_file: Path to quality data CSV

    Returns:
        Tuple containing:
        - Production DataFrame
        - Quality DataFrame
        - Validation report dictionary
    """
    validation_report = {
        "production": {"valid": False, "errors": []},
        "quality": {"valid": False, "errors": []}
    }

    # Load production data
    try:
        production_data = pd.read_csv(production_file)

        # Convert timestamps
        if "timestamp" in production_data.columns:
            production_data["timestamp"] = pd.to_datetime(production_data["timestamp"])

        # Check schema compliance
        missing_columns = set(PRODUCTION_SCHEMA.keys()) - set(production_data.columns)
        if missing_columns:
            validation_report["production"]["errors"].append(
                f"Missing columns: {missing_columns}"
            )

        # Check data types
        for column, expected_type in PRODUCTION_SCHEMA.items():
            if column in production_data.columns:
                try:
                    production_data[column] = production_data[column].astype(expected_type)
                except Exception as e:
                    validation_report["production"]["errors"].append(
                        f"Column {column} type conversion error: {str(e)}"
                    )

        # Check for negative values
        for column in ["input_amount", "output_amount", "energy_used"]:
            if column in production_data.columns and (production_data[column] < 0).any():
                validation_report["production"]["errors"].append(
                    f"Column {column} contains negative values"
                )

        # Mark as valid if no errors
        validation_report["production"]["valid"] = len(validation_report["production"]["errors"]) == 0

    except Exception as e:
        validation_report["production"]["errors"].append(f"Failed to load file: {str(e)}")
        production_data = pd.DataFrame()

    # Load quality data (similar pattern)
    try:
        quality_data = pd.read_csv(quality_file)
        # Similar validation steps...
        validation_report["quality"]["valid"] = True  # Simplified
    except Exception as e:
        validation_report["quality"]["errors"].append(f"Failed to load file: {str(e)}")
        quality_data = pd.DataFrame()

    return production_data, quality_data, validation_report
```

## Conclusion

This reference document provides a comprehensive overview of the data schemas and types used in CIRCMAN5.0. By following these schema definitions and usage patterns, developers can ensure data consistency and reliability throughout the system.
