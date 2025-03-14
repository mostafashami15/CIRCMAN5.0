# Data Processing Pipeline Implementation Guide

## Overview

This guide documents the implementation of enhanced Data Processing capabilities in the CIRCMAN5.0 system, focusing on real-time data handling, streaming data validation, and integration with external data sources.

## Core Components

### Real-time Data Loading

The `load_real_time_data()` method configures data streaming from manufacturing sources:

```python
def load_real_time_data(self, data_source=None, buffer_size=100):
    """
    Real-time data streaming from manufacturing sources.

    This method configures a data streaming system that continuously reads
    manufacturing data from specified sources. It establishes connection
    parameters, buffer configuration, and validation rules for the stream.

    Args:
        data_source: Optional data source specification (connection string, config)
        buffer_size: Size of the data buffer for stream processing

    Returns:
        DataStream: A data stream object for continuous processing
    """
    # Implementation details here...
```

#### Key Features

- Configures real-time data streaming from manufacturing sources
- Uses constants service for default source configuration
- Provides customizable buffer sizes for performance tuning
- Includes comprehensive error handling and logging

#### Usage Example

```python
# Initialize data loader
data_loader = ManufacturingDataLoader()

# Configure default data stream
stream = data_loader.load_real_time_data()

# Configure custom data stream
custom_source = {
    "type": "database",
    "name": "production_db",
    "connection_params": {
        "host": "localhost",
        "port": 5432,
        "username": "production_user",
        "database": "manufacturing"
    }
}
custom_stream = data_loader.load_real_time_data(
    data_source=custom_source,
    buffer_size=200
)

# Use the stream to process data
# (Future implementation would include methods like:)
# for data_point in stream:
#     process_data_point(data_point)
```

### Streaming Data Validation

The `validate_streaming_data()` method ensures quality and consistency of real-time data:

```python
def validate_streaming_data(self, data_point):
    """
    Validate incoming streaming data points.

    This method checks that incoming data points contain required fields
    with valid values, ensuring data quality for real-time analytics.
    Validation includes field presence, data types, and value ranges.

    Args:
        data_point: The data point to validate

    Returns:
        bool: True if valid, False otherwise
    """
    # Implementation details here...
```

#### Key Features

- Validates required fields in data points
- Checks data types for correctness
- Enforces value range constraints
- Provides detailed validation error information

#### Usage Example

```python
# Initialize data loader
data_loader = ManufacturingDataLoader()

# Example data point
data_point = {
    "timestamp": "2025-01-01T10:00:00",
    "batch_id": "BATCH001",
    "input_amount": 100.0,
    "energy_used": 50.0,
    "cycle_time": 30.0
}

# Validate data point
if data_loader.validate_streaming_data(data_point):
    print("Data point is valid")
    # Process valid data
else:
    print("Data point failed validation")
    # Handle invalid data (log, alert, discard)
```

### External Source Integration

The `integrate_external_sources()` method enables connection with external data systems:

```python
def integrate_external_sources(self, source_config):
    """
    Integrate external data sources into the processing pipeline.

    This method establishes connections to external data sources such as
    ERP systems, databases, APIs, or sensor networks, and integrates them
    into the manufacturing analytics pipeline.

    Args:
        source_config: Configuration for external source

    Returns:
        bool: True if integration successful, False otherwise
    """
    # Implementation details here...
```

#### Key Features

- Validates external source configuration
- Establishes connection to external systems
- Registers data handlers for processing incoming data
- Provides error handling and connection management

#### Usage Example

```python
# Initialize data loader
data_loader = ManufacturingDataLoader()

# Configure external database source
db_source = {
    "name": "erp_system",
    "type": "database",
    "connection_params": {
        "host": "erp.example.com",
        "port": 5432,
        "username": "api_user",
        "password": "secure_password",
        "database": "erp_production"
    }
}

# Integrate external source
if data_loader.integrate_external_sources(db_source):
    print(f"Successfully integrated {db_source['name']}")
else:
    print(f"Failed to integrate {db_source['name']}")

# Example API source
api_source = {
    "name": "sensor_api",
    "type": "api",
    "connection_params": {
        "url": "https://sensors.example.com/api/v1",
        "auth_type": "bearer",
        "auth_token": "API_TOKEN_HERE",
        "refresh_interval": 60
    }
}

# Integrate API source
data_loader.integrate_external_sources(api_source)
```

## Integration Patterns

### Data Source Configuration

External data sources follow a standard configuration structure:

```python
source_config = {
    "name": "source_name",       # Unique identifier
    "type": "source_type",       # database, api, file, sensor
    "connection_params": {
        # Type-specific parameters
    }
}
```

Supported source types and their parameters:

1. **Database Sources**:
   ```python
   {
       "type": "database",
       "connection_params": {
           "host": "hostname",
           "port": 5432,
           "username": "user",
           "password": "password",
           "database": "db_name"
       }
   }
   ```

2. **API Sources**:
   ```python
   {
       "type": "api",
       "connection_params": {
           "url": "https://api.example.com/endpoint",
           "auth_type": "bearer|basic|apikey",
           "auth_token": "token_value"  # or username/password for basic
       }
   }
   ```

3. **File Sources**:
   ```python
   {
       "type": "file",
       "connection_params": {
           "path": "/path/to/file",
           "format": "csv|json|xml",
           "watch": true  # Monitor for changes
       }
   }
   ```

4. **Sensor Sources**:
   ```python
   {
       "type": "sensor",
       "connection_params": {
           "device_id": "sensor123",
           "protocol": "mqtt|modbus|opc-ua",
           "address": "device_address"
       }
   }
   ```

### Data Stream Processing

Real-time data streams follow a common processing pattern:

1. **Stream Configuration**: Set up connection and buffer parameters
2. **Data Reception**: Continuously receive data from sources
3. **Validation**: Apply validation rules to incoming data
4. **Transformation**: Convert data to standard format
5. **Processing**: Apply analytics to processed data
6. **Storage**: Archive processed data using results_manager

### Results Management

All processed data follows the results_manager pattern for consistent archiving:

```python
# Example saving processed data
processed_data_df = pd.DataFrame(processed_data)
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"processed_data_{timestamp_str}.csv"
processed_data_df.to_csv(filename, index=False)
results_manager.save_file(Path(filename), "processed_data")
Path(filename).unlink()  # Clean up temporary file
```

## Extension Points

The implementation includes several extension points for future development:

1. **Advanced Data Stream Implementation**: Replace the configuration-only approach with a full streaming implementation
2. **Real-time Data Transformation**: Add data transformation capabilities to the streaming pipeline
3. **Data Quality Metrics**: Implement comprehensive data quality monitoring
4. **Schema Evolution**: Support for evolving data schemas over time

## Testing

Comprehensive unit tests verify the functionality of all components:

```python
def test_validate_streaming_data(self, data_loader):
    """Test validation of streaming data points."""
    # Valid data point
    valid_data = {
        "timestamp": "2025-01-01T10:00:00",
        "batch_id": "BATCH001",
        "input_amount": 100.0,
        "energy_used": 50.0,
        "cycle_time": 30.0
    }
    assert data_loader.validate_streaming_data(valid_data) is True

    # Invalid - missing fields
    invalid_data1 = {
        "timestamp": "2025-01-01T10:00:00",
        "batch_id": "BATCH001",
        # Missing required fields
    }
    assert data_loader.validate_streaming_data(invalid_data1) is False
```

Run tests using pytest:

```bash
pytest tests/unit/manufacturing/test_data_loader.py
```

## Best Practices

1. **Comprehensive Validation**: Always validate incoming data before processing
2. **Secure Credentials**: Store connection credentials securely, ideally in environment variables
3. **Error Resilience**: Implement retry logic for external connections
4. **Performance Monitoring**: Track stream performance metrics (latency, throughput)
5. **Graceful Degradation**: Handle external source unavailability gracefully

## Data Validation Rules

### Required Fields

All streaming data points must include these fields:

- `timestamp`: When the data was recorded
- `batch_id`: Unique identifier for the manufacturing batch
- `input_amount`: Amount of input material
- `energy_used`: Energy consumption
- `cycle_time`: Process cycle time

### Type Constraints

Fields must conform to these type constraints:

- `timestamp`: ISO-8601 timestamp string (or numeric timestamp)
- `batch_id`: String
- `input_amount`: Positive number (float or integer)
- `energy_used`: Non-negative number (float or integer)
- `cycle_time`: Positive number (float or integer)

### Value Range Constraints

These value ranges ensure reasonable data:

- `input_amount`: Must be positive
- `energy_used`: Must be non-negative
- `cycle_time`: Must be positive and reasonable (typically 5-300 seconds)

## Troubleshooting

### Common Issues

1. **External Source Connection Failures**:
   - Issue: "Failed to connect to external source"
   - Solutions:
     - Verify network connectivity
     - Check credentials
     - Ensure source system is available
     - Review firewall settings

2. **Data Validation Failures**:
   - Issue: "Data validation failed for field X"
   - Solutions:
     - Check data types and formats
     - Verify required fields are present
     - Ensure values are in acceptable ranges

3. **Performance Bottlenecks**:
   - Issue: "Slow data processing"
   - Solutions:
     - Increase buffer size for bursty sources
     - Optimize validation logic
     - Consider batch processing where appropriate
     - Monitor system resource usage

## Future Development Roadmap

1. **Advanced Streaming Implementation**: Full-featured data streaming system with backpressure handling
2. **Data Transformation Pipeline**: Configurable data transformation stages for streaming data
3. **Data Quality Monitoring**: Real-time monitoring of data quality metrics
4. **Schema Registry**: Central schema registry for managing evolving data structures
5. **Data Lineage Tracking**: Tracking data provenance through processing pipeline
