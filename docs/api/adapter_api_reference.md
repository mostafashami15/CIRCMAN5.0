# Adapter API Reference

## Overview

This document provides comprehensive API reference for the CIRCMAN5.0 Adapter System, which offers a standardized approach to configuration management throughout the application. The adapter pattern enables decoupled, modular, and testable configuration handling for all system components.

## Core Components

### ConfigAdapterBase

`ConfigAdapterBase` is an abstract base class that defines the interface all configuration adapters must implement.

**Module**: `src.circman5.adapters.base.adapter_base`

```python
class ConfigAdapterBase(ABC):
    """Base interface for configuration adapters."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize adapter with optional config path.

        Args:
            config_path: Optional path to configuration file
        """
```

#### Methods

##### `load_config() -> Dict[str, Any]`

Loads configuration from source.

**Returns**:
- `Dict[str, Any]`: Configuration dictionary

**Raises**:
- `FileNotFoundError`: If config file not found
- `ValueError`: If config is invalid

**Description**:
Abstract method that must be implemented by concrete adapters to load configuration from their specific source.

---

##### `validate_config(config: Dict[str, Any]) -> bool`

Validates configuration structure.

**Parameters**:
- `config` (Dict[str, Any]): Configuration dictionary to validate

**Returns**:
- `bool`: True if configuration is valid

**Description**:
Abstract method that must be implemented by concrete adapters to validate their specific configuration structure.

---

##### `get_defaults() -> Dict[str, Any]`

Gets default configuration values.

**Returns**:
- `Dict[str, Any]`: Default configuration dictionary

**Description**:
Abstract method that must be implemented by concrete adapters to provide default configuration when the actual configuration is missing or invalid.

---

##### `_load_json_config(path: Path) -> Dict[str, Any]`

Helper method to load JSON configuration.

**Parameters**:
- `path` (Path): Path to JSON config file

**Returns**:
- `Dict[str, Any]`: Loaded configuration

**Raises**:
- `FileNotFoundError`: If file doesn't exist
- `json.JSONDecodeError`: If invalid JSON

**Description**:
Utility method to load and parse JSON configuration files with proper error handling.

---

### ConfigurationManager

`ConfigurationManager` manages configuration loading and validation across different adapters.

**Module**: `src.circman5.adapters.base.config_manager`

```python
class ConfigurationManager:
    """Manages configuration loading and validation across different adapters."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional base path for configuration files
        """
```

#### Methods

##### `register_adapter(name: str, adapter: ConfigAdapterBase) -> None`

Registers a new configuration adapter.

**Parameters**:
- `name` (str): Unique identifier for the adapter
- `adapter` (ConfigAdapterBase): Adapter instance to register

**Raises**:
- `ValueError`: If adapter name already exists

**Description**:
Registers an adapter instance with a unique name for later use.

---

##### `load_config(adapter_name: str) -> Dict[str, Any]`

Loads configuration using specified adapter.

**Parameters**:
- `adapter_name` (str): Name of the adapter to use

**Returns**:
- `Dict[str, Any]`: Loaded configuration

**Raises**:
- `ValueError`: If adapter not found
- Passes through any exceptions from the adapter's `load_config` method

**Description**:
Uses the specified adapter to load its configuration, validates it, and caches the result.

---

##### `get_config(adapter_name: str) -> Dict[str, Any]`

Gets loaded configuration for an adapter.

**Parameters**:
- `adapter_name` (str): Name of adapter

**Returns**:
- `Dict[str, Any]`: Current configuration

**Raises**:
- `ValueError`: If adapter not found or config not loaded

**Description**:
Retrieves the cached configuration for the specified adapter, loading it first if necessary.

---

##### `reload_all() -> None`

Reloads all registered configurations.

**Description**:
Forces reloading of all registered adapters' configurations.

---

##### `get_adapter(name: str) -> ConfigAdapterBase`

Gets registered adapter by name.

**Parameters**:
- `name` (str): Adapter name

**Returns**:
- `ConfigAdapterBase`: Registered adapter

**Raises**:
- `ValueError`: If adapter not found

**Description**:
Retrieves a registered adapter instance by name.

---

### ConstantsService

`ConstantsService` provides a centralized access point for all system constants as a singleton.

**Module**: `src.circman5.adapters.services.constants_service`

```python
class ConstantsService:
    """Service for centralized access to all system constants."""

    # Singleton implementation
    _instance = None

    def __new__(cls):
        # Singleton pattern implementation
        pass

    def __init__(self):
        """Initialize the constants service."""
        pass
```

#### Methods

##### `get_manufacturing_constants() -> Dict[str, Any]`

Gets manufacturing constants.

**Returns**:
- `Dict[str, Any]`: Manufacturing configuration dictionary

**Description**:
Retrieves the complete manufacturing configuration.

---

##### `get_impact_factors() -> Dict[str, Any]`

Gets impact factors constants.

**Returns**:
- `Dict[str, Any]`: Impact factors configuration dictionary

**Description**:
Retrieves the complete impact factors configuration.

---

##### `get_optimization_config() -> Dict[str, Any]`

Gets optimization configuration.

**Returns**:
- `Dict[str, Any]`: Optimization configuration dictionary

**Description**:
Retrieves the complete optimization configuration.

---

##### `get_visualization_config() -> Dict[str, Any]`

Gets visualization configuration.

**Returns**:
- `Dict[str, Any]`: Visualization configuration dictionary

**Description**:
Retrieves the complete visualization configuration.

---

##### `get_monitoring_config() -> Dict[str, Any]`

Gets monitoring configuration.

**Returns**:
- `Dict[str, Any]`: Monitoring configuration dictionary

**Description**:
Retrieves the complete monitoring configuration.

---

##### `get_digital_twin_config() -> Dict[str, Any]`

Gets digital twin configuration.

**Returns**:
- `Dict[str, Any]`: Digital twin configuration dictionary

**Description**:
Retrieves the complete digital twin configuration.

---

##### `get_constant(adapter: str, key: str) -> Any`

Gets specific constant by adapter and key.

**Parameters**:
- `adapter` (str): Adapter name
- `key` (str): Configuration key

**Returns**:
- `Any`: Configuration value

**Raises**:
- `KeyError`: If key not found

**Description**:
Retrieves a specific configuration value from the specified adapter's configuration.

---

##### `reload_configs() -> None`

Reloads all configurations.

**Description**:
Forces all configurations to be reloaded from their sources.

---

##### `_register_adapters() -> None`

Registers all configuration adapters.

**Description**:
Internal method that registers all standard adapters with the configuration manager.

---

##### `_load_all_configs() -> None`

Loads all configurations.

**Description**:
Internal method that loads all configurations from all registered adapters.

---

## Domain-Specific Adapters

### DigitalTwinAdapter

`DigitalTwinAdapter` is an adapter for Digital Twin configuration.

**Module**: `src.circman5.adapters.config.digital_twin`

```python
class DigitalTwinAdapter(ConfigAdapterBase):
    """Adapter for Digital Twin configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Digital Twin adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "digital_twin.json"
        )
```

#### Methods

##### `load_config() -> Dict[str, Any]`

Loads Digital Twin configuration.

**Returns**:
- `Dict[str, Any]`: Digital Twin configuration

**Raises**:
- `FileNotFoundError`: If config file not found
- `ValueError`: If config is invalid

**Description**:
Loads the Digital Twin configuration from the JSON file, falling back to defaults if not found.

---

##### `validate_config(config: Dict[str, Any]) -> bool`

Validates Digital Twin configuration.

**Parameters**:
- `config` (Dict[str, Any]): Configuration to validate

**Returns**:
- `bool`: True if configuration is valid

**Description**:
Validates the Digital Twin configuration structure by checking for required sections and parameters.

---

##### `get_defaults() -> Dict[str, Any]`

Gets default Digital Twin configuration.

**Returns**:
- `Dict[str, Any]`: Default configuration

**Description**:
Provides default values for the Digital Twin configuration when the actual configuration is missing or invalid.

---

## Usage Examples

### Basic Adapter Usage

```python
# Create a configuration adapter
from src.circman5.adapters.config.digital_twin import DigitalTwinAdapter

adapter = DigitalTwinAdapter()
config = adapter.load_config()

# Access configuration values
update_frequency = config.get("DIGITAL_TWIN_CONFIG", {}).get("update_frequency", 1.0)
```

### Configuration Manager Usage

```python
# Create and use configuration manager
from src.circman5.adapters.base.config_manager import ConfigurationManager
from src.circman5.adapters.config.digital_twin import DigitalTwinAdapter

# Create manager
manager = ConfigurationManager()

# Register adapter
manager.register_adapter("digital_twin", DigitalTwinAdapter())

# Load and access configuration
config = manager.load_config("digital_twin")
```

### Constants Service Usage

```python
# Using the constants service
from src.circman5.adapters.services.constants_service import ConstantsService

# Get or create singleton instance
constants = ConstantsService()

# Get complete configuration
dt_config = constants.get_digital_twin_config()

# Get specific constant with direct access
update_frequency = constants.get_constant(
    "digital_twin", "DIGITAL_TWIN_CONFIG"
).get("update_frequency", 1.0)
```

### Custom Adapter Implementation

```python
from pathlib import Path
from typing import Dict, Any, Optional
from src.circman5.adapters.base.adapter_base import ConfigAdapterBase

class CustomAdapter(ConfigAdapterBase):
    """Custom configuration adapter."""

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__(config_path)
        self.config_path = config_path or Path("config/custom_config.json")

    def load_config(self) -> Dict[str, Any]:
        """Load custom configuration."""
        if not self.config_path.exists():
            self.logger.warning(f"Config file not found: {self.config_path}")
            return self.get_defaults()

        return self._load_json_config(self.config_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate custom configuration."""
        required_keys = {"API_KEY", "API_URL", "TIMEOUT"}

        if not all(key in config for key in required_keys):
            self.logger.error(f"Missing required keys: {required_keys - set(config.keys())}")
            return False

        return True

    def get_defaults(self) -> Dict[str, Any]:
        """Get default custom configuration."""
        return {
            "API_KEY": "",
            "API_URL": "https://api.example.com",
            "TIMEOUT": 30,
            "RETRY_COUNT": 3
        }
```

## Integration with Other Components

### Using Adapters in Digital Twin Components

```python
from src.circman5.adapters.services.constants_service import ConstantsService

class DigitalTwinCore:
    """Digital Twin core implementation."""

    def __init__(self):
        """Initialize Digital Twin core."""
        # Get constants service
        self.constants = ConstantsService()

        # Get configuration
        self.config = self.constants.get_digital_twin_config()

        # Extract configuration values with defaults
        dt_config = self.config.get("DIGITAL_TWIN_CONFIG", {})
        self.name = dt_config.get("name", "DefaultTwin")
        self.update_frequency = dt_config.get("update_frequency", 1.0)
        self.history_length = dt_config.get("history_length", 1000)

        # Initialize based on configuration
        self._setup_components()

    def _setup_components(self):
        """Set up twin components based on configuration."""
        pass
```

### Using Adapters in Manufacturing Components

```python
from src.circman5.adapters.services.constants_service import ConstantsService

class ManufacturingAnalyzer:
    """Manufacturing data analyzer."""

    def __init__(self):
        """Initialize analyzer."""
        # Get constants service
        self.constants = ConstantsService()

        # Get manufacturing configuration
        self.config = self.constants.get_manufacturing_constants()

        # Get specific thresholds
        thresholds = self.config.get("THRESHOLDS", {})
        self.temperature_threshold = thresholds.get("TEMPERATURE", 25.0)
        self.energy_threshold = thresholds.get("ENERGY_CONSUMPTION", 100.0)
```

### Using Adapters in Test Components

```python
import unittest
from unittest.mock import patch, MagicMock
from src.circman5.adapters.services.constants_service import ConstantsService

class TestComponent(unittest.TestCase):
    """Test component using configuration."""

    def setUp(self):
        """Set up test case."""
        # Create mock constants service
        self.mock_constants = MagicMock(spec=ConstantsService)

        # Configure mock to return test configuration
        self.mock_constants.get_digital_twin_config.return_value = {
            "DIGITAL_TWIN_CONFIG": {
                "name": "TestTwin",
                "update_frequency": 0.1
            }
        }

        # Inject mock into your test subject
        with patch('src.circman5.adapters.services.constants_service.ConstantsService') as mock_service:
            mock_service.return_value = self.mock_constants

            # Create component under test with mock config
            self.component = YourComponent()

    def test_component_initialization(self):
        """Test component initializes with configuration."""
        # Verify component used configuration
        self.assertEqual(self.component.name, "TestTwin")
        self.assertEqual(self.component.update_frequency, 0.1)
```

## Error Handling

### Configuration Loading Errors

```python
try:
    config = adapter.load_config()
except FileNotFoundError:
    print("Configuration file not found, using defaults")
    config = adapter.get_defaults()
except ValueError as e:
    print(f"Invalid configuration: {e}")
    config = adapter.get_defaults()
```

### Configuration Validation

```python
# Load configuration
config = adapter.load_config()

# Check if valid
if not adapter.validate_config(config):
    print("Configuration failed validation, using defaults")
    config = adapter.get_defaults()
```

### Missing Keys

```python
# Get a key with a default fallback value
value = config.get("SECTION", {}).get("key", default_value)

# Or with more detailed error handling
try:
    section = config["SECTION"]
    value = section["key"]
except KeyError as e:
    print(f"Missing configuration key: {e}")
    value = default_value
```

## Best Practices

### 1. Use the Constants Service

Always access configuration through the `ConstantsService` singleton:

```python
from src.circman5.adapters.services.constants_service import ConstantsService

constants = ConstantsService()
config = constants.get_digital_twin_config()
```

### 2. Provide Default Values

Always provide sensible defaults when accessing specific configuration keys:

```python
# Good: Nested get with default
timeout = config.get("CONNECTION", {}).get("timeout", 30)

# Avoid: Direct access without fallback
# timeout = config["CONNECTION"]["timeout"]  # Might raise KeyError
```

### 3. Validate Configurations

Always validate loaded configurations before use:

```python
config = adapter.load_config()
if not adapter.validate_config(config):
    logger.warning("Configuration failed validation, using defaults")
    config = adapter.get_defaults()
```

### 4. Use Strongly Typed Configurations

Convert raw configuration values to appropriate types:

```python
# Convert to appropriate types
timeout = int(config.get("timeout", "30"))
retry_enabled = config.get("retry_enabled", "true").lower() == "true"
log_level = getattr(logging, config.get("log_level", "INFO").upper())
```

### 5. Handle Missing Configurations

Handle the case when configurations are missing:

```python
try:
    config = constants.get_config("feature")
except ValueError:
    logger.warning("Feature configuration not found, using defaults")
    config = {
        "ENABLED": False,
        "TIMEOUT": 30
    }
```
