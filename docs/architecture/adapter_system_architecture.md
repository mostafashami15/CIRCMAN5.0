# Adapter System Architecture

## 1. Overview

The Adapter System in CIRCMAN5.0 provides a standardized approach to configuration management and component interaction. This architectural pattern decouples configuration sources from their consumers, enhancing modularity and maintainability throughout the system. The adapter architecture enables consistent configuration loading, validation, and access across different subsystems including Digital Twin, Human Interface, Manufacturing Analytics, and Lifecycle Assessment components.

## 2. Architectural Principles

The Adapter System follows these key architectural principles:

1. **Interface Segregation**: Adapter interfaces are focused on specific domains and responsibilities.
2. **Dependency Inversion**: High-level components depend on abstractions, not concrete implementations.
3. **Single Responsibility**: Each adapter handles one specific type of configuration.
4. **Open/Closed Principle**: The system is open for extension but closed for modification.
5. **Fail-Fast Validation**: Configurations are validated immediately during loading to prevent cascading errors.
6. **Centralized Access**: A single service provides access to all configurations.
7. **Default Behaviors**: All adapters provide sensible defaults when configurations are missing.

## 3. System Architecture Diagram

```
+-----------------------+     +------------------------+
|                       |     |                        |
| Consumer Components   |     | Configuration Sources  |
| (Digital Twin,        |     | (JSON Files,           |
|  Human Interface,     |     |  Environment Vars,     |
|  Manufacturing, etc.) |     |  Database, etc.)       |
|                       |     |                        |
+-----------+-----------+     +------------+-----------+
            |                              |
            v                              |
+---------------------------------------------------+
|                                                   |
|                Constants Service                  |
|        (Singleton, Central Access Point)          |
|                                                   |
+---------------------+---------------------------+--+
                      |                           |
                      v                           v
  +-------------------+--+         +-------------+---------------+
  |                      |         |                             |
  | Configuration Manager|         | Domain-Specific Adapters    |
  | (Manages adapters    |<--------+ (Digital Twin Adapter,      |
  |  and configurations) |         |  Manufacturing Adapter,     |
  |                      |         |  Impact Factors Adapter,    |
  +----------------------+         |  etc.)                      |
                                   |                             |
                                   +-----------------------------+
                                               ^
                                               |
                                   +-----------+-------------+
                                   |                         |
                                   | Adapter Base Class      |
                                   | (Abstract Interface)    |
                                   |                         |
                                   +-------------------------+
```

## 4. Core Components

### 4.1 Adapter Base Class

The `ConfigAdapterBase` abstract class defines the interface all configuration adapters must implement. It provides a common foundation for adapter functionality.

#### Key Responsibilities:
- Define standard interface for configuration adapters
- Provide helper methods for common operations
- Ensure consistent error handling and logging
- Establish configuration validation patterns

#### Interfaces:
- `load_config()`: Load configuration from source
- `validate_config()`: Validate configuration structure
- `get_defaults()`: Get default configuration values
- `_load_json_config()`: Helper for JSON loading

### 4.2 Configuration Manager

The `ConfigurationManager` class orchestrates multiple adapters and manages configuration loading and access.

#### Key Responsibilities:
- Register and manage configuration adapters
- Load configurations on demand
- Cache loaded configurations for performance
- Provide controlled access to configurations
- Handle configuration reloading

#### Interfaces:
- `register_adapter()`: Register a new adapter
- `load_config()`: Load configuration using an adapter
- `get_config()`: Get loaded configuration
- `reload_all()`: Reload all configurations
- `get_adapter()`: Get registered adapter

### 4.3 Constants Service

The `ConstantsService` provides a centralized access point for all system constants as a singleton.

#### Key Responsibilities:
- Provide a single point of access for configurations
- Register all system adapters
- Manage adapter lifecycle
- Provide specialized access methods
- Support runtime configuration reloading

#### Interfaces:
- `get_manufacturing_constants()`: Get manufacturing configuration
- `get_impact_factors()`: Get impact factors configuration
- `get_optimization_config()`: Get optimization configuration
- `get_constant()`: Get specific constant by adapter and key
- `reload_configs()`: Reload all configurations

### 4.4 Domain-Specific Adapters

Concrete adapter implementations for different configuration domains.

#### Examples:
- `DigitalTwinAdapter`: Digital twin configuration
- `ManufacturingAdapter`: Manufacturing system configuration
- `ImpactFactorsAdapter`: Environmental impact factors
- `OptimizationAdapter`: AI optimization parameters
- `VisualizationAdapter`: Visualization settings
- `MonitoringAdapter`: System monitoring configuration

## 5. Implementation Details

### 5.1 Adapter Base Class Implementation

The `ConfigAdapterBase` class is an abstract base class that defines the core interface:

```python
class ConfigAdapterBase(ABC):
    """Base interface for configuration adapters."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize adapter with optional config path."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path

    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from source."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        pass

    @abstractmethod
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        pass
```

### 5.2 Configuration Manager Implementation

The `ConfigurationManager` manages the lifecycle of multiple adapters:

```python
class ConfigurationManager:
    """Manages configuration loading and validation across different adapters."""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path or Path("config")
        self.adapters: Dict[str, ConfigAdapterBase] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}

    def register_adapter(self, name: str, adapter: ConfigAdapterBase) -> None:
        """Register a new configuration adapter."""
        if name in self.adapters:
            raise ValueError(f"Adapter already registered: {name}")

        self.adapters[name] = adapter
        self.logger.info(f"Registered adapter: {name}")
```

### 5.3 Constants Service Implementation

The `ConstantsService` implements the Singleton pattern for centralized access:

```python
class ConstantsService:
    """Service for centralized access to all system constants."""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConstantsService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the constants service."""
        if self._initialized:
            return

        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = ConfigurationManager()

        # Register all adapters
        self._register_adapters()

        # Load initial configurations
        self._load_all_configs()

        self._initialized = True
```

### 5.4 Domain-Specific Adapter Example

The `DigitalTwinAdapter` shows a concrete implementation:

```python
class DigitalTwinAdapter(ConfigAdapterBase):
    """Adapter for Digital Twin configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize Digital Twin adapter."""
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "digital_twin.json"
        )

    def load_config(self) -> Dict[str, Any]:
        """Load Digital Twin configuration."""
        if not self.config_path.exists():
            self.logger.warning(
                f"Config file not found: {self.config_path}, using defaults"
            )
            return self.get_defaults()

        return self._load_json_config(self.config_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Digital Twin configuration."""
        required_sections = {
            "DIGITAL_TWIN_CONFIG",
            "SIMULATION_PARAMETERS",
            "SYNCHRONIZATION_CONFIG",
            "STATE_MANAGEMENT",
        }

        # Check required top-level sections
        if not all(section in config for section in required_sections):
            self.logger.error(
                f"Missing required sections: {required_sections - set(config.keys())}"
            )
            return False

        # Additional validation logic...
        return True
```

## 6. Adapter Pattern Usage

### 6.1 Configuration Loading

Components access configuration through the Constants Service:

```python
# Get access to constants service
constants = ConstantsService()

# Get a specific configuration
digital_twin_config = constants.get_digital_twin_config()

# Access a specific key from an adapter
update_frequency = constants.get_constant(
    "digital_twin", "DIGITAL_TWIN_CONFIG"
).get("update_frequency", 1.0)
```

### 6.2 Adding New Configuration Sources

To add a new configuration source:

1. Create a new adapter class extending `ConfigAdapterBase`
2. Implement the required abstract methods
3. Register the adapter with the `ConfigurationManager`
4. Add access methods to the `ConstantsService`

Example:
```python
# 1. Create new adapter
class NewFeatureAdapter(ConfigAdapterBase):
    def load_config(self) -> Dict[str, Any]:
        # Implementation
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        # Implementation
        pass

    def get_defaults(self) -> Dict[str, Any]:
        # Implementation
        pass

# 2. Register with manager
config_manager.register_adapter("new_feature", NewFeatureAdapter())

# 3. Add to ConstantsService
def get_new_feature_config(self) -> Dict[str, Any]:
    return self.config_manager.get_config("new_feature")
```

## 7. Configuration Flow

### 7.1 Initialization Flow

1. System components instantiate or access the `ConstantsService` singleton
2. `ConstantsService` creates a `ConfigurationManager`
3. Domain adapters are registered with the `ConfigurationManager`
4. Initial configurations are loaded on first access
5. Components access configuration through the service

### 7.2 Configuration Access Flow

```
Component → ConstantsService → ConfigurationManager → Domain Adapter → Configuration Source
```

### 7.3 Configuration Reload Flow

```
Reload Trigger → ConstantsService.reload_configs() → ConfigurationManager.reload_all() → Individual Adapters reload
```

## 8. Error Handling

The adapter system implements comprehensive error handling:

1. Configuration loading errors are logged and default values are used
2. Validation failures result in fallback to defaults
3. Missing configuration sources generate warnings but don't crash the system
4. Configuration access errors provide clear error messages with context

Example error handling:

```python
def load_config(self) -> Dict[str, Any]:
    """Load configuration with error handling."""
    try:
        # Loading logic
        return config
    except FileNotFoundError:
        self.logger.warning("Config file not found, using defaults")
        return self.get_defaults()
    except json.JSONDecodeError as e:
        self.logger.error(f"Invalid JSON configuration: {str(e)}")
        return self.get_defaults()
```

## 9. Best Practices

### 9.1 Configuration Design

1. **Hierarchical Structure**: Organize configurations in logical hierarchies
2. **Descriptive Keys**: Use clear, descriptive keys for configuration items
3. **Sensible Defaults**: Provide reasonable default values for all configuration items
4. **Thorough Validation**: Validate all configuration values upon loading
5. **Clear Documentation**: Document expected structure and values

### 9.2 Adapter Usage

1. **Single Responsibility**: Each adapter should focus on one configuration domain
2. **Clear Abstractions**: Adapt complex sources to simple, unified interfaces
3. **Stateless Design**: Adapters should be primarily stateless
4. **Fail Fast**: Validate configurations early to catch errors
5. **Graceful Degradation**: Fall back to defaults when configurations are invalid

### 9.3 Configuration Access

1. **Centralized Access**: Always access through the Constants Service
2. **Type Checking**: Verify types when retrieving configuration values
3. **Default Values**: Always provide default values when getting specific items
4. **Caching**: Cache frequently accessed configuration values
5. **Explicit Dependencies**: Make configuration dependencies explicit in component initialization

## 10. Integration with Other Components

### 10.1 Digital Twin Integration

The Digital Twin system uses the adapter pattern for configuration:

```python
class DigitalTwinCore:
    def __init__(self):
        # Get configuration via Constants Service
        self.constants = ConstantsService()
        self.config = self.constants.get_digital_twin_config()

        # Extract configuration values with defaults
        self.update_frequency = self.config.get("DIGITAL_TWIN_CONFIG", {}).get(
            "update_frequency", 1.0
        )
        self.history_length = self.config.get("DIGITAL_TWIN_CONFIG", {}).get(
            "history_length", 1000
        )
```

### 10.2 Human Interface Integration

The Human Interface components access configurations:

```python
class InterfaceManager:
    def __init__(self):
        # Get configuration via Constants Service
        self.constants = ConstantsService()

        # Get different domain configurations
        self.viz_config = self.constants.get_visualization_config()
        self.dt_config = self.constants.get_digital_twin_config()
```

### 10.3 Manufacturing Analytics Integration

Manufacturing components leverage configuration:

```python
class ManufacturingAnalyzer:
    def __init__(self):
        # Get configuration via Constants Service
        self.constants = ConstantsService()
        self.manufacturing_config = self.constants.get_manufacturing_constants()

        # Get specific constraint values
        self.min_efficiency = self.constants.get_constant(
            "manufacturing", "CONSTRAINTS"
        ).get("MIN_EFFICIENCY", 0.8)
```

## 11. Future Architectural Extensions

### 11.1 Extended Configuration Sources

1. **Environment Variables**: Support for environment variable configuration
2. **Command-Line Arguments**: Integration with command-line parameter parsing
3. **Remote Configuration**: Support for remote configuration sources
4. **Database Storage**: Configuration loading from database systems
5. **Dynamic Reconfiguration**: Runtime configuration changes with validation

### 11.2 Advanced Features

1. **Configuration Versioning**: Track configuration versions and changes
2. **Schema Validation**: Use JSON Schema for advanced validation
3. **Configuration Discovery**: Automatic discovery of configuration sources
4. **Conditional Configuration**: Configuration based on system state or environment
5. **Configuration Encryption**: Support for encrypted sensitive configuration values

### 11.3 Integration Enhancements

1. **Event-Driven Updates**: Publish events on configuration changes
2. **Configuration Monitoring**: Monitor and report on configuration usage
3. **Configuration UI**: User interface for configuration management
4. **Import/Export**: Support for importing and exporting configurations
5. **Migration Tools**: Tools for configuration migration between versions
