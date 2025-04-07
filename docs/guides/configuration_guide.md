# CIRCMAN5.0 Configuration Guide

## 1. Introduction

This guide provides comprehensive instructions for configuring CIRCMAN5.0 to meet your specific requirements. It covers the configuration architecture, file locations, parameter settings, and best practices for different deployment scenarios.

CIRCMAN5.0 uses a modular configuration approach with adapter-based configuration loading, allowing for flexible and extensible configuration management. This guide will help you understand and utilize this architecture effectively.

## 2. Configuration Architecture

### 2.1 Overview

CIRCMAN5.0 uses a hierarchical configuration architecture with the following components:

- **Configuration Manager**: Central component for managing configuration loading and validation
- **Configuration Adapters**: Components that load configuration from different sources
- **Constants Service**: Service for accessing configuration values across the system
- **Configuration Files**: JSON files containing configuration parameters
- **Environment Variables**: Environment-specific configuration overrides

The architecture follows these principles:

- **Modularity**: Configuration is organized by domain (e.g., manufacturing, digital twin)
- **Validation**: Configuration is validated to ensure correctness
- **Defaults**: Default values are provided for all configuration parameters
- **Extensibility**: New configuration domains can be added without changing existing code

### 2.2 Configuration Flow

The configuration flow in CIRCMAN5.0 is as follows:

1. **Initialization**: The Constants Service is initialized during system startup
2. **Adapter Registration**: Configuration adapters are registered with the Configuration Manager
3. **Configuration Loading**: Adapters load configuration from various sources
4. **Validation**: Configuration is validated against schema definitions
5. **Access**: Components access configuration through the Constants Service

## 3. Configuration Files

### 3.1 File Locations

CIRCMAN5.0 configuration files are located in the following directories:

- **Default Configuration**: `src/circman5/adapters/config/json/`
- **User Configuration**: `config/` (in the project root)
- **Environment-Specific Configuration**: `config/{environment}/`

### 3.2 Configuration File Format

Configuration files are in JSON format and organized by domain. Here's the general structure:

```json
{
  "SECTION_NAME": {
    "parameter1": value1,
    "parameter2": value2,
    "nested_section": {
      "nested_parameter1": nested_value1
    }
  },
  "ANOTHER_SECTION": {
    "parameter3": value3
  }
}
```

### 3.3 Core Configuration Files

CIRCMAN5.0 includes the following core configuration files:

#### 3.3.1 Digital Twin Configuration (`digital_twin.json`)

```json
{
    "DIGITAL_TWIN_CONFIG": {
        "name": "SoliTek_DigitalTwin",
        "update_frequency": 1.0,
        "history_length": 1000,
        "simulation_steps": 10,
        "data_sources": ["sensors", "manual_input", "manufacturing_system"],
        "synchronization_mode": "real_time",
        "log_level": "INFO"
    },
    "SIMULATION_PARAMETERS": {
        "temperature_increment": 0.5,
        "energy_consumption_increment": 2.0,
        "production_rate_increment": 0.2,
        "default_simulation_steps": 10,
        "target_temperature": 22.5,
        "temperature_regulation": 0.1,
        "silicon_wafer_consumption_rate": 0.5,
        "solar_glass_consumption_rate": 0.2
    },
    "SYNCHRONIZATION_CONFIG": {
        "default_sync_interval": 1.0,
        "default_sync_mode": "real_time",
        "retry_interval": 1.0,
        "timeout": 10.0
    },
    "STATE_MANAGEMENT": {
        "default_history_length": 1000,
        "validation_level": "standard",
        "auto_timestamp": true
    },
    "SCENARIO_MANAGEMENT": {
        "max_scenarios": 100
    },
    "EVENT_NOTIFICATION": {
        "persistence_enabled": true,
        "max_events": 1000,
        "default_alert_severity": "warning",
        "publish_state_changes": true,
        "publish_threshold_breaches": true,
        "publish_simulation_results": true
    },
    "PARAMETERS": {
        "PARAMETER_GROUPS": [
            {
                "name": "Process Control",
                "description": "Manufacturing process control parameters",
                "category": "process",
                "parameters": [
                    {
                        "name": "target_temperature",
                        "description": "Target temperature for manufacturing process",
                        "type": "float",
                        "default_value": 22.5,
                        "path": "production_line.temperature",
                        "min_value": 18.0,
                        "max_value": 30.0,
                        "units": "°C",
                        "tags": ["temperature", "control"]
                    },
                    {
                        "name": "energy_limit",
                        "description": "Maximum energy consumption allowed",
                        "type": "float",
                        "default_value": 200.0,
                        "path": "production_line.energy_limit",
                        "min_value": 50.0,
                        "max_value": 500.0,
                        "units": "kWh",
                        "tags": ["energy", "limit"]
                    },
                    {
                        "name": "production_rate",
                        "description": "Target production rate",
                        "type": "float",
                        "default_value": 5.0,
                        "path": "production_line.production_rate",
                        "min_value": 1.0,
                        "max_value": 10.0,
                        "units": "units/hour",
                        "tags": ["production", "rate"]
                    }
                ]
            },
            {
                "name": "Quality Control",
                "description": "Quality control parameters",
                "category": "quality",
                "parameters": [
                    {
                        "name": "defect_threshold",
                        "description": "Maximum allowed defect rate",
                        "type": "float",
                        "default_value": 0.05,
                        "path": "production_line.defect_threshold",
                        "min_value": 0.01,
                        "max_value": 0.1,
                        "units": "",
                        "tags": ["quality", "threshold"]
                    },
                    {
                        "name": "inspection_frequency",
                        "description": "Frequency of quality inspections",
                        "type": "integer",
                        "default_value": 10,
                        "path": "production_line.inspection_frequency",
                        "min_value": 1,
                        "max_value": 100,
                        "units": "units",
                        "tags": ["quality", "inspection"]
                    }
                ]
            },
            {
                "name": "System Settings",
                "description": "General system settings",
                "category": "system",
                "parameters": [
                    {
                        "name": "update_frequency",
                        "description": "Digital twin update frequency",
                        "type": "float",
                        "default_value": 1.0,
                        "path": "",
                        "min_value": 0.1,
                        "max_value": 10.0,
                        "units": "Hz",
                        "tags": ["system", "performance"],
                        "requires_restart": true
                    },
                    {
                        "name": "log_level",
                        "description": "Logging level",
                        "type": "enum",
                        "default_value": "INFO",
                        "path": "",
                        "enum_values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        "tags": ["system", "logging"]
                    }
                ]
            }
        ]
    },
    "PARAMETER_THRESHOLDS": {
        "production_line.temperature": {
            "name": "Production Line Temperature",
            "value": 25.0,
            "comparison": "greater_than",
            "severity": "WARNING"
        },
        "production_line.energy_consumption": {
            "name": "Energy Consumption",
            "value": 200.0,
            "comparison": "greater_than",
            "severity": "WARNING"
        },
        "production_line.defect_rate": {
            "name": "Defect Rate",
            "value": 0.1,
            "comparison": "greater_than",
            "severity": "ERROR"
        }
    },
    "AI_INTEGRATION": {
        "DEFAULT_PARAMETERS": {
            "input_amount": 100.0,
            "energy_used": 50.0,
            "cycle_time": 30.0,
            "efficiency": 0.9,
            "defect_rate": 0.05,
            "thickness_uniformity": 95.0
        },
        "PARAMETER_MAPPING": {
            "production_rate": "output_amount",
            "energy_consumption": "energy_used",
            "temperature": "temperature",
            "cycle_time": "cycle_time"
        },
        "OPTIMIZATION_CONSTRAINTS": {
            "energy_used": [10.0, 100.0],
            "cycle_time": [20.0, 60.0],
            "defect_rate": [0.01, 0.1]
        }
    }
}
```

#### 3.3.2 Manufacturing Configuration (`manufacturing.json`)

```json
{
    "MANUFACTURING_STAGES": {
        "silicon_purification": {
            "input": "raw_silicon",
            "output": "purified_silicon",
            "expected_yield": 0.90
        },
        "wafer_production": {
            "input": "purified_silicon",
            "output": "silicon_wafer",
            "expected_yield": 0.95
        },
        "cell_production": {
            "input": "silicon_wafer",
            "output": "solar_cell",
            "expected_yield": 0.98
        }
    },
    "QUALITY_THRESHOLDS": {
        "min_efficiency": 18.0,
        "max_defect_rate": 5.0,
        "min_thickness_uniformity": 90.0,
        "max_contamination_level": 1.0
    },
    "OPTIMIZATION_TARGETS": {
        "min_yield_rate": 92.0,
        "min_energy_efficiency": 0.7,
        "min_water_reuse": 80.0,
        "min_recycled_content": 30.0
    }
}
```

#### 3.3.3 Impact Factors Configuration (`impact_factors.json`)

```json
{
    "MATERIAL_IMPACT_FACTORS": {
        "silicon_wafer": 32.8,
        "polysilicon": 45.2,
        "metallization_paste": 23.4,
        "solar_glass": 0.9,
        "tempered_glass": 1.2,
        "eva_sheet": 2.6,
        "backsheet": 4.8,
        "aluminum_frame": 8.9,
        "mounting_structure": 2.7,
        "junction_box": 7.5,
        "copper_wiring": 3.2
    },
    "ENERGY_IMPACT_FACTORS": {
        "grid_electricity": 0.5,
        "natural_gas": 0.2,
        "solar_pv": 0.0,
        "wind": 0.0
    },
    "TRANSPORT_IMPACT_FACTORS": {
        "road": 0.062,
        "rail": 0.022,
        "sea": 0.008,
        "air": 0.602
    },
    "RECYCLING_BENEFIT_FACTORS": {
        "silicon": -28.4,
        "glass": -0.7,
        "aluminum": -8.1,
        "copper": -2.8,
        "plastic": -1.8,
        "silicon_wafer": 0.7,
        "solar_glass": 0.8,
        "aluminum_frame": 0.9,
        "copper_wire": 0.85
    },
    "PROCESS_IMPACT_FACTORS": {
        "wafer_cutting": 0.8,
        "cell_processing": 1.2,
        "module_assembly": 0.5,
        "testing": 0.1
    },
    "GRID_CARBON_INTENSITIES": {
        "eu_average": 0.275,
        "us_average": 0.417,
        "china": 0.555,
        "india": 0.708
    },
    "DEGRADATION_RATES": {
        "mono_perc": 0.5,
        "poly_bsf": 0.6,
        "thin_film": 0.7,
        "bifacial": 0.45
    },
    "CARBON_INTENSITY_FACTORS": {
        "grid": 0.5,
        "solar": 0.0,
        "wind": 0.0,
        "electricity": 0.5,
        "natural_gas": 0.2,
        "petroleum": 0.25
    },
    "SUSTAINABILITY_WEIGHTS": {
        "material": 0.4,
        "recycling": 0.3,
        "energy": 0.3,
        "material_efficiency": 0.4,
        "carbon_footprint": 0.4,
        "energy_efficiency": 0.3,
        "recycling_rate": 0.3
    },
    "QUALITY_WEIGHTS": {
        "defect": 0.4,
        "efficiency": 0.4,
        "uniformity": 0.2,
        "defect_rate": 0.4,
        "efficiency_score": 0.4,
        "uniformity_score": 0.2
    },
    "MONITORING_WEIGHTS": {
        "defect": 0.4,
        "yield": 0.4,
        "uniformity": 0.2
    },
    "WATER_FACTOR": 0.1,
    "WASTE_FACTOR": 0.05
}
```

#### 3.3.4 Optimization Configuration (`optimization.json`)

```json
{
    "MODEL_CONFIG": {
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "model_params": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "subsample": 0.8
        }
    },
    "FEATURE_COLUMNS": [
        "input_amount",
        "energy_used",
        "cycle_time",
        "efficiency",
        "defect_rate",
        "thickness_uniformity"
    ],
    "OPTIMIZATION_CONSTRAINTS": {
        "min_yield_rate": 92.0,
        "max_cycle_time": 60.0,
        "min_efficiency": 18.0,
        "max_defect_rate": 5.0,
        "min_thickness_uniformity": 90.0,
        "max_energy_consumption": 160.0
    },
    "TRAINING_PARAMETERS": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "tol": 1e-4
    },
    "ADVANCED_MODELS": {
        "deep_learning": {
            "model_type": "lstm",
            "hidden_layers": [64, 32],
            "activation": "relu",
            "dropout_rate": 0.2,
            "l2_regularization": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 10
        },
        "ensemble": {
            "base_models": ["random_forest", "gradient_boosting", "extra_trees"],
            "meta_model": "linear",
            "cv_folds": 5,
            "use_probabilities": true,
            "voting_strategy": "soft"
        }
    },
    "ONLINE_LEARNING": {
        "window_size": 100,
        "learning_rate": 0.01,
        "update_frequency": 10,
        "regularization": 0.001,
        "forgetting_factor": 0.95,
        "max_model_age": 24,
        "model_persistence_interval": 60
    },
    "VALIDATION": {
        "cross_validation": {
            "method": "stratified_kfold",
            "n_splits": 5,
            "shuffle": true,
            "random_state": 42,
            "metrics": ["accuracy", "precision", "recall", "f1", "r2", "mse"]
        },
        "uncertainty": {
            "method": "monte_carlo_dropout",
            "samples": 30,
            "confidence_level": 0.95,
            "calibration_method": "temperature_scaling"
        }
    }
}
```

## 4. Configuration Components

### 4.1 Configuration Manager

The Configuration Manager (`config_manager.py`) is responsible for loading and validating configuration from various adapters:

```python
class ConfigurationManager:
    """Manages configuration loading and validation across different adapters."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional base path for configuration files
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path or Path("config")
        self.adapters: Dict[str, ConfigAdapterBase] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}

    def register_adapter(self, name: str, adapter: ConfigAdapterBase) -> None:
        """
        Register a new configuration adapter.

        Args:
            name: Unique identifier for the adapter
            adapter: Adapter instance to register

        Raises:
            ValueError: If adapter name already exists
        """
        if name in self.adapters:
            raise ValueError(f"Adapter already registered: {name}")

        self.adapters[name] = adapter
        self.logger.info(f"Registered adapter: {name}")

    def load_config(self, adapter_name: str) -> Dict[str, Any]:
        """Load configuration using specified adapter."""
        if adapter_name not in self.adapters:
            self.logger.error(f"Unknown adapter: {adapter_name}")
            raise ValueError(f"Unknown adapter: {adapter_name}")

        adapter = self.adapters[adapter_name]
        self.logger.info(
            f"Loading config for {adapter_name} using adapter {adapter.__class__.__name__}"
        )

        try:
            config = adapter.load_config()
            self.logger.info(
                f"Loaded config for {adapter_name}: keys: {list(config.keys())}"
            )

            if not adapter.validate_config(config):
                self.logger.warning(
                    f"Invalid configuration for {adapter_name}, using defaults"
                )
                config = adapter.get_defaults()

            self.configs[adapter_name] = config
            return config

        except Exception as e:
            self.logger.error(f"Error loading config for {adapter_name}: {str(e)}")
            raise

    def get_config(self, adapter_name: str) -> Dict[str, Any]:
        """
        Get loaded configuration for an adapter.

        Args:
            adapter_name: Name of adapter

        Returns:
            Dict[str, Any]: Current configuration

        Raises:
            ValueError: If adapter not found or config not loaded
        """
        if adapter_name not in self.configs:
            self.load_config(adapter_name)

        return self.configs[adapter_name]

    def reload_all(self) -> None:
        """Reload all registered configurations."""
        for name in self.adapters.keys():
            self.load_config(name)

    def get_adapter(self, name: str) -> ConfigAdapterBase:
        """
        Get registered adapter by name.

        Args:
            name: Adapter name

        Returns:
            ConfigAdapterBase: Registered adapter

        Raises:
            ValueError: If adapter not found
        """
        if name not in self.adapters:
            raise ValueError(f"Unknown adapter: {name}")

        return self.adapters[name]
```

### 4.2 Configuration Adapter Base

The Configuration Adapter Base (`adapter_base.py`) defines the interface for configuration adapters:

```python
class ConfigAdapterBase(ABC):
    """Base interface for configuration adapters."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize adapter with optional config path."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path

    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from source.

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary to validate

        Returns:
            bool: True if configuration is valid
        """
        pass

    @abstractmethod
    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default configuration values.

        Returns:
            Dict[str, Any]: Default configuration
        """
        pass
```

### 4.3 Constants Service

The Constants Service (`constants_service.py`) provides centralized access to configuration values:

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

    @classmethod
    def _reset_instance(cls):
        """Reset the singleton instance - FOR TESTING ONLY."""
        cls._instance = None

    def _register_adapters(self) -> None:
        """Register all configuration adapters."""
        try:
            self.config_manager.register_adapter(
                "manufacturing", ManufacturingAdapter()
            )
            self.config_manager.register_adapter(
                "impact_factors", ImpactFactorsAdapter()
            )
            self.config_manager.register_adapter("optimization", OptimizationAdapter())
            self.config_manager.register_adapter("monitoring", MonitoringAdapter())
            self.config_manager.register_adapter(
                "visualization", VisualizationAdapter()
            )
            self.config_manager.register_adapter("digital_twin", DigitalTwinAdapter())
            # Add any other adapters here
        except Exception as e:
            self.logger.error(f"Error registering adapters: {str(e)}")
            raise

    def _load_all_configs(self) -> None:
        """Load all configurations."""
        try:
            for adapter_name in self.config_manager.adapters.keys():
                self.config_manager.load_config(adapter_name)
        except Exception as e:
            self.logger.error(f"Error loading configurations: {str(e)}")
            raise

    def get_manufacturing_constants(self) -> Dict[str, Any]:
        """Get manufacturing constants."""
        return self.config_manager.get_config("manufacturing")

    def get_impact_factors(self) -> Dict[str, Any]:
        """Get impact factors constants."""
        return self.config_manager.get_config("impact_factors")

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self.config_manager.get_config("optimization")

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config_manager.get_config("visualization")

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.config_manager.get_config("monitoring")

    def get_digital_twin_config(self) -> Dict[str, Any]:
        """Get digital twin configuration"""
        return self.config_manager.get_config("digital_twin")

    def get_constant(self, adapter: str, key: str) -> Any:
        """
        Get specific constant by adapter and key.

        Args:
            adapter: Adapter name
            key: Configuration key

        Returns:
            Any: Configuration value

        Raises:
            KeyError: If key not found
        """
        config = self.config_manager.get_config(adapter)
        self.logger.debug(
            f"Looking for key '{key}' in {adapter} config with keys: {list(config.keys())}"
        )
        if key not in config:
            self.logger.error(f"Key not found: '{key}' in {adapter} config")
            raise KeyError(f"Key not found in {adapter} config: {key}")
        return config[key]

    def reload_configs(self) -> None:
        """Reload all configurations."""
        self.config_manager.reload_all()
```

## 5. Configuring System Components

### 5.1 Digital Twin Configuration

To configure the Digital Twin component:

1. **Create or modify** `config/digital_twin.json`
2. **Update settings** according to your requirements
3. **Reload configuration** through the Constants Service

Example configuration process:

```python
from circman5.adapters.services.constants_service import ConstantsService
import json
from pathlib import Path

# Create configuration directory if it doesn't exist
config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

# Load current configuration
constants_service = ConstantsService()
current_config = constants_service.get_digital_twin_config()

# Modify configuration
modified_config = current_config.copy()
modified_config["DIGITAL_TWIN_CONFIG"]["update_frequency"] = 2.0
modified_config["SIMULATION_PARAMETERS"]["target_temperature"] = 24.0

# Save modified configuration
with open(config_dir / "digital_twin.json", "w") as f:
    json.dump(modified_config, f, indent=2)

# Reload configuration
constants_service.reload_configs()

# Verify changes
updated_config = constants_service.get_digital_twin_config()
print(f"Update Frequency: {updated_config['DIGITAL_TWIN_CONFIG']['update_frequency']}")
print(f"Target Temperature: {updated_config['SIMULATION_PARAMETERS']['target_temperature']}")
```

### 5.2 Manufacturing Configuration

To configure the Manufacturing component:

1. **Create or modify** `config/manufacturing.json`
2. **Update settings** according to your requirements
3. **Reload configuration** through the Constants Service

Example configuration process:

```python
from circman5.adapters.services.constants_service import ConstantsService
import json
from pathlib import Path

# Create configuration directory if it doesn't exist
config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

# Load current configuration
constants_service = ConstantsService()
current_config = constants_service.get_manufacturing_constants()

# Modify configuration
modified_config = current_config.copy()
modified_config["QUALITY_THRESHOLDS"]["min_efficiency"] = 20.0
modified_config["OPTIMIZATION_TARGETS"]["min_yield_rate"] = 95.0

# Save modified configuration
with open(config_dir / "manufacturing.json", "w") as f:
    json.dump(modified_config, f, indent=2)

# Reload configuration
constants_service.reload_configs()

# Verify changes
updated_config = constants_service.get_manufacturing_constants()
print(f"Min Efficiency: {updated_config['QUALITY_THRESHOLDS']['min_efficiency']}")
print(f"Min Yield Rate: {updated_config['OPTIMIZATION_TARGETS']['min_yield_rate']}")
```

### 5.3 Optimization Configuration

To configure the Optimization component:

1. **Create or modify** `config/optimization.json`
2. **Update settings** according to your requirements
3. **Reload configuration** through the Constants Service

Example configuration process:

```python
from circman5.adapters.services.constants_service import ConstantsService
import json
from pathlib import Path

# Create configuration directory if it doesn't exist
config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

# Load current configuration
constants_service = ConstantsService()
current_config = constants_service.get_optimization_config()

# Modify configuration
modified_config = current_config.copy()
modified_config["MODEL_CONFIG"]["test_size"] = 0.25
modified_config["MODEL_CONFIG"]["model_params"]["n_estimators"] = 200

# Save modified configuration
with open(config_dir / "optimization.json", "w") as f:
    json.dump(modified_config, f, indent=2)

# Reload configuration
constants_service.reload_configs()

# Verify changes
updated_config = constants_service.get_optimization_config()
print(f"Test Size: {updated_config['MODEL_CONFIG']['test_size']}")
print(f"N Estimators: {updated_config['MODEL_CONFIG']['model_params']['n_estimators']}")
```

## 6. Environment-Specific Configuration

### 6.1 Environment Variables

CIRCMAN5.0 supports configuration through environment variables, which override values in configuration files. Environment variables follow this naming convention:

```
CIRCMAN_<DOMAIN>_<KEY>
```

For example:

```bash
# Set Digital Twin update frequency
export CIRCMAN_DIGITAL_TWIN_UPDATE_FREQUENCY=2.0

# Set optimization test size
export CIRCMAN_OPTIMIZATION_TEST_SIZE=0.25

# Set manufacturing min efficiency
export CIRCMAN_MANUFACTURING_MIN_EFFICIENCY=20.0
```

### 6.2 Development Environment

For development environments, create a `config/development/` directory with configuration files specific to development:

```bash
mkdir -p config/development
```

Example development configuration (`config/development/digital_twin.json`):

```json
{
    "DIGITAL_TWIN_CONFIG": {
        "log_level": "DEBUG"
    },
    "EVENT_NOTIFICATION": {
        "persistence_enabled": false
    }
}
```

Load development configuration:

```python
import os
os.environ["CIRCMAN_ENV"] = "development"

from circman5.adapters.services.constants_service import ConstantsService
constants_service = ConstantsService()
```

### 6.3 Production Environment

For production environments, create a `config/production/` directory with configuration files specific to production:

```bash
mkdir -p config/production
```

Example production configuration (`config/production/digital_twin.json`):

```json
{
    "DIGITAL_TWIN_CONFIG": {
        "log_level": "WARNING"
    },
    "EVENT_NOTIFICATION": {
        "persistence_enabled": true,
        "max_events": 10000
    }
}
```

Load production configuration:

```python
import os
os.environ["CIRCMAN_ENV"] = "production"

from circman5.adapters.services.constants_service import ConstantsService
constants_service = ConstantsService()
```

## 7. Dynamic Configuration

### 7.1 Runtime Configuration Changes

CIRCMAN5.0 supports runtime configuration changes through the Constants Service. Changes made at runtime are not persisted to configuration files and will be reset when the system is restarted.

Example of runtime configuration change:

```python
from circman5.adapters.services.constants_service import ConstantsService

# Get constants service
constants_service = ConstantsService()

# Create an update for digital twin configuration
digital_twin_updates = {
    "DIGITAL_TWIN_CONFIG": {
        "update_frequency": 2.0
    }
}

# Apply update
for section, values in digital_twin_updates.items():
    current_config = constants_service.get_digital_twin_config()
    if section in current_config:
        for key, value in values.items():
            if key in current_config[section]:
                current_config[section][key] = value

# Notify system of changes
constants_service.config_manager.configs["digital_twin"] = current_config
```

### 7.2 Configuration Subscribers

Components can subscribe to configuration changes to be notified when configuration is updated:

```python
class ConfigurationSubscriber:
    """Base class for configuration change subscribers."""

    def __init__(self, constants_service):
        """Initialize the subscriber."""
        self.constants_service = constants_service
        self.constants_service.register_subscriber(self)

    def handle_configuration_change(self, domain, changes):
        """Handle configuration change notification."""
        pass

class DigitalTwinConfigSubscriber(ConfigurationSubscriber):
    """Subscriber for Digital Twin configuration changes."""

    def handle_configuration_change(self, domain, changes):
        """Handle configuration change notification."""
        if domain == "digital_twin":
            print(f"Digital Twin configuration changed: {changes}")
            # Apply changes to Digital Twin

# Register subscriber
subscriber = DigitalTwinConfigSubscriber(constants_service)
```

## 8. Configuration Validation

### 8.1 Schema Validation

CIRCMAN5.0 validates configuration against schema definitions to ensure correctness. Each configuration adapter includes validation logic:

```python
def validate_config(self, config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if configuration is valid
    """
    # Check required sections
    required_sections = ["DIGITAL_TWIN_CONFIG", "SIMULATION_PARAMETERS", "STATE_MANAGEMENT"]
    for section in required_sections:
        if section not in config:
            self.logger.error(f"Missing required section: {section}")
            return False

    # Check specific parameters
    dt_config = config.get("DIGITAL_TWIN_CONFIG", {})
    required_params = ["name", "update_frequency", "history_length"]
    for param in required_params:
        if param not in dt_config:
            self.logger.error(f"Missing required parameter: {param} in DIGITAL_TWIN_CONFIG")
            return False

    # Validate parameter types
    if not isinstance(dt_config.get("update_frequency"), (int, float)):
        self.logger.error("update_frequency must be a number")
        return False

    if not isinstance(dt_config.get("history_length"), int):
        self.logger.error("history_length must be an integer")
        return False

    return True
```

### 8.2 Configuration Testing

To test configuration validity:

```python
import pytest
from circman5.adapters.config.digital_twin import DigitalTwinAdapter

def test_digital_twin_config():
    """Test Digital Twin configuration validation."""
    adapter = DigitalTwinAdapter()

    # Test valid configuration
    valid_config = {
        "DIGITAL_TWIN_CONFIG": {
            "name": "Test_DT",
            "update_frequency": 1.0,
            "history_length": 1000
        },
        "SIMULATION_PARAMETERS": {},
        "STATE_MANAGEMENT": {}
    }
    assert adapter.validate_config(valid_config) is True

    # Test invalid configuration (missing section)
    invalid_config1 = {
        "DIGITAL_TWIN_CONFIG": {
            "name": "Test_DT",
            "update_frequency": 1.0,
            "history_length": 1000
        }
    }
    assert adapter.validate_config(invalid_config1) is False

    # Test invalid configuration (missing parameter)
    invalid_config2 = {
        "DIGITAL_TWIN_CONFIG": {
            "name": "Test_DT"
        },
        "SIMULATION_PARAMETERS": {},
        "STATE_MANAGEMENT": {}
    }
    assert adapter.validate_config(invalid_config2) is False

    # Test invalid configuration (wrong parameter type)
    invalid_config3 = {
        "DIGITAL_TWIN_CONFIG": {
            "name": "Test_DT",
            "update_frequency": "1.0",  # Should be number
            "history_length": 1000
        },
        "SIMULATION_PARAMETERS": {},
        "STATE_MANAGEMENT": {}
    }
    assert adapter.validate_config(invalid_config3) is False
```

## 9. Configuration Best Practices

### 9.1 Configuration Structure

Follow these guidelines for configuration structure:

- **Use sections** to organize related parameters
- **Use consistent naming conventions** for parameters
- **Include default values** for all parameters
- **Document parameters** with comments or descriptions
- **Use appropriate data types** for parameters
- **Keep related parameters together** in the same section

### 9.2 Environment-Specific Configuration

Follow these guidelines for environment-specific configuration:

- **Create separate directories** for each environment
- **Override only necessary parameters** in environment-specific configuration
- **Document environment differences** in comments or documentation
- **Use environment variables** for sensitive or deployment-specific values
- **Test configuration changes** in a staging environment before production

### 9.3 Configuration Security

Follow these guidelines for configuration security:

- **Don't store sensitive information** in configuration files
- **Use environment variables** for sensitive values
- **Encrypt sensitive configuration** when necessary
- **Restrict access** to configuration files
- **Validate inputs** before using them in configuration
- **Don't expose configuration** through APIs or interfaces

## 10. Troubleshooting Configuration Issues

### 10.1 Common Configuration Issues

#### 10.1.1 Missing Configuration Files

Issue: System fails to start due to missing configuration files.

Solution: Ensure that default configuration files are present in the expected locations:

```bash
mkdir -p config
cp src/circman5/adapters/config/json/*.json config/
```

#### 10.1.2 Invalid Configuration Format

Issue: System fails to load configuration due to invalid JSON format.

Solution: Validate JSON syntax in configuration files:

```bash
python -m json.tool config/digital_twin.json > /dev/null
```

#### 10.1.3 Schema Validation Failures

Issue: Configuration fails validation due to missing or invalid parameters.

Solution: Review validation errors in logs and update configuration accordingly:

```bash
grep "ERROR.*config" logs/circman.log
```

### 10.2 Configuration Debugging

To enable detailed logging for configuration debugging:

```python
import logging
logging.getLogger("circman5.adapters.base.config_manager").setLevel(logging.DEBUG)
logging.getLogger("circman5.adapters.services.constants_service").setLevel(logging.DEBUG)
```

### 10.3 Configuration Reset

To reset configuration to defaults:

```python
from circman5.adapters.services.constants_service import ConstantsService

# Reset singleton instance
ConstantsService._reset_instance()

# Get fresh instance with default configuration
constants_service = ConstantsService()
```

## 11. Conclusion

This guide has provided comprehensive instructions for configuring CIRCMAN5.0 to meet your specific requirements. By understanding the configuration architecture and following the guidelines provided, you can effectively configure the system for different environments and use cases.

For more information on deploying CIRCMAN5.0, refer to the [Deployment Guide](deployment_guide.md).
