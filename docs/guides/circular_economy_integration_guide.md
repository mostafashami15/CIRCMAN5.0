# Circular Economy Integration Guide for CIRCMAN5.0

## 1. Introduction

This integration guide provides detailed instructions for integrating the Circular Economy components of CIRCMAN5.0 with other system modules, external systems, and manufacturing processes. It covers integration patterns, APIs, data flows, and best practices for achieving seamless integration.

## 2. Integration Overview

The Circular Economy components in CIRCMAN5.0 are designed to integrate with:

1. **Other CIRCMAN5.0 Modules**:
   - Digital Twin System
   - Human-Machine Interface
   - Life Cycle Assessment System
   - Manufacturing Control System

2. **External Systems**:
   - ERP Systems
   - MES (Manufacturing Execution Systems)
   - SCADA Systems
   - Supply Chain Management Systems

3. **Data Sources**:
   - Production Data Streams
   - Quality Management Systems
   - Environmental Monitoring Systems
   - Energy Management Systems

## 3. Integration with Digital Twin System

The Circular Economy components integrate with the Digital Twin system to enable real-time optimization and monitoring of manufacturing processes.

### 3.1 Digital Twin Integration Architecture

```
┌───────────────────┐      ┌───────────────────┐
│                   │      │                   │
│  Digital Twin     │      │  Circular Economy │
│  System           │      │  Components       │
│                   │      │                   │
└───────┬───────────┘      └───────┬───────────┘
        │                          │
        ▼                          ▼
┌───────────────────────────────────────────────┐
│                                               │
│  Integration Layer                            │
│                                               │
└───────────────────────────────────────────────┘
```

### 3.2 Digital Twin Integration Steps

1. **Configure Digital Twin Adapter**:

```python
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer

# Initialize Digital Twin integration
digital_twin = get_digital_twin_instance()  # Get DT instance
ai_integration = AIIntegration(digital_twin)

# Connect optimizer to Digital Twin
optimizer = ProcessOptimizer()
ai_integration.register_optimizer(optimizer)
```

2. **Subscribe to Digital Twin Events**:

```python
from circman5.manufacturing.digital_twin.event_notification.subscribers import EventSubscriber

class OptimizationSubscriber(EventSubscriber):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    def handle_event(self, event):
        if event.type == "process_completed":
            # Get latest process data
            process_data = event.data

            # Run optimization
            current_params = process_data["parameters"]
            optimized_params = self.optimizer.optimize_process_parameters(current_params)

            # Send back to Digital Twin
            digital_twin.update_parameters(optimized_params)

# Register subscriber
subscriber = OptimizationSubscriber(optimizer)
digital_twin.event_manager.subscribe("process_completed", subscriber)
```

3. **Access Digital Twin State**:

```python
# Get current state from Digital Twin
current_state = digital_twin.state_manager.get_current_state()

# Extract relevant parameters for optimization
process_params = {
    "input_amount": current_state.get("input_amount"),
    "energy_used": current_state.get("energy_used"),
    "cycle_time": current_state.get("cycle_time")
}

# Optimize parameters
optimized_params = optimizer.optimize_process_parameters(process_params)

# Update Digital Twin with optimized parameters
digital_twin.state_manager.update_parameters(optimized_params)
```

### 3.3 Digital Twin Synchronization

To ensure the Digital Twin accurately reflects real-world conditions:

```python
# Configure synchronization settings
digital_twin.synchronization.configure({
    "sync_interval": 60,  # seconds
    "data_validation": True,
    "auto_correction": True
})

# Register sustainability metrics for synchronization
digital_twin.synchronization.register_metrics([
    "material_efficiency",
    "energy_efficiency",
    "recycling_rate",
    "carbon_footprint"
])
```

## 4. Integration with Human-Machine Interface

The Circular Economy components integrate with the Human-Machine Interface (HMI) to provide visualization and control of circular economy features.

### 4.1 HMI Integration Architecture

```
┌───────────────────┐      ┌───────────────────┐
│                   │      │                   │
│  Human-Machine    │      │  Circular Economy │
│  Interface        │      │  Components       │
│                   │      │                   │
└───────┬───────────┘      └───────┬───────────┘
        │                          │
        ▼                          ▼
┌───────────────────────────────────────────────┐
│                                               │
│  Integration Layer                            │
│                                               │
└───────────────────────────────────────────────┘
```

### 4.2 HMI Dashboard Integration

1. **Register Circular Economy Dashboards**:

```python
from circman5.manufacturing.human_interface.core.dashboard_manager import DashboardManager
from circman5.manufacturing.human_interface.components.dashboard.main_dashboard import MainDashboard

# Get dashboard manager
dashboard_manager = DashboardManager()

# Register circular economy dashboard
dashboard_manager.register_dashboard({
    "id": "circular_economy",
    "title": "Circular Economy Metrics",
    "panels": [
        {
            "id": "material_efficiency_panel",
            "title": "Material Efficiency",
            "type": "metric",
            "data_source": "material_efficiency_data"
        },
        {
            "id": "sustainability_panel",
            "title": "Sustainability Metrics",
            "type": "chart",
            "data_source": "sustainability_data"
        },
        {
            "id": "optimization_panel",
            "title": "Process Optimization",
            "type": "control",
            "data_source": "optimization_controls"
        }
    ]
})
```

2. **Connect Data Sources**:

```python
from circman5.manufacturing.human_interface.services.data_service import DataService
from circman5.manufacturing.analyzers.sustainability import SustainabilityAnalyzer
from circman5.manufacturing.analyzers.efficiency import EfficiencyAnalyzer

# Get data service
data_service = DataService()

# Create analyzers
sustainability_analyzer = SustainabilityAnalyzer()
efficiency_analyzer = EfficiencyAnalyzer()

# Register data sources
data_service.register_source("material_efficiency_data", {
    "provider": efficiency_analyzer,
    "method": "analyze_batch_efficiency",
    "refresh_interval": 300,  # seconds
    "parameters": {
        "data": "production_data"
    }
})

data_service.register_source("sustainability_data", {
    "provider": sustainability_analyzer,
    "method": "calculate_sustainability_metrics",
    "refresh_interval": 300,  # seconds
    "parameters": {
        "energy_data": "energy_data",
        "material_flow": "material_data"
    }
})
```

3. **Register Control Handlers**:

```python
from circman5.manufacturing.human_interface.services.command_service import CommandService
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer

# Get command service
command_service = CommandService()

# Create optimizer
optimizer = ProcessOptimizer()

# Register command handler
def handle_optimization_command(command, params):
    if command == "optimize_parameters":
        current_params = params["current_params"]
        constraints = params.get("constraints", {})

        # Run optimization
        optimized_params = optimizer.optimize_process_parameters(
            current_params,
            constraints=constraints
        )

        return {
            "success": True,
            "result": optimized_params
        }

    return {
        "success": False,
        "error": "Unknown command"
    }

command_service.register_handler("optimization_controls", handle_optimization_command)
```

### 4.3 HMI Alert Integration

To integrate Circular Economy alerts with the HMI alert system:

```python
from circman5.manufacturing.human_interface.components.alerts.notification_manager import NotificationManager

# Get notification manager
notification_manager = NotificationManager()

# Register alert types
notification_manager.register_alert_type({
    "id": "resource_efficiency",
    "name": "Resource Efficiency Alert",
    "priority": "medium",
    "icon": "resource_icon"
})

notification_manager.register_alert_type({
    "id": "recycling_rate",
    "name": "Recycling Rate Alert",
    "priority": "medium",
    "icon": "recycle_icon"
})

# Create alert function
def check_efficiency_alerts(efficiency_data):
    alerts = []

    # Check material efficiency
    if efficiency_data["yield_rate"] < 80:
        alerts.append({
            "type": "resource_efficiency",
            "message": f"Material yield rate below threshold: {efficiency_data['yield_rate']:.2f}%",
            "value": efficiency_data["yield_rate"],
            "threshold": 80,
            "timestamp": datetime.now()
        })

    # Check energy efficiency
    if "energy_efficiency" in efficiency_data and efficiency_data["energy_efficiency"] < 70:
        alerts.append({
            "type": "resource_efficiency",
            "message": f"Energy efficiency below threshold: {efficiency_data['energy_efficiency']:.2f}",
            "value": efficiency_data["energy_efficiency"],
            "threshold": 70,
            "timestamp": datetime.now()
        })

    return alerts

# Register alert handler
notification_manager.register_alert_handler("efficiency_alerts", check_efficiency_alerts)
```

## 5. Integration with Life Cycle Assessment System

The Circular Economy components integrate with the Life Cycle Assessment (LCA) system to provide comprehensive environmental impact analysis.

### 5.1 LCA Integration Architecture

```
┌───────────────────┐      ┌───────────────────┐
│                   │      │                   │
│  Life Cycle       │      │  Circular Economy │
│  Assessment       │      │  Components       │
│                   │      │                   │
└───────┬───────────┘      └───────┬───────────┘
        │                          │
        ▼                          ▼
┌───────────────────────────────────────────────┐
│                                               │
│  Integration Layer                            │
│                                               │
└───────────────────────────────────────────────┘
```

### 5.2 LCA Data Integration

1. **Share Impact Factors**:

```python
from circman5.manufacturing.lifecycle.impact_factors import (
    MATERIAL_IMPACT_FACTORS,
    ENERGY_IMPACT_FACTORS,
    RECYCLING_BENEFIT_FACTORS
)
from circman5.manufacturing.lifecycle.lca_analyzer import LCAAnalyzer

# Initialize LCA Analyzer
lca_analyzer = LCAAnalyzer()

# Configure with impact factors
lca_analyzer.configure_impact_factors({
    "material_factors": MATERIAL_IMPACT_FACTORS,
    "energy_factors": ENERGY_IMPACT_FACTORS,
    "recycling_factors": RECYCLING_BENEFIT_FACTORS
})
```

2. **Provide Process Data to LCA**:

```python
from circman5.manufacturing.analyzers.sustainability import SustainabilityAnalyzer

# Initialize sustainability analyzer
sustainability_analyzer = SustainabilityAnalyzer()

# Get sustainability metrics
sustainability_metrics = sustainability_analyzer.calculate_sustainability_metrics(
    energy_data=energy_data,
    material_flow=material_flow
)

# Provide to LCA system
lca_analyzer.update_process_data({
    "carbon_footprint": sustainability_metrics["carbon_footprint"],
    "material_efficiency": sustainability_metrics["material_efficiency"],
    "recycling_rate": sustainability_metrics["material_efficiency"]["recycling_rate"]
})
```

3. **Retrieve LCA Results**:

```python
# Calculate LCA impact
lca_results = lca_analyzer.calculate_lifecycle_impact({
    "functional_unit": "1 kW PV module",
    "lifecycle_stages": ["manufacturing", "use", "end_of_life"],
    "impact_categories": ["global_warming", "resource_depletion", "energy_consumption"]
})

# Process results
print("Life Cycle Impact Results:")
for category, value in lca_results["impact_values"].items():
    print(f"  {category}: {value:.2f}")

# Generate visualization
lca_analyzer.visualize_results(
    lca_results,
    save_path="lca_results.png"
)
```

### 5.3 Circular Economy Integration with LCA

To provide circular economy focus within LCA:

```python
# Configure circular economy analysis
lca_analyzer.configure_circular_analysis({
    "enable_recycling_benefits": True,
    "enable_reuse_modeling": True,
    "enable_waste_reduction_scenarios": True
})

# Define circular scenarios
circular_scenarios = {
    "baseline": {
        "recycling_rate": 0.2,
        "material_reuse": 0.1,
        "lifetime_extension": 0
    },
    "improved": {
        "recycling_rate": 0.6,
        "material_reuse": 0.3,
        "lifetime_extension": 5  # years
    },
    "circular": {
        "recycling_rate": 0.9,
        "material_reuse": 0.5,
        "lifetime_extension": 10  # years
    }
}

# Calculate impact for each scenario
scenario_results = {}
for scenario_name, scenario_params in circular_scenarios.items():
    scenario_results[scenario_name] = lca_analyzer.calculate_lifecycle_impact(
        {
            "functional_unit": "1 kW PV module",
            "lifecycle_stages": ["manufacturing", "use", "end_of_life"],
            "impact_categories": ["global_warming", "resource_depletion", "energy_consumption"]
        },
        circular_params=scenario_params
    )

# Compare scenarios
lca_analyzer.compare_scenarios(
    scenario_results,
    save_path="circular_scenario_comparison.png"
)
```

## 6. Integration with External Systems

### 6.1 ERP System Integration

The Circular Economy components can integrate with ERP systems to access and update material, inventory, and manufacturing data.

1. **Configure ERP Connector**:

```python
from circman5.adapters.base.adapter_base import BaseAdapter

class ERPAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.erp_client = self._create_erp_client(config)

    def _create_erp_client(self, config):
        # Implementation depends on specific ERP system
        # Example for SAP:
        try:
            from sap_client import SAPClient
            return SAPClient(
                host=config["host"],
                port=config["port"],
                username=config["username"],
                password=config["password"]
            )
        except ImportError:
            raise ImportError("SAP client library not found")

    def get_material_data(self, material_codes):
        """Get material data from ERP."""
        return self.erp_client.query_materials(material_codes)

    def get_production_orders(self, date_from, date_to):
        """Get production orders from ERP."""
        return self.erp_client.query_production_orders(date_from, date_to)

    def update_material_usage(self, material_usage_data):
        """Update material usage in ERP."""
        return self.erp_client.update_material_usage(material_usage_data)

# Create ERP adapter
erp_adapter = ERPAdapter({
    "host": "erp.example.com",
    "port": 8000,
    "username": "api_user",
    "password": "api_password"
})
```

2. **Retrieve Material Data from ERP**:

```python
# Get material data
material_data = erp_adapter.get_material_data([
    "MAT001", "MAT002", "MAT003"
])

# Process for sustainability analysis
from circman5.manufacturing.analyzers.sustainability import SustainabilityAnalyzer
import pandas as pd

# Convert to pandas DataFrame
df_material = pd.DataFrame(material_data)

# Initialize analyzer
sustainability_analyzer = SustainabilityAnalyzer()

# Analyze material efficiency
material_metrics = sustainability_analyzer.analyze_material_efficiency(df_material)
print("Material efficiency:", material_metrics)
```

3. **Update ERP with Circular Economy Data**:

```python
# Generate material usage data with recycling information
material_usage = {
    "transaction_id": "TRX123456",
    "date": "2025-02-13",
    "materials": [
        {
            "material_code": "MAT001",
            "quantity_used": 100.0,
            "waste_generated": 10.0,
            "recycled_amount": 8.0
        },
        {
            "material_code": "MAT002",
            "quantity_used": 50.0,
            "waste_generated": 5.0,
            "recycled_amount": 4.0
        }
    ]
}

# Update ERP
update_result = erp_adapter.update_material_usage(material_usage)
```

### 6.2 MES Integration

To integrate with Manufacturing Execution Systems (MES):

1. **Configure MES Adapter**:

```python
from circman5.adapters.base.adapter_base import BaseAdapter

class MESAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.mes_client = self._create_mes_client(config)

    def _create_mes_client(self, config):
        # Implementation depends on specific MES system
        # Example:
        try:
            from mes_client import MESClient
            return MESClient(
                url=config["url"],
                api_key=config["api_key"]
            )
        except ImportError:
            raise ImportError("MES client library not found")

    def get_production_data(self, start_time, end_time):
        """Get production data from MES."""
        return self.mes_client.query_production(start_time, end_time)

    def get_quality_data(self, start_time, end_time):
        """Get quality data from MES."""
        return self.mes_client.query_quality(start_time, end_time)

    def update_process_parameters(self, parameters):
        """Update process parameters in MES."""
        return self.mes_client.update_parameters(parameters)

# Create MES adapter
mes_adapter = MESAdapter({
    "url": "https://mes.example.com/api",
    "api_key": "mes_api_key_123"
})
```

2. **Retrieve Production Data from MES**:

```python
from datetime import datetime, timedelta

# Define time range
end_time = datetime.now()
start_time = end_time - timedelta(days=7)

# Get production data
production_data = mes_adapter.get_production_data(start_time, end_time)
quality_data = mes_adapter.get_quality_data(start_time, end_time)

# Convert to pandas DataFrames
import pandas as pd
df_production = pd.DataFrame(production_data)
df_quality = pd.DataFrame(quality_data)
```

3. **Send Optimized Parameters to MES**:

```python
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer

# Initialize optimizer
optimizer = ProcessOptimizer()

# Get current parameters (simplified example)
current_params = {
    "input_amount": df_production["input_amount"].mean(),
    "energy_used": df_production["energy_used"].mean(),
    "cycle_time": df_production["cycle_time"].mean(),
    "efficiency": df_quality["efficiency"].mean(),
    "defect_rate": df_quality["defect_rate"].mean()
}

# Optimize parameters
optimized_params = optimizer.optimize_process_parameters(current_params)

# Update MES
update_result = mes_adapter.update_process_parameters(optimized_params)
```

## 7. Data Integration Patterns

### 7.1 Real-time Data Integration

For real-time data integration:

```python
import threading
import time
from queue import Queue

class RealTimeDataIntegrator:
    def __init__(self):
        self.data_queue = Queue()
        self.running = False
        self.thread = None

    def start(self):
        """Start the real-time data integrator."""
        self.running = True
        self.thread = threading.Thread(target=self._process_data)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the real-time data integrator."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def add_data(self, data):
        """Add data to the processing queue."""
        self.data_queue.put(data)

    def _process_data(self):
        """Process data from the queue."""
        while self.running:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get(block=False)
                    self._handle_data(data)
                    self.data_queue.task_done()
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing data: {e}")

    def _handle_data(self, data):
        """Handle a single data item."""
        # Process data and update relevant components
        data_type = data.get("type")

        if data_type == "production":
            # Process production data
            self._handle_production_data(data)
        elif data_type == "quality":
            # Process quality data
            self._handle_quality_data(data)
        elif data_type == "energy":
            # Process energy data
            self._handle_energy_data(data)
        elif data_type == "material":
            # Process material data
            self._handle_material_data(data)

    def _handle_production_data(self, data):
        """Handle production data."""
        # Implementation details
        pass

    def _handle_quality_data(self, data):
        """Handle quality data."""
        # Implementation details
        pass

    def _handle_energy_data(self, data):
        """Handle energy data."""
        # Implementation details
        pass

    def _handle_material_data(self, data):
        """Handle material data."""
        # Implementation details
        pass

# Usage example
integrator = RealTimeDataIntegrator()
integrator.start()

# Add data (from some source)
integrator.add_data({
    "type": "production",
    "timestamp": "2025-02-13T10:15:30",
    "data": {
        "input_amount": 100.0,
        "output_amount": 90.0,
        "energy_used": 50.0,
        "cycle_time": 60.0
    }
})
```

### 7.2 Batch Data Integration

For batch data processing:

```python
import pandas as pd
from datetime import datetime

class BatchDataProcessor:
    def __init__(self):
        self.analyzers = {}

    def register_analyzer(self, data_type, analyzer):
        """Register an analyzer for a data type."""
        self.analyzers[data_type] = analyzer

    def process_batch(self, data_files):
        """Process a batch of data files."""
        results = {}

        for data_type, file_path in data_files.items():
            if data_type not in self.analyzers:
                print(f"No analyzer registered for {data_type}")
                continue

            try:
                # Load data
                df = pd.read_csv(file_path)

                # Process with appropriate analyzer
                analyzer = self.analyzers[data_type]
                result = self._process_with_analyzer(analyzer, data_type, df)

                results[data_type] = result
            except Exception as e:
                print(f"Error processing {data_type}: {e}")

        return results

    def _process_with_analyzer(self, analyzer, data_type, df):
        """Process data with the appropriate analyzer."""
        if data_type == "production":
            return analyzer.analyze_batch_efficiency(df)
        elif data_type == "quality":
            return analyzer.analyze_defect_rates(df)
        elif data_type == "energy":
            return analyzer.calculate_carbon_footprint(df)
        elif data_type == "material":
            return analyzer.analyze_material_efficiency(df)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

# Usage example
from circman5.manufacturing.analyzers.efficiency import EfficiencyAnalyzer
from circman5.manufacturing.analyzers.quality import QualityAnalyzer
from circman5.manufacturing.analyzers.sustainability import SustainabilityAnalyzer

# Create processor
processor = BatchDataProcessor()

# Register analyzers
processor.register_analyzer("production", EfficiencyAnalyzer())
processor.register_analyzer("quality", QualityAnalyzer())
processor.register_analyzer("energy", SustainabilityAnalyzer())
processor.register_analyzer("material", SustainabilityAnalyzer())

# Process batch
results = processor.process_batch({
    "production": "production_data.csv",
    "quality": "quality_data.csv",
    "energy": "energy_data.csv",
    "material": "material_data.csv"
})
```

## 8. Custom Integration Examples

### 8.1 Integration with SoliTek Manufacturing System

```python
# SoliTek-specific adapter
class SoliTekAdapter:
    def __init__(self, config):
        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.logger = setup_logger("solitek_adapter")

    def get_production_data(self, date_from, date_to):
        """Get SoliTek production data."""
        try:
            # Implementation details depend on SoliTek API
            endpoint = f"{self.base_url}/api/production"
            params = {
                "dateFrom": date_from.isoformat(),
                "dateTo": date_to.isoformat(),
                "apiKey": self.api_key
            }

            # Make API request (implementation depends on HTTP client)
            # This is a placeholder
            response = self._make_api_request(endpoint, params)

            # Process response
            return self._process_production_data(response)
        except Exception as e:
            self.logger.error(f"Error getting SoliTek production data: {e}")
            raise

    def _make_api_request(self, endpoint, params):
        """Make API request to SoliTek system."""
        # Implementation details
        pass

    def _process_production_data(self, response):
        """Process SoliTek production data response."""
        # Implementation details
        pass

    def send_optimization_results(self, optimization_results):
        """Send optimization results to SoliTek system."""
        try:
            # Implementation details depend on SoliTek API
            endpoint = f"{self.base_url}/api/optimization"
            payload = {
                "apiKey": self.api_key,
                "results": optimization_results
            }

            # Make API request (implementation depends on HTTP client)
            # This is a placeholder
            response = self._make_api_post_request(endpoint, payload)

            return response
        except Exception as e:
            self.logger.error(f"Error sending optimization results to SoliTek: {e}")
            raise

    def _make_api_post_request(self, endpoint, payload):
        """Make POST API request to SoliTek system."""
        # Implementation details
        pass
```

### 8.2 Integration with Water Reuse Systems

```python
class WaterReuseAdapter:
    def __init__(self, config):
        self.system_url = config["system_url"]
        self.credentials = config["credentials"]
        self.logger = setup_logger("water_reuse_adapter")

    def get_water_usage_data(self, date_from, date_to):
        """Get water usage data."""
        # Implementation details
        pass

    def get_reuse_metrics(self, date_from, date_to):
        """Get water reuse metrics."""
        # Implementation details
        pass

    def send_optimization_parameters(self, parameters):
        """Send optimized parameters to water reuse system."""
        # Implementation details
        pass

    def integrate_with_sustainability_analyzer(self, sustainability_analyzer):
        """Integrate with sustainability analyzer."""
        # Get water data
        water_data = self.get_water_usage_data(date_from, date_to)
        reuse_metrics = self.get_reuse_metrics(date_from, date_to)

        # Convert to DataFrame
        import pandas as pd
        df_water = pd.DataFrame(water_data)

        # Calculate water efficiency
        water_efficiency = reuse_metrics["reuse_rate"]

        # Include in sustainability score
        sustainability_score = sustainability_analyzer.calculate_sustainability_score(
            material_efficiency=90.0,
            recycling_rate=80.0,
            energy_efficiency=85.0,
            water_efficiency=water_efficiency
        )

        return sustainability_score
```

## 9. Implementation Best Practices

### 9.1 Error Handling

When integrating systems, robust error handling is essential:

```python
def safe_integration(func):
    """Decorator for safe integration function execution."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"Connection error in {func.__name__}: {e}")
            # Implement retry logic or fallback
            return None
        except TimeoutError as e:
            logger.error(f"Timeout error in {func.__name__}: {e}")
            # Implement retry logic or fallback
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper

# Usage
@safe_integration
def get_external_data(external_system, query_params):
    """Get data from external system."""
    return external_system.query_data(query_params)
```

### 9.2 Data Validation

Always validate data from external systems:

```python
def validate_production_data(data):
    """Validate production data."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    required_columns = ["input_amount", "output_amount", "timestamp"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for NaN values
    if data[required_columns].isna().any().any():
        raise ValueError("Data contains NaN values in required columns")

    # Check for negative values
    if (data["input_amount"] < 0).any() or (data["output_amount"] < 0).any():
        raise ValueError("Data contains negative values for input or output amounts")

    return True
```

### 9.3 Performance Optimization

For large data integration:

```python
def process_large_dataset(file_path, chunksize=10000):
    """Process large dataset in chunks."""
    results = []

    # Process in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Process chunk
        chunk_result = process_chunk(chunk)
        results.append(chunk_result)

    # Combine results
    combined_result = combine_results(results)
    return combined_result

def process_chunk(chunk):
    """Process a single chunk of data."""
    # Implementation details
    pass

def combine_results(results):
    """Combine results from multiple chunks."""
    # Implementation details
    pass
```

### 9.4 Security Considerations

When integrating with external systems:

```python
import os
from cryptography.fernet import Fernet

class SecureIntegration:
    def __init__(self):
        # Get key from environment variable
        key_str = os.environ.get("INTEGRATION_KEY")
        if not key_str:
            raise ValueError("Integration key not found in environment variables")

        # Initialize encryption
        self.cipher = Fernet(key_str)

    def encrypt_payload(self, payload):
        """Encrypt payload for secure transmission."""
        payload_bytes = json.dumps(payload).encode()
        encrypted_bytes = self.cipher.encrypt(payload_bytes)
        return encrypted_bytes

    def decrypt_payload(self, encrypted_bytes):
        """Decrypt received payload."""
        decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
        return json.loads(decrypted_bytes.decode())

    def secure_request(self, url, payload, headers=None):
        """Make secure API request."""
        encrypted_payload = self.encrypt_payload(payload)

        if headers is None:
            headers = {}

        headers["Content-Type"] = "application/octet-stream"

        # Make request with encrypted payload
        # Implementation depends on HTTP client
        pass
```

## 10. Troubleshooting Integration Issues

### 10.1 Common Integration Problems

| Problem | Possible Causes | Solutions |
|---------|----------------|-----------|
| Connection failures | Network issues, invalid credentials | Check network, verify credentials, implement retry logic |
| Data format issues | Incompatible formats, schema changes | Implement data validation, add format conversion |
| Performance problems | Large data volumes, inefficient queries | Use chunked processing, optimize queries, cache results |
| Authentication failures | Expired tokens, incorrect credentials | Implement token refresh, verify credentials |
| Data inconsistencies | Timing issues, partial updates | Implement transactions, add data validation |

### 10.2 Diagnostic Procedures

When troubleshooting integration issues:

1. **Enable verbose logging**:

```python
import logging
logging.getLogger("integration").setLevel(logging.DEBUG)
```

2. **Test connections**:

```python
def test_system_connection(system_adapter):
    """Test connection to external system."""
    try:
        result = system_adapter.ping()
        print(f"Connection successful: {result}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
```

3. **Validate data flows**:

```python
def validate_data_flow(source_adapter, target_adapter, test_data):
    """Validate data flow between systems."""
    # Test data retrieval
    retrieved_data = source_adapter.get_data(test_data["query"])
    if retrieved_data is None:
        print("Failed to retrieve data from source")
        return False

    # Validate retrieved data
    if not validate_data(retrieved_data):
        print("Retrieved data validation failed")
        return False

    # Test data sending
    send_result = target_adapter.send_data(retrieved_data)
    if not send_result:
        print("Failed to send data to target")
        return False

    print("Data flow validation successful")
    return True
```

## 11. Conclusion

This integration guide provides comprehensive instructions for integrating the Circular Economy components of CIRCMAN5.0 with other system modules, external systems, and manufacturing processes. By following these guidelines, you can achieve seamless integration and maximize the value of the Circular Economy features.

The integration patterns, code examples, and best practices outlined in this guide will help you successfully implement and maintain Circular Economy functionality within your manufacturing environment, enabling resource optimization, waste reduction, and improved sustainability.

For further assistance, refer to the API documentation or contact the CIRCMAN5.0 development team.
