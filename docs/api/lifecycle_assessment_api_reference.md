# Lifecycle Assessment API Reference

## Overview

This document provides a comprehensive API reference for the Lifecycle Assessment (LCA) components of CIRCMAN5.0. These components enable the evaluation of environmental impacts throughout the lifecycle of photovoltaic (PV) manufacturing processes, supporting sustainable manufacturing decisions through integration with the Digital Twin system.

## Table of Contents

1. [Core Components](#core-components)
   - [LCAAnalyzer](#lcaanalyzer)
   - [LifeCycleImpact](#lifecycleimpact)
   - [LCAVisualizer](#lcavisualizer)
2. [Integration Components](#integration-components)
   - [LCAIntegration](#lcaintegration)
3. [Configuration Components](#configuration-components)
   - [ImpactFactorsAdapter](#impactfactorsadapter)
4. [Constants and Factors](#constants-and-factors)
   - [Material Impact Factors](#material-impact-factors)
   - [Energy Impact Factors](#energy-impact-factors)
   - [Other Impact Factors](#other-impact-factors)
5. [Integration Examples](#integration-examples)

## Core Components

### LCAAnalyzer

The `LCAAnalyzer` performs lifecycle assessment calculations for PV manufacturing.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initializes the LCA analyzer with default settings
- **Parameters**: None

#### Methods

```python
def calculate_manufacturing_impact(
    self, material_inputs: Dict[str, float], energy_consumption: float
) -> float
```
- **Description**: Calculate manufacturing phase impact
- **Parameters**:
  - `material_inputs`: Dictionary of material types and quantities
  - `energy_consumption`: Total energy consumed in kWh
- **Returns**: Manufacturing phase impact in kg CO2-eq
- **Raises**:
  - `ValueError`: If material_inputs is empty or energy_consumption is negative

```python
def _calculate_use_phase_impact(
    self,
    annual_generation: float,
    lifetime: float,
    grid_intensity: float,
    degradation_rate: Optional[float] = None,
) -> float
```
- **Description**: Calculate use phase impact considering power generation and degradation
- **Parameters**:
  - `annual_generation`: Annual energy generation (kWh)
  - `lifetime`: System lifetime (years)
  - `grid_intensity`: Grid carbon intensity (kg CO2/kWh)
  - `degradation_rate`: Optional annual degradation rate (as decimal)
- **Returns**: Use phase impact in kg CO2-eq

```python
def calculate_end_of_life_impact(
    self,
    material_inputs: Dict[str, float],
    recycling_rates: Dict[str, float],
    transport_distance: float,
) -> float
```
- **Description**: Calculate end of life impact including recycling benefits and transport impacts
- **Parameters**:
  - `material_inputs`: Dictionary of material types and quantities
  - `recycling_rates`: Dictionary of material types and their recycling rates
  - `transport_distance`: Transport distance in km
- **Returns**: End of life impact in kg CO2-eq

```python
def calculate_detailed_impacts(
    self, material_inputs: Dict[str, float], energy_consumption: float
) -> Dict[str, float]
```
- **Description**: Calculate detailed environmental impacts beyond carbon footprint
- **Parameters**:
  - `material_inputs`: Dictionary of material types and quantities
  - `energy_consumption`: Total energy consumed in kWh
- **Returns**: Dictionary of detailed environmental impacts

```python
def perform_full_lca(
    self,
    material_inputs: Dict[str, float],
    energy_consumption: float,
    lifetime_years: float,
    annual_energy_generation: float,
    grid_carbon_intensity: float,
    recycling_rates: Dict[str, float],
    transport_distance: float,
) -> LifeCycleImpact
```
- **Description**: Perform comprehensive lifecycle assessment
- **Parameters**:
  - `material_inputs`: Dictionary of material quantities
  - `energy_consumption`: Total energy consumption in kWh
  - `lifetime_years`: Expected system lifetime in years
  - `annual_energy_generation`: Expected annual energy generation in kWh
  - `grid_carbon_intensity`: Grid carbon intensity in kg CO2/kWh
  - `recycling_rates`: Dictionary of material recycling rates
  - `transport_distance`: Transport distance in km
- **Returns**: LifeCycleImpact object with impact results

```python
def _aggregate_material_inputs(
    self, material_data: pd.DataFrame
) -> Dict[str, float]
```
- **Description**: Aggregate material quantities by type
- **Parameters**:
  - `material_data`: DataFrame containing material flow data
- **Returns**: Dictionary mapping material types to total quantities

```python
def calculate_recycling_rates(
    self, material_data: pd.DataFrame
) -> Dict[str, float]
```
- **Description**: Calculate recycling rates from material flow data
- **Parameters**:
  - `material_data`: DataFrame containing material flow data
- **Returns**: Dictionary mapping material types to recycling rates

```python
def calculate_energy_generation(self, material_inputs: Dict[str, float]) -> float
```
- **Description**: Calculate expected annual energy generation
- **Parameters**:
  - `material_inputs`: Dictionary of material quantities
- **Returns**: Expected annual energy generation in kWh

```python
def save_results(
    self,
    impact: LifeCycleImpact,
    batch_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> None
```
- **Description**: Save LCA results to file
- **Parameters**:
  - `impact`: LifeCycleImpact object with results
  - `batch_id`: Optional batch identifier
  - `output_dir`: Optional output directory
- **Raises**:
  - `Exception`: If error saving results

### LifeCycleImpact

The `LifeCycleImpact` data class holds lifecycle impact assessment results.

#### Constructor

```python
def __init__(
    self,
    manufacturing_impact: float,
    use_phase_impact: float,
    end_of_life_impact: float,
    total_carbon_footprint: Optional[float] = None,
)
```

- **Description**: Initialize the lifecycle impact results
- **Parameters**:
  - `manufacturing_impact`: Impact from manufacturing phase (kg CO2-eq)
  - `use_phase_impact`: Impact from use phase (kg CO2-eq)
  - `end_of_life_impact`: Impact from end-of-life phase (kg CO2-eq)
  - `total_carbon_footprint`: Optional parameter (calculated automatically if not provided)

#### Properties

```python
@property
def total_carbon_footprint(self) -> float
```
- **Description**: Calculate total carbon footprint across all phases
- **Returns**: Total carbon footprint in kg CO2-eq

```python
@property
def total_impact(self) -> float
```
- **Description**: Alias for total_carbon_footprint to maintain compatibility
- **Returns**: Total carbon footprint in kg CO2-eq

#### Methods

```python
def to_dict(self) -> Dict[str, float]
```
- **Description**: Convert impact data to dictionary for saving
- **Returns**: Dictionary with impact values

### LCAVisualizer

The `LCAVisualizer` creates visualizations for lifecycle assessment results.

#### Constructor

```python
def __init__(self)
```

- **Description**: Initialize visualization settings
- **Parameters**: None

#### Methods

```python
def _handle_report_plotting(
    self,
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None
```
- **Description**: Handle time series plotting for reports with proper axis handling
- **Parameters**:
  - `df`: DataFrame to plot
  - `save_path`: Optional path to save plot
  - `title`: Plot title
  - `xlabel`: X-axis label
  - `ylabel`: Y-axis label

```python
def plot_impact_distribution(
    self, impact_data: Dict[str, float], save_path: Optional[str] = None
) -> None
```
- **Description**: Create a pie chart showing distribution of environmental impacts
- **Parameters**:
  - `impact_data`: Dictionary of impact categories and their values
  - `save_path`: Optional path to save the visualization

```python
def plot_lifecycle_comparison(
    self,
    manufacturing_impact: float,
    use_phase_impact: float,
    end_of_life_impact: float,
    save_path: Optional[str] = None,
) -> None
```
- **Description**: Create bar chart comparing impacts across lifecycle phases
- **Parameters**:
  - `manufacturing_impact`: Impact from manufacturing phase
  - `use_phase_impact`: Impact from use phase
  - `end_of_life_impact`: Impact from end-of-life phase
  - `save_path`: Optional path to save the visualization

```python
def plot_material_flow(
    self, material_data: pd.DataFrame, save_path: Optional[str] = None
) -> None
```
- **Description**: Create a material flow visualization showing inputs, waste, and recycling
- **Parameters**:
  - `material_data`: DataFrame containing material flow information
  - `save_path`: Optional path to save the visualization

```python
def plot_energy_consumption_trends(
    self, energy_data: pd.DataFrame, save_path: Optional[str] = None
) -> None
```
- **Description**: Create line plot showing energy consumption trends over time
- **Parameters**:
  - `energy_data`: DataFrame containing energy consumption data
  - `save_path`: Optional path to save the visualization

```python
def _get_visualization_path(self, filename: str) -> str
```
- **Description**: Get the proper path for saving visualizations
- **Parameters**:
  - `filename`: Base filename for the visualization
- **Returns**: Path to save the visualization

```python
def _ensure_save_path(self, filename: str, batch_id: Optional[str] = None) -> str
```
- **Description**: Ensure visualization is saved in the correct directory with proper naming
- **Parameters**:
  - `filename`: Base filename for the visualization
  - `batch_id`: Optional batch identifier for batch-specific visualizations
- **Returns**: Full path to save the visualization

```python
def create_comprehensive_report(
    self,
    impact_data: Dict[str, float],
    material_data: pd.DataFrame,
    energy_data: pd.DataFrame,
    output_dir: Union[str, Path],
    batch_id: Optional[str] = None,
) -> None
```
- **Description**: Generate all LCA-related visualizations
- **Parameters**:
  - `impact_data`: Dictionary of impact results
  - `material_data`: DataFrame of material flow data
  - `energy_data`: DataFrame of energy consumption data
  - `output_dir`: Output directory for visualizations
  - `batch_id`: Optional batch identifier
- **Raises**:
  - `Exception`: If error generating visualizations

## Integration Components

### LCAIntegration

The `LCAIntegration` class integrates LCA functionality with the Digital Twin.

#### Constructor

```python
def __init__(
    self,
    digital_twin: "DigitalTwin",
    lca_analyzer: Optional[LCAAnalyzer] = None,
    lca_visualizer: Optional[LCAVisualizer] = None,
)
```

- **Description**: Initialize the LCA integration
- **Parameters**:
  - `digital_twin`: Digital Twin instance to integrate with
  - `lca_analyzer`: Optional LCAAnalyzer instance (created if not provided)
  - `lca_visualizer`: Optional LCAVisualizer instance (created if not provided)

#### Methods

```python
def extract_material_data_from_state(
    self, state: Optional[Dict[str, Any]] = None
) -> pd.DataFrame
```
- **Description**: Extract material data from digital twin state for LCA calculations
- **Parameters**:
  - `state`: Optional state dictionary (uses current state if None)
- **Returns**: DataFrame with material flow data for LCA calculations

```python
def extract_energy_data_from_state(
    self, state: Optional[Dict[str, Any]] = None
) -> pd.DataFrame
```
- **Description**: Extract energy consumption data from digital twin state
- **Parameters**:
  - `state`: Optional state dictionary (uses current state if None)
- **Returns**: DataFrame with energy consumption data for LCA calculations

```python
def perform_lca_analysis(
    self,
    state: Optional[Dict[str, Any]] = None,
    batch_id: Optional[str] = None,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
) -> LifeCycleImpact
```
- **Description**: Perform lifecycle assessment analysis based on digital twin state
- **Parameters**:
  - `state`: Optional state dictionary (uses current state if None)
  - `batch_id`: Optional batch identifier for the analysis
  - `save_results`: Whether to save results to file
  - `output_dir`: Optional directory to save results
- **Returns**: LifeCycleImpact object with results

```python
def _aggregate_material_quantities(
    self, material_data: pd.DataFrame
) -> Dict[str, float]
```
- **Description**: Aggregate material quantities by type from material flow data
- **Parameters**:
  - `material_data`: DataFrame with material flow data
- **Returns**: Dictionary mapping material types to quantities

```python
def _save_lca_history(self) -> None
```
- **Description**: Save LCA analysis history to file using results_manager
- **Raises**:
  - `Exception`: If error saving history

```python
def generate_lca_report(self, num_results: int = 5) -> Dict[str, Any]
```
- **Description**: Generate a comprehensive LCA report
- **Parameters**:
  - `num_results`: Number of most recent results to include in the report
- **Returns**: Dictionary with report data

```python
def compare_scenarios(
    self,
    baseline_state: Dict[str, Any],
    alternative_state: Dict[str, Any],
    scenario_name: str = "scenario_comparison",
) -> Dict[str, Any]
```
- **Description**: Compare LCA impacts between two different digital twin states
- **Parameters**:
  - `baseline_state`: Baseline state for comparison
  - `alternative_state`: Alternative state for comparison
  - `scenario_name`: Name of the comparison scenario
- **Returns**: Dictionary with comparison results

```python
def simulate_lca_improvements(
    self, improvement_scenarios: Dict[str, Dict[str, float]]
) -> Mapping[str, Union[LifeCycleImpact, str]]
```
- **Description**: Simulate LCA improvements based on various improvement scenarios
- **Parameters**:
  - `improvement_scenarios`: Dictionary mapping scenario names to parameter adjustments
- **Returns**: Dictionary mapping scenario names to results

```python
def _apply_scenario_adjustments(
    self, base_state: Dict[str, Any], adjustments: Dict[str, float]
) -> Dict[str, Any]
```
- **Description**: Apply scenario adjustments to create a modified state
- **Parameters**:
  - `base_state`: Base state to modify
  - `adjustments`: Dictionary of adjustments to apply
- **Returns**: Modified state dictionary

```python
def _generate_scenarios_comparison(
    self, results: Dict[str, LifeCycleImpact]
) -> None
```
- **Description**: Generate a comparison report for multiple scenarios
- **Parameters**:
  - `results`: Dictionary mapping scenario names to LCA results
- **Raises**:
  - `Exception`: If error generating comparison

## Configuration Components

### ImpactFactorsAdapter

The `ImpactFactorsAdapter` provides access to lifecycle impact factors.

#### Constructor

```python
def __init__(self, config_path: Optional[Path] = None)
```

- **Description**: Initialize impact factors adapter
- **Parameters**:
  - `config_path`: Optional path to configuration file

#### Methods

```python
def load_config(self) -> Dict[str, Any]
```
- **Description**: Load impact factors configuration
- **Returns**: Impact factors configuration dictionary
- **Raises**:
  - `FileNotFoundError`: If config file not found
  - `ValueError`: If config is invalid

```python
def validate_config(self, config: Dict[str, Any]) -> bool
```
- **Description**: Validate impact factors configuration
- **Parameters**:
  - `config`: Configuration to validate
- **Returns**: True if configuration is valid

```python
def get_defaults(self) -> Dict[str, Any]
```
- **Description**: Get default impact factors configuration
- **Returns**: Default configuration dictionary

## Constants and Factors

### Material Impact Factors

```python
MATERIAL_IMPACT_FACTORS: Dict[str, float] = {
    # Silicon-based materials
    "silicon_wafer": 32.8,
    "polysilicon": 45.2,
    "metallization_paste": 23.4,
    # Glass and encapsulation
    "solar_glass": 0.9,
    "tempered_glass": 1.2,
    "eva_sheet": 2.6,
    "backsheet": 4.8,
    # Frame and mounting
    "aluminum_frame": 8.9,
    "mounting_structure": 2.7,
    # Electronics
    "junction_box": 7.5,
    "copper_wiring": 3.2,
}
```

- **Description**: Impact factors for materials in kg CO2-eq per kg material

### Energy Impact Factors

```python
ENERGY_IMPACT_FACTORS: Dict[str, float] = {
    "grid_electricity": 0.5,
    "natural_gas": 0.2,
    "solar_pv": 0.0,
    "wind": 0.0,
}
```

- **Description**: Impact factors for energy sources in kg CO2-eq per kWh

### Other Impact Factors

```python
TRANSPORT_IMPACT_FACTORS: Dict[str, float] = {
    "road": 0.062,
    "rail": 0.022,
    "sea": 0.008,
    "air": 0.602,
}

RECYCLING_BENEFIT_FACTORS: Dict[str, float] = {
    "silicon": -28.4,
    "glass": -0.7,
    "aluminum": -8.1,
    "copper": -2.8,
    "plastic": -1.8,
}

PROCESS_IMPACT_FACTORS: Dict[str, float] = {
    "wafer_cutting": 0.8,
    "cell_processing": 1.2,
    "module_assembly": 0.5,
    "testing": 0.1,
}

GRID_CARBON_INTENSITIES: Dict[str, float] = {
    "eu_average": 0.275,
    "us_average": 0.417,
    "china": 0.555,
    "india": 0.708,
}

DEGRADATION_RATES: Dict[str, float] = {
    "mono_perc": 0.5,
    "poly_bsf": 0.6,
    "thin_film": 0.7,
    "bifacial": 0.45,
}
```

- **Description**: Additional impact factors for LCA calculations

## Integration Examples

### Basic LCA Analysis

```python
from circman5.manufacturing.lifecycle.lca_analyzer import LCAAnalyzer
from circman5.manufacturing.lifecycle.visualizer import LCAVisualizer

# Initialize LCA components
lca_analyzer = LCAAnalyzer()
lca_visualizer = LCAVisualizer()

# Define material inputs
material_inputs = {
    "silicon_wafer": 100.0,
    "solar_glass": 200.0,
    "aluminum_frame": 50.0,
    "eva_sheet": 20.0,
    "backsheet": 15.0,
    "junction_box": 5.0,
    "copper_wiring": 2.0,
}

# Define energy consumption
energy_consumption = 1000.0  # kWh

# Define recycling rates
recycling_rates = {
    "silicon_wafer": 0.7,
    "solar_glass": 0.8,
    "aluminum_frame": 0.9,
    "eva_sheet": 0.5,
    "backsheet": 0.4,
    "junction_box": 0.6,
    "copper_wiring": 0.85,
}

# Perform LCA
impact = lca_analyzer.perform_full_lca(
    material_inputs=material_inputs,
    energy_consumption=energy_consumption,
    lifetime_years=25.0,
    annual_energy_generation=10000.0,
    grid_carbon_intensity=0.275,
    recycling_rates=recycling_rates,
    transport_distance=100.0,
)

# Print results
print(f"Manufacturing impact: {impact.manufacturing_impact:.2f} kg CO2-eq")
print(f"Use phase impact: {impact.use_phase_impact:.2f} kg CO2-eq")
print(f"End of life impact: {impact.end_of_life_impact:.2f} kg CO2-eq")
print(f"Total carbon footprint: {impact.total_carbon_footprint:.2f} kg CO2-eq")

# Save results
lca_analyzer.save_results(impact, batch_id="example_batch")

# Create visualizations
impact_data = impact.to_dict()
# ... (create material_data and energy_data DataFrames) ...
lca_visualizer.create_comprehensive_report(
    impact_data=impact_data,
    material_data=material_data,
    energy_data=energy_data,
    output_dir="results/lca",
    batch_id="example_batch",
)
```

### Digital Twin Integration

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.lca_integration import LCAIntegration

# Get Digital Twin instance
digital_twin = DigitalTwin()

# Initialize LCA integration
lca_integration = LCAIntegration(digital_twin)

# Perform LCA based on current Digital Twin state
impact = lca_integration.perform_lca_analysis()

# Print results
print(f"Total carbon footprint: {impact.total_carbon_footprint:.2f} kg CO2-eq")

# Generate report
report = lca_integration.generate_lca_report()

# Compare scenarios
baseline_state = digital_twin.get_current_state()
# ... (create alternative_state) ...
comparison = lca_integration.compare_scenarios(
    baseline_state=baseline_state,
    alternative_state=alternative_state,
    scenario_name="energy_optimization",
)

# Simulate improvements
improvement_scenarios = {
    "material_efficiency": {
        "silicon_wafer_quantity": 0.9,  # 10% reduction
        "solar_glass_quantity": 0.95,  # 5% reduction
    },
    "energy_efficiency": {
        "energy_efficiency": 1.2,  # 20% improvement
    },
    "combined": {
        "silicon_wafer_quantity": 0.9,
        "solar_glass_quantity": 0.95,
        "energy_efficiency": 1.2,
    },
}

scenario_results = lca_integration.simulate_lca_improvements(improvement_scenarios)

# Print scenario results
for scenario, result in scenario_results.items():
    if isinstance(result, str):
        print(f"{scenario}: {result}")
    else:
        print(f"{scenario}: {result.total_carbon_footprint:.2f} kg CO2-eq")
```

### Material Data Analysis

```python
import pandas as pd
from circman5.manufacturing.lifecycle.lca_analyzer import LCAAnalyzer

# Initialize LCA analyzer
lca_analyzer = LCAAnalyzer()

# Create material flow data
material_data = pd.DataFrame([
    {
        "batch_id": "batch_001",
        "timestamp": "2025-02-01T08:00:00",
        "material_type": "silicon_wafer",
        "quantity_used": 100.0,
        "waste_generated": 10.0,
        "recycled_amount": 8.0,
    },
    {
        "batch_id": "batch_001",
        "timestamp": "2025-02-01T08:00:00",
        "material_type": "solar_glass",
        "quantity_used": 200.0,
        "waste_generated": 15.0,
        "recycled_amount": 12.0,
    },
    # ... more materials ...
])

# Calculate recycling rates
recycling_rates = lca_analyzer.calculate_recycling_rates(material_data)

# Print recycling rates
for material, rate in recycling_rates.items():
    print(f"{material} recycling rate: {rate * 100:.1f}%")

# Aggregate material inputs
material_inputs = lca_analyzer._aggregate_material_inputs(material_data)

# Calculate manufacturing impact
impact = lca_analyzer.calculate_manufacturing_impact(
    material_inputs=material_inputs,
    energy_consumption=1000.0,
)

print(f"Manufacturing impact: {impact:.2f} kg CO2-eq")
```

### Visualization

```python
import pandas as pd
from circman5.manufacturing.lifecycle.visualizer import LCAVisualizer

# Initialize visualizer
visualizer = LCAVisualizer()

# Create impact data
impact_data = {
    "Manufacturing Impact": 5000.0,
    "Use Phase Impact": -68000.0,
    "End of Life Impact": 500.0,
    "Total Carbon Footprint": -62500.0,
}

# Create material flow data
material_data = pd.DataFrame([
    # ... material flow data ...
])

# Create energy consumption data
energy_data = pd.DataFrame([
    # ... energy consumption data ...
])

# Create visualizations
visualizer.plot_impact_distribution(
    impact_data=impact_data,
    save_path="results/impact_distribution.png",
)

visualizer.plot_lifecycle_comparison(
    manufacturing_impact=impact_data["Manufacturing Impact"],
    use_phase_impact=impact_data["Use Phase Impact"],
    end_of_life_impact=impact_data["End of Life Impact"],
    save_path="results/lifecycle_comparison.png",
)

visualizer.plot_material_flow(
    material_data=material_data,
    save_path="results/material_flow.png",
)

visualizer.plot_energy_consumption_trends(
    energy_data=energy_data,
    save_path="results/energy_trends.png",
)

# Create comprehensive report
visualizer.create_comprehensive_report(
    impact_data=impact_data,
    material_data=material_data,
    energy_data=energy_data,
    output_dir="results/lca_report",
    batch_id="example_batch",
)
```
