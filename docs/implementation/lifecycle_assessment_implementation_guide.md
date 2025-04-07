# Lifecycle Assessment Implementation Guide

## 1. Introduction

This guide provides detailed instructions for implementing and extending the Lifecycle Assessment (LCA) components within the CIRCMAN5.0 framework. The LCA system enables environmental impact assessment of PV manufacturing processes, supporting sustainable manufacturing decisions through integration with the Digital Twin system.

## 2. Prerequisites

Before implementing or extending the LCA system, ensure you have:

- Python 3.10 or higher
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
- CIRCMAN5.0 core framework
- Digital Twin components implemented
- Configuration system in place
- Results management system

## 3. System Overview

The LCA system consists of these primary components:

1. **LCA Analyzer**: Performs lifecycle calculations
2. **Impact Factors**: Provides environmental coefficients
3. **LCA Visualizer**: Creates visualizations
4. **LCA Integration**: Connects with Digital Twin

The implementation follows this structure:

```
src/circman5/manufacturing/lifecycle/
├── __init__.py
├── impact_factors.py     # Impact factors definition
├── lca_analyzer.py       # Core LCA calculation
└── visualizer.py         # Visualization capabilities

src/circman5/manufacturing/digital_twin/integration/
└── lca_integration.py    # Digital Twin integration

src/circman5/adapters/config/
└── impact_factors.py     # Configuration adapter

src/circman5/adapters/config/json/
└── impact_factors.json   # Impact factors configuration
```

## 4. Core Components Implementation

### 4.1 LCA Analyzer

The LCA Analyzer performs lifecycle impact calculations.

#### 4.1.1 Implementation Structure

Create `lca_analyzer.py` with the following structure:

```python
# src/circman5/manufacturing/lifecycle/lca_analyzer.py

from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
from pathlib import Path
from circman5.utils.results_manager import results_manager
from circman5.utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService

@dataclass
class LifeCycleImpact:
    """Data class to hold lifecycle impact assessment results."""

    manufacturing_impact: float
    use_phase_impact: float
    end_of_life_impact: float

    @property
    def total_carbon_footprint(self) -> float:
        """Calculate total carbon footprint across all phases."""
        return (
            self.manufacturing_impact + self.use_phase_impact + self.end_of_life_impact
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert impact data to dictionary for saving."""
        return {
            "Manufacturing Impact": self.manufacturing_impact,
            "Use Phase Impact": self.use_phase_impact,
            "End of Life Impact": self.end_of_life_impact,
            "Total Carbon Footprint": self.total_carbon_footprint,
        }

class LCAAnalyzer:
    """Analyzes lifecycle impacts of PV manufacturing."""

    def __init__(self):
        # Implementation here
        pass

    def calculate_manufacturing_impact(
        self, material_inputs: Dict[str, float], energy_consumption: float
    ) -> float:
        # Implementation here
        pass

    # Additional methods here
```

#### 4.1.2 Implementing Core Methods

Implement the core calculation methods:

```python
def calculate_manufacturing_impact(
    self, material_inputs: Dict[str, float], energy_consumption: float
) -> float:
    """
    Calculate manufacturing phase impact.

    Args:
        material_inputs: Dictionary of material types and quantities
        energy_consumption: Total energy consumed in kWh

    Returns:
        float: Manufacturing phase impact in kg CO2-eq

    Raises:
        ValueError: If material_inputs is empty or energy_consumption is negative
    """
    # Validate inputs
    if not material_inputs:
        raise ValueError("Material inputs dictionary cannot be empty")
    if energy_consumption < 0:
        raise ValueError("Energy consumption cannot be negative")

    impact = 0.0

    # Material impacts
    material_impact_factors = self.constants.get_constant(
        "impact_factors", "MATERIAL_IMPACT_FACTORS"
    )
    for material, quantity in material_inputs.items():
        if quantity < 0:
            raise ValueError(f"Material quantity cannot be negative: {material}")
        impact_factor = material_impact_factors.get(material, 0.0)
        impact += quantity * impact_factor

    # Energy impact
    energy_impact_factors = self.constants.get_constant(
        "impact_factors", "ENERGY_IMPACT_FACTORS"
    )
    impact += energy_consumption * energy_impact_factors.get(
        "grid_electricity", 0.5
    )

    return impact
```

Add the end-of-life impact calculation:

```python
def calculate_end_of_life_impact(
    self,
    material_inputs: Dict[str, float],
    recycling_rates: Dict[str, float],
    transport_distance: float,
) -> float:
    """
    Calculate end of life impact including recycling benefits and transport impacts.

    Args:
        material_inputs: Dictionary of material types and quantities
        recycling_rates: Dictionary of material types and their recycling rates
        transport_distance: Transport distance in km

    Returns:
        float: End of life impact in kg CO2-eq
    """
    impact = 0.0

    # Recycling benefits
    recycling_benefit_factors = self.constants.get_constant(
        "impact_factors", "RECYCLING_BENEFIT_FACTORS"
    )
    for material, quantity in material_inputs.items():
        recycling_rate = recycling_rates.get(material, 0.0)
        benefit_factor = recycling_benefit_factors.get(material, 0.0)
        impact += quantity * recycling_rate * benefit_factor

    # Transport impact
    total_mass = sum(material_inputs.values())
    transport_impact_factors = self.constants.get_constant(
        "impact_factors", "TRANSPORT_IMPACT_FACTORS"
    )
    transport_factor = transport_impact_factors.get(
        "road", 0.062
    )  # kg CO2-eq per tonne-km
    impact += (total_mass / 1000) * transport_distance * transport_factor

    return impact
```

Implement the full LCA calculation:

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
) -> LifeCycleImpact:
    """Perform comprehensive lifecycle assessment."""
    try:
        # Manufacturing phase impact
        manufacturing_impact = self.calculate_manufacturing_impact(
            material_inputs, energy_consumption
        )

        # Use phase impact
        use_phase_impact = self._calculate_use_phase_impact(
            annual_energy_generation, lifetime_years, grid_carbon_intensity
        )

        # End of life impact
        end_of_life_impact = self.calculate_end_of_life_impact(
            material_inputs, recycling_rates, transport_distance
        )

        return LifeCycleImpact(
            manufacturing_impact=manufacturing_impact,
            use_phase_impact=use_phase_impact,
            end_of_life_impact=end_of_life_impact,
        )

    except Exception as e:
        self.logger.error(f"Error performing LCA: {str(e)}")
        return LifeCycleImpact(0.0, 0.0, 0.0)
```

#### 4.1.3 Utility Methods

Implement utility methods for data processing and result saving:

```python
def calculate_recycling_rates(
    self, material_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate recycling rates from material flow data.

    Args:
        material_data: DataFrame containing material flow data

    Returns:
        Dict mapping material types to recycling rates
    """
    if material_data.empty:
        return {}

    try:
        recycling_rates = {}
        for material_type in material_data["material_type"].unique():
            material_subset = material_data[
                material_data["material_type"] == material_type
            ]
            waste = material_subset["waste_generated"].sum()
            recycled = material_subset["recycled_amount"].sum()

            if waste > 0:
                recycling_rates[material_type] = recycled / waste
            else:
                recycling_rates[material_type] = 0.0

        return recycling_rates
    except Exception as e:
        self.logger.error(f"Error calculating recycling rates: {str(e)}")
        return {}

def save_results(
    self,
    impact: LifeCycleImpact,
    batch_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Save LCA results to file."""
    try:
        if output_dir is None:
            reports_dir = results_manager.get_path("reports")
        else:
            reports_dir = output_dir

        filename = f"lca_impact_{batch_id}.xlsx" if batch_id else "lca_impact.xlsx"
        file_path = reports_dir / filename

        pd.DataFrame([impact.to_dict()]).to_excel(file_path, index=False)

        self.logger.info(f"Saved LCA results to {file_path}")
    except Exception as e:
        self.logger.error(f"Error saving LCA results: {str(e)}")
        raise
```

### 4.2 Impact Factors

Create `impact_factors.py` to define environmental impact factors:

```python
# src/circman5/manufacturing/lifecycle/impact_factors.py
"""
Standard impact factors for PV manufacturing LCA calculations.
Values based on published literature and industry standards.
"""

from typing import Dict

# Manufacturing phase impact factors (kg CO2-eq per kg material)
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

# Energy source impact factors (kg CO2-eq per kWh)
ENERGY_IMPACT_FACTORS: Dict[str, float] = {
    "grid_electricity": 0.5,
    "natural_gas": 0.2,
    "solar_pv": 0.0,
    "wind": 0.0,
}

# Transport impact factors (kg CO2-eq per tonne-km)
TRANSPORT_IMPACT_FACTORS: Dict[str, float] = {
    "road": 0.062,
    "rail": 0.022,
    "sea": 0.008,
    "air": 0.602,
}

# Recycling benefit factors (kg CO2-eq avoided per kg recycled)
RECYCLING_BENEFIT_FACTORS: Dict[str, float] = {
    "silicon": -28.4,
    "glass": -0.7,
    "aluminum": -8.1,
    "copper": -2.8,
    "plastic": -1.8,
}

# Manufacturing process impact factors (kg CO2-eq per process unit)
PROCESS_IMPACT_FACTORS: Dict[str, float] = {
    "wafer_cutting": 0.8,
    "cell_processing": 1.2,
    "module_assembly": 0.5,
    "testing": 0.1,
}

# Use phase factors
GRID_CARBON_INTENSITIES: Dict[str, float] = {
    "eu_average": 0.275,
    "us_average": 0.417,
    "china": 0.555,
    "india": 0.708,
}

# Performance degradation factors (% per year)
DEGRADATION_RATES: Dict[str, float] = {
    "mono_perc": 0.5,
    "poly_bsf": 0.6,
    "thin_film": 0.7,
    "bifacial": 0.45,
}
```

### 4.3 LCA Visualizer

Implement the LCA Visualizer for creating visualizations:

```python
# src/circman5/manufacturing/lifecycle/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List, Union
from pathlib import Path
from circman5.utils.logging_config import setup_logger
from circman5.utils.results_manager import results_manager

class LCAVisualizer:
    """Visualizes Life Cycle Assessment results."""

    def __init__(self):
        """Initialize visualization settings."""
        self.logger = setup_logger("lca_visualizer")
        self.viz_dir = results_manager.get_path("visualizations")

        # Configure plot styles
        plt.style.use("default")
        sns.set_theme(style="whitegrid")
        self.colors = sns.color_palette("husl", 8)

        # Set default figure parameters
        plt.rcParams.update(
            {
                "figure.autolayout": True,
                "figure.figsize": (12, 6),
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )

    # Add visualization methods:
    # - plot_impact_distribution
    # - plot_lifecycle_comparison
    # - plot_material_flow
    # - plot_energy_consumption_trends
    # - create_comprehensive_report
```

Add the key visualization methods:

```python
def plot_impact_distribution(
    self, impact_data: Dict[str, float], save_path: Optional[str] = None
) -> None:
    """
    Create a pie chart showing distribution of environmental impacts.

    Args:
        impact_data: Dictionary of impact categories and their values
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(10, 6))

    # Convert dictionary to lists explicitly
    values: List[float] = []
    labels: List[str] = []

    # Build lists from dictionary
    for key, value in impact_data.items():
        labels.append(str(key))
        values.append(float(value))

    # Create pie chart with explicit lists
    abs_values = [abs(v) for v in values]
    pie_labels = [
        f"{l} ({'+' if v >= 0 else '-'}{abs(v):.1f})"
        for l, v in zip(labels, values)
    ]

    plt.pie(abs_values, labels=pie_labels, autopct="%1.1f%%", colors=self.colors)
    plt.title(
        "Distribution of Environmental Impacts\n(Negative values indicate benefits)"
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
```

Implement the comprehensive report generation method:

```python
def create_comprehensive_report(
    self,
    impact_data: Dict[str, float],
    material_data: pd.DataFrame,
    energy_data: pd.DataFrame,
    output_dir: Union[str, Path],
    batch_id: Optional[str] = None,
) -> None:
    """Generate all LCA-related visualizations."""
    try:
        # Use provided output directory or fall back to results_manager
        viz_dir = Path(output_dir) if output_dir else self.viz_dir
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Impact distribution plot
        self.plot_impact_distribution(
            impact_data, save_path=str(viz_dir / "impact_distribution.png")
        )

        # Lifecycle comparison
        self.plot_lifecycle_comparison(
            impact_data["Manufacturing Impact"],
            impact_data["Use Phase Impact"],
            impact_data["End of Life Impact"],
            save_path=str(viz_dir / "lifecycle_comparison.png"),
        )

        # Material flow with robust handling
        if not material_data.empty:
            material_data_timeseries = material_data.set_index("timestamp")
            self._handle_report_plotting(
                material_data_timeseries,
                save_path=str(viz_dir / "material_flow.png"),
                title="Material Flow Analysis",
                xlabel="Time",
                ylabel="Amount (kg)",
            )

        # Energy trends with robust handling
        if not energy_data.empty:
            energy_data_timeseries = energy_data.set_index("timestamp")
            self._handle_report_plotting(
                energy_data_timeseries,
                save_path=str(viz_dir / "energy_trends.png"),
                title="Energy Consumption Trends",
                xlabel="Time",
                ylabel="Energy (kWh)",
            )

        self.logger.info(f"Generated visualizations in {viz_dir}")

    except Exception as e:
        self.logger.error(f"Error generating LCA visualizations: {str(e)}")
        raise
```

## 5. Digital Twin Integration

### 5.1 LCA Integration

Create the LCA integration module to connect with the Digital Twin:

```python
# src/circman5/manufacturing/digital_twin/integration/lca_integration.py

from typing import Dict, Any, List, Optional, Union, Tuple, Mapping, TYPE_CHECKING

# Only import type hints during type checking to avoid circular imports
if TYPE_CHECKING:
    from ..core.twin_core import DigitalTwin

import pandas as pd
import datetime
import json
from pathlib import Path

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from ...lifecycle.lca_analyzer import LCAAnalyzer, LifeCycleImpact
from ...lifecycle.visualizer import LCAVisualizer

class LCAIntegration:
    """
    Integrates the digital twin with lifecycle assessment components.
    """

    def __init__(
        self,
        digital_twin: "DigitalTwin",
        lca_analyzer: Optional[LCAAnalyzer] = None,
        lca_visualizer: Optional[LCAVisualizer] = None,
    ):
        """
        Initialize the LCA integration.

        Args:
            digital_twin: Digital Twin instance to integrate with
            lca_analyzer: Optional LCAAnalyzer instance (created if not provided)
            lca_visualizer: Optional LCAVisualizer instance (created if not provided)
        """
        self.digital_twin = digital_twin
        self.constants = ConstantsService()
        self.logger = setup_logger("lca_integration")

        # Initialize LCA components
        self.lca_analyzer = lca_analyzer or LCAAnalyzer()
        self.lca_visualizer = lca_visualizer or LCAVisualizer()

        # Initialize storage for LCA results
        self.lca_results_history: List[Dict[str, Any]] = []

        self.logger.info("LCA Integration initialized")

    # Add key integration methods here:
    # - extract_material_data_from_state
    # - extract_energy_data_from_state
    # - perform_lca_analysis
    # - compare_scenarios
    # - simulate_lca_improvements
```

Implement the data extraction methods:

```python
def extract_material_data_from_state(
    self, state: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Extract material data from digital twin state for LCA calculations.

    Args:
        state: Optional state dictionary (uses current state if None)

    Returns:
        pd.DataFrame: Material flow data frame for LCA calculations
    """
    # Get current state if not provided
    if state is None:
        state = self.digital_twin.get_current_state()

    # Check if state is None after retrieval
    if state is None:
        self.logger.warning(
            "Received None state in extract_material_data_from_state"
        )
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "batch_id",
                "timestamp",
                "material_type",
                "quantity_used",
                "waste_generated",
                "recycled_amount",
            ]
        )

    # Extract timestamp
    timestamp = state.get("timestamp", datetime.datetime.now().isoformat())
    if isinstance(timestamp, str):
        try:
            timestamp_obj = datetime.datetime.fromisoformat(timestamp)
        except ValueError:
            timestamp_obj = datetime.datetime.now()
    else:
        timestamp_obj = timestamp

    # Extract batch ID (use timestamp if not available)
    batch_id = state.get(
        "batch_id", f"batch_{timestamp_obj.strftime('%Y%m%d%H%M%S')}"
    )

    # Initialize empty dataframe
    material_data = []

    # Extract materials data
    if "materials" in state:
        materials = state["materials"]
        for material_name, material_info in materials.items():
            if isinstance(material_info, dict):
                # Calculate waste and recycling based on inventory
                inventory = float(material_info.get("inventory", 0))
                quality = float(material_info.get("quality", 0.9))

                # Estimate waste as a function of quality (lower quality = more waste)
                waste_rate = max(0.05, 1.0 - quality)
                waste_generated = inventory * waste_rate

                # Estimate recycling based on material type
                # Map material name to recycling material type
                material_type_mapping = {
                    "silicon_wafer": "silicon",
                    "solar_glass": "glass",
                    "tempered_glass": "glass",
                    "aluminum_frame": "aluminum",
                    "copper_wiring": "copper",
                    "backsheet": "plastic",
                    "eva_sheet": "plastic",
                    "junction_box": "plastic",
                }

                # Get recycling rate based on material type
                recycling_material = material_type_mapping.get(
                    material_name, "plastic"
                )
                # Use a default recycling rate of 0.5 if not found
                recycling_rate = 0.5

                # Calculate recycled amount
                recycled_amount = waste_generated * recycling_rate

                material_data.append(
                    {
                        "batch_id": batch_id,
                        "timestamp": timestamp_obj,
                        "material_type": material_name,
                        "quantity_used": inventory,
                        "waste_generated": waste_generated,
                        "recycled_amount": recycled_amount,
                    }
                )

    # Create dataframe
    if not material_data:
        # Create empty dataframe with required columns
        return pd.DataFrame(
            columns=[
                "batch_id",
                "timestamp",
                "material_type",
                "quantity_used",
                "waste_generated",
                "recycled_amount",
            ]
        )

    df = pd.DataFrame(material_data)
    self.logger.debug(f"Extracted material data from state: {len(df)} rows")
    return df
```

Implement the LCA analysis method:

```python
def perform_lca_analysis(
    self,
    state: Optional[Dict[str, Any]] = None,
    batch_id: Optional[str] = None,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
) -> LifeCycleImpact:
    """
    Perform lifecycle assessment analysis based on digital twin state.

    Args:
        state: Optional state dictionary (uses current state if None)
        batch_id: Optional batch identifier for the analysis
        save_results: Whether to save results to file
        output_dir: Optional directory to save results (uses results_manager if None)

    Returns:
        LifeCycleImpact: Results of the lifecycle assessment
    """
    try:
        # Get current state if not provided
        if state is None:
            state = self.digital_twin.get_current_state()

        # Check if state is None after retrieval
        if state is None:
            self.logger.warning("Cannot perform LCA analysis: state is None")
            # Return empty impact object
            return LifeCycleImpact(
                manufacturing_impact=0.0,
                use_phase_impact=0.0,
                end_of_life_impact=0.0,
            )

        # Extract timestamp for batch_id if not provided
        if batch_id is None:
            timestamp = state.get("timestamp", datetime.datetime.now().isoformat())
            if isinstance(timestamp, str):
                try:
                    timestamp_obj = datetime.datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp_obj = datetime.datetime.now()
            else:
                timestamp_obj = timestamp

            batch_id = f"batch_{timestamp_obj.strftime('%Y%m%d%H%M%S')}"

        # Extract material and energy data
        material_data = self.extract_material_data_from_state(state)
        energy_data = self.extract_energy_data_from_state(state)

        # Get aggregated material inputs
        material_inputs = self._aggregate_material_quantities(material_data)

        # Get total energy consumption
        total_energy = (
            energy_data["energy_consumption"].sum()
            if not energy_data.empty
            else 0.0
        )

        # Calculate recycling rates
        recycling_rates = self.lca_analyzer.calculate_recycling_rates(material_data)

        # Set default parameters for LCA calculation
        lifetime_years = 25.0  # Default PV panel lifetime
        transport_distance = 100.0  # Default transport distance in km

        # Calculate annual energy generation based on material inputs
        annual_energy_generation = self.lca_analyzer.calculate_energy_generation(
            material_inputs
        )

        # Get grid carbon intensity from constants
        grid_intensities = self.constants.get_constant(
            "impact_factors", "GRID_CARBON_INTENSITIES"
        )
        grid_carbon_intensity = grid_intensities.get("eu_average", 0.275)

        # Perform lifecycle assessment
        impact = self.lca_analyzer.perform_full_lca(
            material_inputs=material_inputs,
            energy_consumption=total_energy,
            lifetime_years=lifetime_years,
            annual_energy_generation=annual_energy_generation,
            grid_carbon_intensity=grid_carbon_intensity,
            recycling_rates=recycling_rates,
            transport_distance=transport_distance,
        )

        # Save results if requested
        if save_results:
            if output_dir is None:
                # Use results_manager
                lca_results_dir = results_manager.get_path("lca_results")
            else:
                lca_results_dir = output_dir

            self.lca_analyzer.save_results(
                impact=impact, batch_id=batch_id, output_dir=lca_results_dir
            )

            # Create visualizations
            self.lca_visualizer.create_comprehensive_report(
                impact_data=impact.to_dict(),
                material_data=material_data,
                energy_data=energy_data,
                output_dir=lca_results_dir,
                batch_id=batch_id,
            )

        # Store results in history
        self.lca_results_history.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "batch_id": batch_id,
                "impact": impact.to_dict(),
                "material_inputs": material_inputs,
                "energy_consumption": total_energy,
            }
        )

        # Save history to file
        self._save_lca_history()

        self.logger.info(
            f"LCA analysis completed for {batch_id}. "
            f"Total impact: {impact.total_carbon_footprint:.2f} kg CO2-eq"
        )

        return impact

    except Exception as e:
        self.logger.error(f"Error performing LCA analysis: {str(e)}")
        # Return empty impact object
        return LifeCycleImpact(
            manufacturing_impact=0.0, use_phase_impact=0.0, end_of_life_impact=0.0
        )
```

### 5.2 Scenario Comparison Method

Implement the scenario comparison functionality:

```python
def compare_scenarios(
    self,
    baseline_state: Dict[str, Any],
    alternative_state: Dict[str, Any],
    scenario_name: str = "scenario_comparison",
) -> Dict[str, Any]:
    """
    Compare LCA impacts between two different digital twin states.

    Args:
        baseline_state: Baseline state for comparison
        alternative_state: Alternative state for comparison
        scenario_name: Name of the comparison scenario

    Returns:
        Dict[str, Any]: Comparison results
    """
    try:
        # Perform LCA for both scenarios
        baseline_impact = self.perform_lca_analysis(
            state=baseline_state,
            batch_id=f"{scenario_name}_baseline",
            save_results=True,
        )

        alternative_impact = self.perform_lca_analysis(
            state=alternative_state,
            batch_id=f"{scenario_name}_alternative",
            save_results=True,
        )

        # Calculate differences
        impact_differences = {
            "manufacturing_impact": alternative_impact.manufacturing_impact
            - baseline_impact.manufacturing_impact,
            "use_phase_impact": alternative_impact.use_phase_impact
            - baseline_impact.use_phase_impact,
            "end_of_life_impact": alternative_impact.end_of_life_impact
            - baseline_impact.end_of_life_impact,
            "total_carbon_footprint": alternative_impact.total_carbon_footprint
            - baseline_impact.total_carbon_footprint,
        }

        # Calculate percentage differences
        percent_differences = {}
        for key, value in impact_differences.items():
            baseline_value = getattr(baseline_impact, key, 0)
            if baseline_value != 0:
                percent_differences[key] = (value / baseline_value) * 100
            else:
                percent_differences[key] = 0

        # Create comparison report
        comparison = {
            "timestamp": datetime.datetime.now().isoformat(),
            "scenario_name": scenario_name,
            "baseline_impact": baseline_impact.to_dict(),
            "alternative_impact": alternative_impact.to_dict(),
            "absolute_differences": impact_differences,
            "percent_differences": percent_differences,
        }

        # Save comparison report
        report_file = f"lca_comparison_{scenario_name}.json"
        with open(report_file, "w") as f:
            json.dump(comparison, f, indent=2)

        results_manager.save_file(Path(report_file), "reports")
        Path(report_file).unlink()

        self.logger.info(
            f"LCA comparison completed for {scenario_name}. "
            f"Total impact difference: {impact_differences['total_carbon_footprint']:.2f} kg CO2-eq "
            f"({percent_differences['total_carbon_footprint']:.2f}%)"
        )

        return comparison

    except Exception as e:
        self.logger.error(f"Error comparing scenarios: {str(e)}")
        return {"error": str(e)}
```

## 6. Configuration Implementation

### 6.1 Impact Factors Adapter

Create the impact factors configuration adapter:

```python
# src/circman5/adapters/config/impact_factors.py

from pathlib import Path
from typing import Dict, Any, Optional
import json

from ..base.adapter_base import ConfigAdapterBase


class ImpactFactorsAdapter(ConfigAdapterBase):
    """Adapter for Life Cycle Assessment (LCA) impact factors configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize impact factors adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "impact_factors.json"
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load impact factors configuration.

        Returns:
            Dict[str, Any]: Impact factors configuration

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        if not self.config_path.exists():
            self.logger.warning(
                f"Config file not found: {self.config_path}, using defaults"
            )
            return self.get_defaults()

        return self._load_json_config(self.config_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate impact factors configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        required_factors = {
            "MATERIAL_IMPACT_FACTORS",
            "ENERGY_IMPACT_FACTORS",
            "RECYCLING_BENEFIT_FACTORS",
            "TRANSPORT_IMPACT_FACTORS",
            "PROCESS_IMPACT_FACTORS",
            "GRID_CARBON_INTENSITIES",
            "DEGRADATION_RATES",
        }

        # Check required top-level keys
        return all(key in config for key in required_factors)

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default impact factors configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        # Add a comprehensive set of default values here
        # This should match the structure in impact_factors.json
        return {
            "MATERIAL_IMPACT_FACTORS": {
                "silicon_wafer": 32.8,
                "polysilicon": 45.2,
                # Additional materials...
            },
            "ENERGY_IMPACT_FACTORS": {
                "grid_electricity": 0.5,
                "natural_gas": 0.2,
                "solar_pv": 0.0,
                "wind": 0.0,
            },
            # Additional factor categories...
        }
```

### 6.2 JSON Configuration

Create the impact factors configuration file:

```json
// src/circman5/adapters/config/json/impact_factors.json
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
        "plastic": -1.8
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
    }
}
```

## 7. Testing Implementation

### 7.1 LCA Analyzer Tests

Create unit tests for the LCA Analyzer:

```python
# tests/unit/manufacturing/lifecycle/test_lca_analyzer.py

import unittest
import pandas as pd
from circman5.manufacturing.lifecycle.lca_analyzer import LCAAnalyzer, LifeCycleImpact

class TestLCAAnalyzer(unittest.TestCase):
    """Tests for the LCAAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = LCAAnalyzer()

        # Test material inputs
        self.material_inputs = {
            "silicon_wafer": 100.0,
            "solar_glass": 200.0,
            "aluminum_frame": 50.0,
        }

        # Test energy consumption
        self.energy_consumption = 1000.0

        # Test recycling rates
        self.recycling_rates = {
            "silicon_wafer": 0.7,
            "solar_glass": 0.8,
            "aluminum_frame": 0.9,
        }

        # Test material data
        self.material_data = pd.DataFrame([
            {
                "batch_id": "batch_001",
                "timestamp": "2025-02-01T08:00:00",
                "material_type": "silicon_wafer",
                "quantity_used": 100.0,
                "waste_generated": 10.0,
                "recycled_amount": 7.0,
            },
            {
                "batch_id": "batch_001",
                "timestamp": "2025-02-01T08:00:00",
                "material_type": "solar_glass",
                "quantity_used": 200.0,
                "waste_generated": 20.0,
                "recycled_amount": 16.0,
            },
        ])

    def test_calculate_manufacturing_impact(self):
        """Test manufacturing impact calculation."""
        impact = self.analyzer.calculate_manufacturing_impact(
            self.material_inputs, self.energy_consumption
        )

        # Verify impact is positive and sensible
        self.assertGreater(impact, 0)
        # Simple sanity check - actual value will depend on impact factors
        self.assertLess(impact, 10000)

    def test_calculate_end_of_life_impact(self):
        """Test end of life impact calculation."""
        impact = self.analyzer.calculate_end_of_life_impact(
            self.material_inputs,
            self.recycling_rates,
            transport_distance=100.0
        )

        # End of life impact should be negative (benefit) or small positive
        self.assertLess(impact, 100)

    def test_perform_full_lca(self):
        """Test full LCA calculation."""
        impact = self.analyzer.perform_full_lca(
            material_inputs=self.material_inputs,
            energy_consumption=self.energy_consumption,
            lifetime_years=25.0,
            annual_energy_generation=10000.0,
            grid_carbon_intensity=0.275,
            recycling_rates=self.recycling_rates,
            transport_distance=100.0,
        )

        # Check result type
        self.assertIsInstance(impact, LifeCycleImpact)

        # Check that properties are accessible
        self.assertIsInstance(impact.manufacturing_impact, float)
        self.assertIsInstance(impact.use_phase_impact, float)
        self.assertIsInstance(impact.end_of_life_impact, float)
        self.assertIsInstance(impact.total_carbon_footprint, float)

    def test_calculate_recycling_rates(self):
        """Test recycling rate calculation from material data."""
        rates = self.analyzer.calculate_recycling_rates(self.material_data)

        # Check recycling rates
        self.assertIn("silicon_wafer", rates)
        self.assertIn("solar_glass", rates)

        # Check values (should match waste and recycled amounts in test data)
        self.assertAlmostEqual(rates["silicon_wafer"], 0.7, delta=0.01)
        self.assertAlmostEqual(rates["solar_glass"], 0.8, delta=0.01)

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_manufacturing_impact({}, 1000.0)

        # Empty material data should return empty recycling rates
        empty_df = pd.DataFrame()
        rates = self.analyzer.calculate_recycling_rates(empty_df)
        self.assertEqual(rates, {})
```

### 7.2 LCA Integration Tests

Create integration tests for the LCA system:

```python
# tests/integration/test_lca_integration.py

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import datetime
from pathlib import Path

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.lca_integration import LCAIntegration
from circman5.manufacturing.lifecycle.lca_analyzer import LCAAnalyzer, LifeCycleImpact

class TestLCAIntegration(unittest.TestCase):
    """Integration tests for the LCA system with Digital Twin."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock digital twin with test state
        self.digital_twin = MagicMock(spec=DigitalTwin)
        self.test_state = {
            "timestamp": datetime.datetime.now().isoformat(),
            "batch_id": "test_batch",
            "materials": {
                "silicon_wafer": {"inventory": 100.0, "quality": 0.9},
                "solar_glass": {"inventory": 200.0, "quality": 0.95},
                "aluminum_frame": {"inventory": 50.0, "quality": 0.98},
            },
            "production_line": {
                "status": "running",
                "energy_consumption": 1000.0,
                "energy_source": "grid_electricity",
            }
        }
        self.digital_twin.get_current_state.return_value = self.test_state

        # Initialize LCA integration with real analyzer and visualizer
        self.lca_analyzer = LCAAnalyzer()
        self.lca_visualizer = MagicMock()  # Mock visualizer to avoid file operations
        self.lca_integration = LCAIntegration(
            self.digital_twin,
            self.lca_analyzer,
            self.lca_visualizer
        )

    def test_extract_material_data(self):
        """Test material data extraction from Digital Twin state."""
        # Extract material data
        material_data = self.lca_integration.extract_material_data_from_state(self.test_state)

        # Check that data is not empty
        self.assertFalse(material_data.empty)

        # Check that expected columns exist
        expected_columns = [
            "batch_id", "timestamp", "material_type",
            "quantity_used", "waste_generated", "recycled_amount"
        ]
        for col in expected_columns:
            self.assertIn(col, material_data.columns)

        # Check that expected materials exist
        material_types = material_data["material_type"].unique()
        self.assertIn("silicon_wafer", material_types)
        self.assertIn("solar_glass", material_types)
        self.assertIn("aluminum_frame", material_types)

    def test_extract_energy_data(self):
        """Test energy data extraction from Digital Twin state."""
        # Extract energy data
        energy_data = self.lca_integration.extract_energy_data_from_state(self.test_state)

        # Check that data is not empty
        self.assertFalse(energy_data.empty)

        # Check expected columns
        expected_columns = [
            "batch_id", "timestamp", "energy_source",
            "energy_consumption", "process_stage"
        ]
        for col in expected_columns:
            self.assertIn(col, energy_data.columns)

        # Check energy consumption value
        self.assertEqual(energy_data["energy_consumption"].sum(), 1000.0)

    def test_perform_lca_analysis(self):
        """Test LCA analysis with Digital Twin state."""
        # Mock save methods to avoid file operations
        with patch.object(self.lca_analyzer, 'save_results') as mock_save:
            # Perform LCA analysis
            impact = self.lca_integration.perform_lca_analysis(
                state=self.test_state,
                batch_id="test_analysis",
                save_results=True
            )

            # Check that save was called
            mock_save.assert_called_once()

        # Check result type
        self.assertIsInstance(impact, LifeCycleImpact)

        # Check that results are stored in history
        self.assertEqual(len(self.lca_integration.lca_results_history), 1)
        self.assertEqual(self.lca_integration.lca_results_history[0]["batch_id"], "test_analysis")

    def test_compare_scenarios(self):
        """Test scenario comparison."""
        # Create alternative state with improved efficiency
        alternative_state = self.test_state.copy()
        alternative_state["production_line"]["energy_consumption"] = 800.0  # 20% reduction

        # Mock perform_lca_analysis to avoid duplicating testing
        original_method = self.lca_integration.perform_lca_analysis
        self.lca_integration.perform_lca_analysis = MagicMock()
        self.lca_integration.perform_lca_analysis.side_effect = [
            LifeCycleImpact(5000.0, -50000.0, 500.0),  # Baseline
            LifeCycleImpact(4500.0, -50000.0, 500.0)   # Alternative (lower manufacturing impact)
        ]

        # Compare scenarios
        comparison = self.lca_integration.compare_scenarios(
            baseline_state=self.test_state,
            alternative_state=alternative_state,
            scenario_name="energy_efficiency"
        )

        # Restore original method
        self.lca_integration.perform_lca_analysis = original_method

        # Check comparison result
        self.assertEqual(comparison["scenario_name"], "energy_efficiency")
        self.assertIn("absolute_differences", comparison)
        self.assertIn("percent_differences", comparison)

        # Check that manufacturing impact is reduced
        self.assertEqual(comparison["absolute_differences"]["manufacturing_impact"], -500.0)
```

## 8. Extension Implementation

### 8.1 Adding New Impact Factors

To extend the system with new impact factors:

1. Add new factors to the impact factors module:

```python
# Add to src/circman5/manufacturing/lifecycle/impact_factors.py

# Water consumption factors (m³ per kg material)
WATER_CONSUMPTION_FACTORS: Dict[str, float] = {
    "silicon_wafer": 0.2,
    "polysilicon": 0.5,
    "metallization_paste": 0.1,
    "solar_glass": 0.05,
    "tempered_glass": 0.06,
    "eva_sheet": 0.03,
    "backsheet": 0.04,
    "aluminum_frame": 0.15,
    "mounting_structure": 0.1,
    "junction_box": 0.02,
    "copper_wiring": 0.08,
}

# Land use factors (m² per kg material)
LAND_USE_FACTORS: Dict[str, float] = {
    "silicon_wafer": 0.1,
    "polysilicon": 0.15,
    "metallization_paste": 0.05,
    "solar_glass": 0.02,
    "tempered_glass": 0.025,
    "eva_sheet": 0.01,
    "backsheet": 0.015,
    "aluminum_frame": 0.07,
    "mounting_structure": 0.05,
    "junction_box": 0.01,
    "copper_wiring": 0.03,
}
```

2. Update the impact factors configuration JSON:

```json
// Add to src/circman5/adapters/config/json/impact_factors.json
{
    "WATER_CONSUMPTION_FACTORS": {
        "silicon_wafer": 0.2,
        "polysilicon": 0.5,
        "metallization_paste": 0.1,
        "solar_glass": 0.05,
        "tempered_glass": 0.06,
        "eva_sheet": 0.03,
        "backsheet": 0.04,
        "aluminum_frame": 0.15,
        "mounting_structure": 0.1,
        "junction_box": 0.02,
        "copper_wiring": 0.08
    },
    "LAND_USE_FACTORS": {
        "silicon_wafer": 0.1,
        "polysilicon": 0.15,
        "metallization_paste": 0.05,
        "solar_glass": 0.02,
        "tempered_glass": 0.025,
        "eva_sheet": 0.01,
        "backsheet": 0.015,
        "aluminum_frame": 0.07,
        "mounting_structure": 0.05,
        "junction_box": 0.01,
        "copper_wiring": 0.03
    }
}
```

3. Update the adapter validation:

```python
# Update in src/circman5/adapters/config/impact_factors.py

def validate_config(self, config: Dict[str, Any]) -> bool:
    """
    Validate impact factors configuration.

    Args:
        config: Configuration to validate

    Returns:
        bool: True if configuration is valid
    """
    required_factors = {
        "MATERIAL_IMPACT_FACTORS",
        "ENERGY_IMPACT_FACTORS",
        "RECYCLING_BENEFIT_FACTORS",
        "TRANSPORT_IMPACT_FACTORS",
        "PROCESS_IMPACT_FACTORS",
        "GRID_CARBON_INTENSITIES",
        "DEGRADATION_RATES",
        "WATER_CONSUMPTION_FACTORS",  # New factor
        "LAND_USE_FACTORS",  # New factor
    }

    # Check required top-level keys
    return all(key in config for key in required_factors)
```

4. Add calculation methods to the analyzer:

```python
# Add to src/circman5/manufacturing/lifecycle/lca_analyzer.py

def calculate_water_consumption(
    self, material_inputs: Dict[str, float]
) -> float:
    """
    Calculate water consumption for manufacturing.

    Args:
        material_inputs: Dictionary of material types and quantities

    Returns:
        float: Water consumption in m³
    """
    consumption = 0.0

    # Material water consumption
    water_factors = self.constants.get_constant(
        "impact_factors", "WATER_CONSUMPTION_FACTORS"
    )
    for material, quantity in material_inputs.items():
        factor = water_factors.get(material, 0.0)
        consumption += quantity * factor

    return consumption

def calculate_land_use(
    self, material_inputs: Dict[str, float]
) -> float:
    """
    Calculate land use for manufacturing.

    Args:
        material_inputs: Dictionary of material types and quantities

    Returns:
        float: Land use in m²
    """
    land_use = 0.0

    # Material land use
    land_factors = self.constants.get_constant(
        "impact_factors", "LAND_USE_FACTORS"
    )
    for material, quantity in material_inputs.items():
        factor = land_factors.get(material, 0.0)
        land_use += quantity * factor

    return land_use
```

### 8.2 Creating a New Environmental Impact Class

To represent multiple environmental impacts, create a new class:

```python
# src/circman5/manufacturing/lifecycle/lca_analyzer.py

@dataclass
class DetailedEnvironmentalImpact:
    """Data class to hold detailed environmental impact assessment results."""

    # Carbon footprint
    manufacturing_carbon: float
    use_phase_carbon: float
    end_of_life_carbon: float

    # Water consumption
    water_consumption: float

    # Land use
    land_use: float

    # Energy consumption
    energy_consumption: float

    # Material intensity
    material_intensity: float

    # Waste generation
    waste_generation: float

    @property
    def total_carbon_footprint(self) -> float:
        """Calculate total carbon footprint."""
        return (
            self.manufacturing_carbon +
            self.use_phase_carbon +
            self.end_of_life_carbon
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert impact data to dictionary."""
        return {
            "Manufacturing Carbon": self.manufacturing_carbon,
            "Use Phase Carbon": self.use_phase_carbon,
            "End of Life Carbon": self.end_of_life_carbon,
            "Total Carbon Footprint": self.total_carbon_footprint,
            "Water Consumption": self.water_consumption,
            "Land Use": self.land_use,
            "Energy Consumption": self.energy_consumption,
            "Material Intensity": self.material_intensity,
            "Waste Generation": self.waste_generation,
        }
```

Add a method to calculate detailed impacts:

```python
def perform_detailed_lca(
    self,
    material_inputs: Dict[str, float],
    energy_consumption: float,
    lifetime_years: float,
    annual_energy_generation: float,
    grid_carbon_intensity: float,
    recycling_rates: Dict[str, float],
    transport_distance: float,
) -> DetailedEnvironmentalImpact:
    """
    Perform detailed environmental impact assessment.

    Args:
        material_inputs: Dictionary of material types and quantities
        energy_consumption: Total energy consumed in kWh
        lifetime_years: Lifetime of the system in years
        annual_energy_generation: Annual energy generation in kWh
        grid_carbon_intensity: Grid carbon intensity in kg CO2/kWh
        recycling_rates: Dictionary of material types and their recycling rates
        transport_distance: Transport distance in km

    Returns:
        DetailedEnvironmentalImpact: Detailed environmental impact results
    """
    try:
        # Calculate carbon footprint
        manufacturing_carbon = self.calculate_manufacturing_impact(
            material_inputs, energy_consumption
        )
        use_phase_carbon = self._calculate_use_phase_impact(
            annual_energy_generation, lifetime_years, grid_carbon_intensity
        )
        end_of_life_carbon = self.calculate_end_of_life_impact(
            material_inputs, recycling_rates, transport_distance
        )

        # Calculate additional impacts
        water_consumption = self.calculate_water_consumption(material_inputs)
        land_use = self.calculate_land_use(material_inputs)
        material_intensity = sum(material_inputs.values())

        # Calculate waste generation
        waste_generation = sum(
            quantity * (1.0 - recycling_rates.get(material, 0.0))
            for material, quantity in material_inputs.items()
        )

        return DetailedEnvironmentalImpact(
            manufacturing_carbon=manufacturing_carbon,
            use_phase_carbon=use_phase_carbon,
            end_of_life_carbon=end_of_life_carbon,
            water_consumption=water_consumption,
            land_use=land_use,
            energy_consumption=energy_consumption,
            material_intensity=material_intensity,
            waste_generation=waste_generation,
        )
    except Exception as e:
        self.logger.error(f"Error performing detailed LCA: {str(e)}")
        # Return empty impact object
        return DetailedEnvironmentalImpact(
            manufacturing_carbon=0.0,
            use_phase_carbon=0.0,
            end_of_life_carbon=0.0,
            water_consumption=0.0,
            land_use=0.0,
            energy_consumption=0.0,
            material_intensity=0.0,
            waste_generation=0.0,
        )
```

### 8.3 Adding Real-Time LCA Monitoring

To implement real-time LCA monitoring during manufacturing:

```python
# src/circman5/manufacturing/digital_twin/integration/lca_integration.py

def start_real_time_monitoring(self, interval_seconds: int = 60) -> bool:
    """
    Start real-time LCA monitoring.

    Args:
        interval_seconds: Interval between assessments in seconds

    Returns:
        bool: True if monitoring started successfully
    """
    try:
        # Check if already running
        if hasattr(self, "_monitoring_thread") and self._monitoring_thread is not None:
            if self._monitoring_thread.is_alive():
                self.logger.warning("Real-time LCA monitoring already running")
                return False

        # Set up monitoring flag
        self._monitoring_active = True

        # Start monitoring thread
        import threading
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()

        self.logger.info(f"Started real-time LCA monitoring (interval: {interval_seconds}s)")
        return True
    except Exception as e:
        self.logger.error(f"Error starting real-time monitoring: {str(e)}")
        return False

def stop_real_time_monitoring(self) -> bool:
    """
    Stop real-time LCA monitoring.

    Returns:
        bool: True if monitoring stopped successfully
    """
    try:
        # Check if running
        if not hasattr(self, "_monitoring_active") or not self._monitoring_active:
            self.logger.warning("Real-time LCA monitoring not running")
            return False

        # Set flag to stop
        self._monitoring_active = False

        # Wait for thread to complete
        if hasattr(self, "_monitoring_thread") and self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=5)

        self.logger.info("Stopped real-time LCA monitoring")
        return True
    except Exception as e:
        self.logger.error(f"Error stopping real-time monitoring: {str(e)}")
        return False

def _monitoring_loop(self, interval_seconds: int) -> None:
    """
    Real-time monitoring loop.

    Args:
        interval_seconds: Interval between assessments in seconds
    """
    import time

    while self._monitoring_active:
        try:
            # Get current Digital Twin state
            state = self.digital_twin.get_current_state()

            # Perform LCA analysis
            impact = self.perform_lca_analysis(
                state=state,
                batch_id=f"realtime_{int(time.time())}",
                save_results=False
            )

            # Check for threshold breaches
            self._check_impact_thresholds(impact)

            # Wait for next interval
            time.sleep(interval_seconds)
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {str(e)}")
            # Continue monitoring despite errors
            time.sleep(5)

def _check_impact_thresholds(self, impact: LifeCycleImpact) -> None:
    """
    Check impact values against thresholds.

    Args:
        impact: Impact results to check
    """
    # Define impact thresholds
    thresholds = {
        "manufacturing_impact": 10000.0,
        "total_carbon_footprint": 5000.0,
    }

    # Check thresholds
    threshold_breaches = []

    for impact_type, threshold in thresholds.items():
        current_value = getattr(impact, impact_type)
        if current_value > threshold:
            threshold_breaches.append({
                "impact_type": impact_type,
                "current_value": current_value,
                "threshold": threshold,
                "exceedance": current_value - threshold,
            })

    # Generate alerts for threshold breaches
    if threshold_breaches:
        for breach in threshold_breaches:
            # Log breach
            self.logger.warning(
                f"Impact threshold breach: {breach['impact_type']} = "
                f"{breach['current_value']:.2f} (threshold: {breach['threshold']:.2f})"
            )

            # Publish event if Digital Twin has event system
            if hasattr(self.digital_twin, "event_manager"):
                from circman5.manufacturing.digital_twin.event_notification.event_types import (
                    Event, EventCategory, EventSeverity
                )

                event = Event(
                    category=EventCategory.THRESHOLD,
                    severity=EventSeverity.WARNING,
                    source="lca_monitoring",
                    message=(
                        f"Environmental impact threshold breach: "
                        f"{breach['impact_type']} = {breach['current_value']:.2f} "
                        f"(threshold: {breach['threshold']:.2f})"
                    ),
                    details=breach
                )

                self.digital_twin.event_manager.publish(event)
```

## 9. User Interface Integration

### 9.1 Creating an LCA Dashboard Panel

To create an LCA dashboard panel for the Human Interface:

```python
# src/circman5/manufacturing/human_interface/components/dashboard/lca_panel.py

import datetime
from typing import Dict, Any, Optional
from circman5.utils.logging_config import setup_logger
from circman5.manufacturing.digital_twin.integration.lca_integration import LCAIntegration
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin

class LCAPanel:
    """LCA dashboard panel for the Human Interface."""

    def __init__(self):
        """Initialize the LCA panel."""
        self.logger = setup_logger("lca_panel")

        # Initialize dependencies
        self.digital_twin = DigitalTwin()
        self.lca_integration = LCAIntegration(self.digital_twin)

        # Register with dashboard manager
        from circman5.manufacturing.human_interface.core.dashboard_manager import dashboard_manager
        dashboard_manager.register_component("lca_panel", self)

        self.logger.info("LCA Panel initialized")

    def render_panel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the LCA panel.

        Args:
            config: Panel configuration

        Returns:
            Rendered panel data
        """
        try:
            # Get configuration options
            show_carbon = config.get("show_carbon", True)
            show_water = config.get("show_water", False)
            show_land = config.get("show_land", False)
            show_materials = config.get("show_materials", True)

            # Get current Digital Twin state
            state = self.digital_twin.get_current_state()

            # Check if state is available
            if state is None:
                return {
                    "type": "lca_panel",
                    "title": config.get("title", "Environmental Impact"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": "No digital twin state available",
                }

            # Extract material and energy data
            material_data = self.lca_integration.extract_material_data_from_state(state)
            energy_data = self.lca_integration.extract_energy_data_from_state(state)

            # Get aggregated material inputs
            material_inputs = self.lca_integration._aggregate_material_quantities(material_data)

            # Get total energy consumption
            total_energy = (
                energy_data["energy_consumption"].sum()
                if not energy_data.empty
                else 0.0
            )

            # Prepare panel data
            panel_data = {
                "type": "lca_panel",
                "title": config.get("title", "Environmental Impact"),
                "timestamp": datetime.datetime.now().isoformat(),
                "data": {},
            }

            # Add carbon footprint if requested
            if show_carbon:
                # Perform quick carbon footprint calculation
                analyzer = self.lca_integration.lca_analyzer
                manufacturing_impact = analyzer.calculate_manufacturing_impact(
                    material_inputs, total_energy
                )

                panel_data["data"]["carbon"] = {
                    "manufacturing_impact": manufacturing_impact,
                    "unit": "kg CO2-eq",
                }

            # Add material data if requested
            if show_materials:
                panel_data["data"]["materials"] = {
                    "total_materials": sum(material_inputs.values()),
                    "material_breakdown": material_inputs,
                    "unit": "kg",
                }

            # Add energy data
            panel_data["data"]["energy"] = {
                "total_energy": total_energy,
                "unit": "kWh",
            }

            # Add custom chart config
            panel_data["chart_config"] = {
                "type": config.get("chart_type", "bar"),
                "colors": config.get("colors", ["#4CAF50", "#F44336", "#2196F3"]),
            }

            return panel_data

        except Exception as e:
            self.logger.error(f"Error rendering LCA panel: {str(e)}")
            return {
                "type": "lca_panel",
                "title": config.get("title", "Environmental Impact"),
                "timestamp": datetime.datetime.now().isoformat(),
                "error": f"Error rendering panel: {str(e)}",
            }

    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle panel-specific commands.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Command result
        """
        if command == "run_lca_analysis":
            try:
                # Perform full LCA analysis
                impact = self.lca_integration.perform_lca_analysis(
                    batch_id=params.get("batch_id")
                )

                return {
                    "handled": True,
                    "success": True,
                    "impact": impact.to_dict(),
                }
            except Exception as e:
                self.logger.error(f"Error running LCA analysis: {str(e)}")
                return {
                    "handled": True,
                    "success": False,
                    "error": str(e),
                }

        elif command == "compare_scenarios":
            try:
                # Get scenario parameters
                scenario_params = params.get("scenario_params", {})

                # Create alternative state with scenario parameters
                baseline_state = self.digital_twin.get_current_state()
                alternative_state = self.lca_integration._apply_scenario_adjustments(
                    baseline_state, scenario_params
                )

                # Compare scenarios
                comparison = self.lca_integration.compare_scenarios(
                    baseline_state=baseline_state,
                    alternative_state=alternative_state,
                    scenario_name=params.get("scenario_name", "custom_scenario")
                )

                return {
                    "handled": True,
                    "success": True,
                    "comparison": comparison,
                }
            except Exception as e:
                self.logger.error(f"Error comparing scenarios: {str(e)}")
                return {
                    "handled": True,
                    "success": False,
                    "error": str(e),
                }

        # Not handled by this panel
        return {"handled": False}
```

### 9.2 Adding the Panel to a Dashboard

To add the LCA panel to the main dashboard:

```python
# src/circman5/manufacturing/human_interface/components/dashboard/main_dashboard.py

def _create_dashboard_layout(self):
    """Create dashboard layout for main dashboard."""
    try:
        dashboard_manager.create_layout(
            name="main_dashboard",
            description="Main system dashboard",
            panels={
                # Existing panels...

                # Add LCA panel
                "lca_panel": {
                    "type": "lca_panel",
                    "title": "Environmental Impact",
                    "position": {"row": 2, "col": 0},
                    "size": {"rows": 1, "cols": 1},
                    "show_carbon": True,
                    "show_materials": True,
                    "chart_type": "bar"
                },
            },
            layout_config={"rows": 3, "columns": 2, "spacing": 10}
        )
    except ValueError:
        # Layout already exists
        self.logger.debug("Main dashboard layout already exists")
```

## 10. Troubleshooting Implementation

### 10.1 Missing Impact Factors

If impact factors are missing:

```python
def _check_impact_factors(self):
    """Check if required impact factors are available."""
    required_factors = [
        "MATERIAL_IMPACT_FACTORS",
        "ENERGY_IMPACT_FACTORS",
        "RECYCLING_BENEFIT_FACTORS",
    ]

    missing_factors = []

    for factor in required_factors:
        try:
            self.constants.get_constant("impact_factors", factor)
        except (KeyError, AttributeError):
            missing_factors.append(factor)

    if missing_factors:
        self.logger.warning(f"Missing impact factors: {missing_factors}")
        return False

    return True
```

### 10.2 Common Error Handling

Common error handling patterns:

```python
try:
    # Operation that might fail
    result = self.lca_analyzer.calculate_manufacturing_impact(
        material_inputs, energy_consumption
    )
except ValueError as e:
    # Specific handling for value errors
    self.logger.error(f"Invalid input for LCA calculation: {str(e)}")
    # Provide useful information in returned result
    return {
        "success": False,
        "error": str(e),
        "error_type": "invalid_input",
    }
except KeyError as e:
    # Handling for missing keys
    self.logger.error(f"Missing required data for LCA calculation: {str(e)}")
    return {
        "success": False,
        "error": f"Missing required data: {str(e)}",
        "error_type": "missing_data",
    }
except Exception as e:
    # General exception handling
    self.logger.error(f"Unexpected error in LCA calculation: {str(e)}")
    return {
        "success": False,
        "error": "Unexpected error occurred. See logs for details.",
        "error_type": "unexpected_error",
    }
```

## 11. Conclusion

This implementation guide provides comprehensive instructions for implementing and extending the Lifecycle Assessment system within CIRCMAN5.0. By following these guidelines, you can create a robust LCA system that integrates with the Digital Twin to provide environmental impact assessment capabilities.

The modular architecture allows for easy extension with new impact factors, calculation methods, and visualization capabilities. Integration with the Human Interface enables users to monitor and analyze environmental impacts throughout the manufacturing process.

For additional support, refer to:
- The LCA Architecture document for system design
- The LCA API Reference for detailed method documentation
- The Digital Twin documentation for integration details
- The Human Interface documentation for UI integration
