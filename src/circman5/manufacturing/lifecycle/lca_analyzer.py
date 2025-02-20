# src/circman5/manufacturing/lifecycle/lca_analyzer.py

"""Life Cycle Assessment analyzer for PV manufacturing."""
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
from pathlib import Path
from circman5.utils.results_manager import results_manager
from ...utils.logging_config import setup_logger
from .impact_factors import (
    MATERIAL_IMPACT_FACTORS,
    ENERGY_IMPACT_FACTORS,
    RECYCLING_BENEFIT_FACTORS,
)


@dataclass
class LifeCycleImpact:
    """Data class to hold lifecycle impact assessment results."""

    manufacturing_impact: float
    use_phase_impact: float
    end_of_life_impact: float

    def __init__(
        self,
        manufacturing_impact: float,
        use_phase_impact: float,
        end_of_life_impact: float,
        total_carbon_footprint: Optional[
            float
        ] = None,  # Added for backward compatibility
    ):
        self.manufacturing_impact = manufacturing_impact
        self.use_phase_impact = use_phase_impact
        self.end_of_life_impact = end_of_life_impact
        # Ignore passed total_carbon_footprint as it's calculated

    @property
    def total_carbon_footprint(self) -> float:
        """Calculate total carbon footprint across all phases."""
        return (
            self.manufacturing_impact + self.use_phase_impact + self.end_of_life_impact
        )

    @property
    def total_impact(self) -> float:
        """Alias for total_carbon_footprint to maintain compatibility."""
        return self.total_carbon_footprint

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
        self.logger = setup_logger("lca_analyzer")

        # Initialize with ResultsManager paths
        self.results_dir = results_manager.get_run_dir()
        self.lca_results_dir = results_manager.get_path("lca_results")

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
        for material, quantity in material_inputs.items():
            if quantity < 0:
                raise ValueError(f"Material quantity cannot be negative: {material}")
            impact_factor = MATERIAL_IMPACT_FACTORS.get(material, 0.0)
            impact += quantity * impact_factor

        # Energy impact
        impact += energy_consumption * ENERGY_IMPACT_FACTORS.get(
            "grid_electricity", 0.5
        )

        return impact

    def _calculate_use_phase_impact(
        self, annual_generation: float, lifetime: float, grid_intensity: float
    ) -> float:
        """Calculate use phase impact with meaningful values."""
        # Validate physical constraints
        if annual_generation < 0:
            raise ValueError("Annual energy generation cannot be negative")
        if lifetime <= 0:
            raise ValueError("System lifetime must be positive")
        if grid_intensity < 0:
            raise ValueError("Grid carbon intensity cannot be negative")

        # Calculate avoided emissions (must be meaningfully negative)
        impact = -1.0 * annual_generation * lifetime * grid_intensity
        if impact >= 0:
            self.logger.warning(
                "Use phase impact calculation yielded non-negative value"
            )
        return impact

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
        for material, quantity in material_inputs.items():
            recycling_rate = recycling_rates.get(material, 0.0)
            benefit_factor = RECYCLING_BENEFIT_FACTORS.get(material, 0.0)
            impact += quantity * recycling_rate * benefit_factor

        # Transport impact
        total_mass = sum(material_inputs.values())
        transport_factor = 0.062  # kg CO2-eq per tonne-km
        impact += (total_mass / 1000) * transport_distance * transport_factor

        return impact

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

    def _aggregate_material_inputs(
        self, material_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Aggregate material quantities by type.

        Args:
            material_data: DataFrame containing material flow data

        Returns:
            Dict mapping material types to total quantities
        """
        if material_data.empty:
            return {}

        try:
            material_totals = material_data.groupby("material_type")[
                "quantity_used"
            ].sum()
            return material_totals.to_dict()
        except Exception as e:
            self.logger.error(f"Error aggregating material inputs: {str(e)}")
            return {}

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

    def calculate_energy_generation(self, material_inputs: Dict[str, float]) -> float:
        """
        Calculate expected annual energy generation.

        Args:
            material_inputs: Dictionary of material quantities

        Returns:
            float: Expected annual energy generation in kWh
        """
        try:
            # Estimate panel area from glass weight
            glass_weight = material_inputs.get("solar_glass", 0.0)
            panel_area = glass_weight / 10.0  # Approximate conversion

            # Apply standard efficiency factors
            solar_irradiance = 1000.0  # kWh/m²/year
            panel_efficiency = 0.20  # 20% efficiency

            return panel_area * solar_irradiance * panel_efficiency
        except Exception as e:
            self.logger.error(f"Error calculating energy generation: {str(e)}")
            return 0.0

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
