# src/circman5/manufacturing/lifecycle/lca_analyzer.py

"""Life Cycle Assessment analyzer for PV manufacturing."""
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
from pathlib import Path
from ...utils.logging_config import setup_logger
from ...config.project_paths import project_paths
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

    def calculate_manufacturing_impact(
        self, material_inputs: Dict[str, float], energy_consumption: float
    ) -> float:
        """Calculate manufacturing phase impact."""
        return self._calculate_manufacturing_impact(material_inputs, energy_consumption)

    def calculate_use_phase_impact(
        self,
        annual_energy_generation: float,  # Changed parameter name to match internal method
        lifetime_years: float,
        grid_carbon_intensity: float,
    ) -> float:
        """Calculate use phase impact (usually negative due to clean energy generation)."""
        return self._calculate_use_phase_impact(
            annual_generation=annual_energy_generation,
            lifetime=lifetime_years,
            grid_intensity=grid_carbon_intensity,
        )

    def calculate_end_of_life_impact(
        self,
        material_inputs: Dict[str, float],
        recycling_rates: Dict[str, float],
        transport_distance: float,
    ) -> float:
        """Calculate end of life impact including recycling benefits."""
        return self._calculate_end_of_life_impact(
            material_inputs, recycling_rates, transport_distance
        )

    def _aggregate_material_inputs(
        self, material_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Aggregate material quantities by type."""
        if material_data.empty:
            return {}
        return material_data.groupby("material_type")["quantity_used"].sum().to_dict()

    def calculate_recycling_rates(
        self, material_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate recycling rates from historical data."""
        if material_data.empty:
            return {}

        recycling_rates = {}
        try:
            data_copy = material_data.copy()
            data_copy["waste_generated"] = data_copy["waste_generated"].astype(float)
            data_copy["recycled_amount"] = data_copy["recycled_amount"].astype(float)

            material_totals = (
                data_copy.groupby("material_type")
                .agg({"waste_generated": "sum", "recycled_amount": "sum"})
                .astype(float)
            )

            for material in material_totals.index:
                waste = float(material_totals.at[material, "waste_generated"])
                recycled = float(material_totals.at[material, "recycled_amount"])

                if waste > 0:
                    rate = recycled / waste
                    rate = max(0.0, min(1.0, rate))
                else:
                    rate = 0.0

                recycling_rates[material] = rate

            return recycling_rates
        except Exception as e:
            self.logger.error(f"Error calculating recycling rates: {str(e)}")
            return {}

    def calculate_energy_generation(self, material_inputs: Dict[str, float]) -> float:
        """Calculate expected annual energy generation."""
        try:
            glass_weight = float(material_inputs.get("solar_glass", 0))
            total_panel_area = glass_weight / 10.0  # Approximate area from glass weight

            average_efficiency = 0.20  # 20% efficiency
            solar_irradiance = 1000.0  # kWh/m²/year (typical value)

            return total_panel_area * average_efficiency * solar_irradiance
        except Exception as e:
            self.logger.error(f"Error calculating energy generation: {str(e)}")
            return 0.0

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
            manufacturing_impact = self._calculate_manufacturing_impact(
                material_inputs, energy_consumption
            )

            # Use phase impact
            use_phase_impact = self._calculate_use_phase_impact(
                annual_energy_generation, lifetime_years, grid_carbon_intensity
            )

            # End of life impact
            end_of_life_impact = self._calculate_end_of_life_impact(
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

    def save_results(
        self, impact: LifeCycleImpact, batch_id: Optional[str] = None
    ) -> None:
        """Save LCA results to Excel file."""
        try:
            # Get the run directory and ensure it exists
            run_dir = project_paths.get_run_directory()
            if not run_dir.exists():
                run_dir.mkdir(parents=True, exist_ok=True)

            # Create the reports directory
            reports_dir = run_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Construct the filename
            filename = f"lca_impact_{batch_id}.xlsx" if batch_id else "lca_impact.xlsx"
            file_path = reports_dir / filename

            # Convert impact data to DataFrame
            df = pd.DataFrame([impact.to_dict()])

            # Save to Excel
            df.to_excel(file_path, index=False)

            # Log success
            self.logger.info(
                f"LCA Results saved successfully for batch {batch_id or 'default'}:\n"
                f"File Path: {file_path}\n"
                f"Manufacturing Impact: {impact.manufacturing_impact}\n"
                f"Use Phase Impact: {impact.use_phase_impact}\n"
                f"End of Life Impact: {impact.end_of_life_impact}"
            )

        except Exception as e:
            # Log detailed error message
            self.logger.error(
                f"Error saving LCA results for batch {batch_id or 'default'}: {str(e)}"
            )
            raise

    def _calculate_manufacturing_impact(
        self, material_inputs: Dict[str, float], energy_consumption: float
    ) -> float:
        """Calculate manufacturing phase impact."""
        impact = 0.0

        # Material impacts
        for material, quantity in material_inputs.items():
            impact_factor = MATERIAL_IMPACT_FACTORS.get(material, 0.0)
            impact += quantity * impact_factor

        # Energy impact
        impact += energy_consumption * ENERGY_IMPACT_FACTORS.get("grid", 0.5)

        return impact

    def _calculate_use_phase_impact(
        self, annual_generation: float, lifetime: float, grid_intensity: float
    ) -> float:
        """Calculate use phase impact (usually negative due to clean energy generation)."""
        return -annual_generation * lifetime * grid_intensity

    def _calculate_end_of_life_impact(
        self,
        material_inputs: Dict[str, float],
        recycling_rates: Dict[str, float],
        transport_distance: float,
    ) -> float:
        """Calculate end of life impact including recycling benefits."""
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
