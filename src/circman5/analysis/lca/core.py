from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class LifeCycleImpact:
    """Represents the environmental impact across different lifecycle phases"""

    manufacturing_impact: float
    use_phase_impact: float
    end_of_life_impact: float
    total_carbon_footprint: float

    @property
    def total_impact(self) -> float:
        """Calculate total lifecycle impact"""
        return (
            self.manufacturing_impact + self.use_phase_impact + self.end_of_life_impact
        )


class LCAAnalyzer:
    """Core class for Life Cycle Assessment analysis"""

    def __init__(self):
        # Impact factors for different processes (could be loaded from configuration)
        self.impact_factors = {
            "silicon_production": 32.8,  # kg CO2-eq per kg
            "glass_production": 0.9,  # kg CO2-eq per kg
            "energy_consumption": 0.5,  # kg CO2-eq per kWh
            "transport": 0.2,  # kg CO2-eq per km
            "recycling_benefit": -0.3,  # kg CO2-eq savings per kg recycled
        }

    def calculate_manufacturing_impact(
        self, material_inputs: Dict[str, float], energy_consumption: float
    ) -> float:
        """
        Calculate environmental impact of manufacturing phase

        Args:
            material_inputs: Dictionary of material names and quantities (kg)
            energy_consumption: Energy used in manufacturing (kWh)
        """
        impact = 0.0

        # Calculate material production impact
        for material, quantity in material_inputs.items():
            if material in self.impact_factors:
                impact += quantity * self.impact_factors[material]

        # Add energy consumption impact
        impact += energy_consumption * self.impact_factors["energy_consumption"]

        return impact

    def calculate_use_phase_impact(
        self,
        lifetime_years: float,
        annual_energy_generation: float,
        grid_carbon_intensity: float,
    ) -> float:
        """
        Calculate use phase impact, including benefits from clean energy generation

        Args:
            lifetime_years: Expected operational lifetime
            annual_energy_generation: Annual energy output (kWh)
            grid_carbon_intensity: Carbon intensity of displaced grid electricity
        """
        # Calculate total energy generation over lifetime
        total_generation = lifetime_years * annual_energy_generation

        # Calculate emissions avoided by replacing grid electricity
        emissions_avoided = total_generation * grid_carbon_intensity

        # Return negative value as this represents environmental benefit
        return -emissions_avoided

    def calculate_end_of_life_impact(
        self,
        material_quantities: Dict[str, float],
        recycling_rates: Dict[str, float],
        transport_distance: float,
    ) -> float:
        """
        Calculate end-of-life phase impact including recycling benefits

        Args:
            material_quantities: Dictionary of material amounts (kg)
            recycling_rates: Dictionary of recycling rates for each material
            transport_distance: Distance to recycling facility (km)
        """
        impact = 0.0

        # Calculate recycling benefits
        for material, quantity in material_quantities.items():
            if material in recycling_rates:
                recycled_amount = quantity * recycling_rates[material]
                impact += recycled_amount * self.impact_factors["recycling_benefit"]

        # Add transport impact
        impact += transport_distance * self.impact_factors["transport"]

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
        """
        Perform complete lifecycle assessment

        Returns:
            LifeCycleImpact object containing impacts for each phase
        """
        manufacturing_impact = self.calculate_manufacturing_impact(
            material_inputs, energy_consumption
        )

        use_phase_impact = self.calculate_use_phase_impact(
            lifetime_years, annual_energy_generation, grid_carbon_intensity
        )

        end_of_life_impact = self.calculate_end_of_life_impact(
            material_inputs, recycling_rates, transport_distance
        )

        total_carbon_footprint = (
            manufacturing_impact + use_phase_impact + end_of_life_impact
        )

        return LifeCycleImpact(
            manufacturing_impact=manufacturing_impact,
            use_phase_impact=use_phase_impact,
            end_of_life_impact=end_of_life_impact,
            total_carbon_footprint=total_carbon_footprint,
        )
