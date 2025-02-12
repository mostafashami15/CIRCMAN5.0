# src/circman5/manufacturing/analyzers/sustainability.py
"""
Sustainability analysis for PV manufacturing.
"""
import pandas as pd
import numpy as np
from typing import Mapping, Optional, Dict


class SustainabilityAnalyzer:
    def __init__(self):
        self.metrics = {}
        # Carbon intensity factors (kg CO2/kWh)
        self.carbon_factors = {"grid": 0.5, "solar": 0.0, "wind": 0.0}

    def calculate_carbon_footprint(self, energy_data: pd.DataFrame) -> float:
        """
        Calculate carbon footprint based on energy sources.

        Args:
            energy_data: DataFrame containing energy consumption data

        Returns:
            Total carbon footprint in kg CO2
        """
        if energy_data.empty:
            return 0.0

        carbon_footprint = 0.0
        for source in energy_data["energy_source"].unique():
            source_consumption = energy_data[energy_data["energy_source"] == source][
                "energy_consumption"
            ].sum()
            carbon_footprint += source_consumption * self.carbon_factors.get(
                source, 0.5
            )
        return carbon_footprint

    def analyze_material_efficiency(
        self, material_data: pd.DataFrame
    ) -> Mapping[str, float]:
        """
        Analyze material utilization and recycling efficiency.

        Args:
            material_data: DataFrame containing material flow data

        Returns:
            Dictionary of material efficiency metrics
        """
        if material_data.empty:
            return {}

        total_used = material_data["quantity_used"].sum()
        total_waste = material_data["waste_generated"].sum()
        total_recycled = material_data["recycled_amount"].sum()

        metrics = {
            "material_efficiency": (
                (total_used - total_waste) / total_used * 100 if total_used > 0 else 0
            ),
            "recycling_rate": (
                total_recycled / total_waste * 100 if total_waste > 0 else 0
            ),
            "waste_reduction": (
                (1 - total_waste / total_used) * 100 if total_used > 0 else 0
            ),
        }
        self.metrics.update(metrics)
        return metrics

    def calculate_sustainability_score(
        self,
        material_efficiency: float,
        recycling_rate: float,
        energy_efficiency: float,
    ) -> float:
        """
        Calculate overall sustainability score.

        Args:
            material_efficiency: Efficiency of material usage
            recycling_rate: Percentage of materials recycled
            energy_efficiency: Energy usage efficiency

        Returns:
            Weighted sustainability score
        """
        weights = {"material": 0.4, "recycling": 0.3, "energy": 0.3}
        score = (
            material_efficiency * weights["material"]
            + recycling_rate * weights["recycling"]
            + energy_efficiency * weights["energy"]
        )
        return score

    def identify_sustainability_trends(
        self, material_data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Identify sustainability trends over time.

        Args:
            material_data: DataFrame containing material flow data

        Returns:
            Dictionary of sustainability trend series
        """
        if material_data.empty:
            return {}

        # Calculate weekly metrics
        weekly_metrics = material_data.groupby(
            pd.Grouper(key="timestamp", freq="W")
        ).agg(
            {"quantity_used": "sum", "waste_generated": "sum", "recycled_amount": "sum"}
        )

        # Calculate efficiency metrics
        weekly_metrics["material_efficiency"] = (
            (weekly_metrics["quantity_used"] - weekly_metrics["waste_generated"])
            / weekly_metrics["quantity_used"]
            * 100
        )
        weekly_metrics["recycling_rate"] = (
            weekly_metrics["recycled_amount"] / weekly_metrics["waste_generated"] * 100
        )

        return {
            "material_efficiency_trend": weekly_metrics["material_efficiency"],
            "recycling_rate_trend": weekly_metrics["recycling_rate"],
        }
