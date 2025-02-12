# src/circman5/manufacturing/analyzers/sustainability.py

"""Sustainability analysis for PV manufacturing."""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Mapping, Optional
from ...utils.logging_config import setup_logger


class SustainabilityAnalyzer:
    def __init__(self):
        self.metrics = {}
        self.logger = setup_logger("sustainability_analyzer")
        # Carbon intensity factors (kg CO2/kWh)
        self.carbon_factors = {"grid": 0.5, "solar": 0.0, "wind": 0.0}

    def calculate_carbon_footprint(self, energy_data: pd.DataFrame) -> float:
        """Calculate carbon footprint based on energy sources."""
        if energy_data.empty:
            self.logger.warning("Empty energy data provided")
            return 0.0

        carbon_footprint = 0.0
        for source in energy_data["energy_source"].unique():
            source_consumption = energy_data[energy_data["energy_source"] == source][
                "energy_consumption"
            ].sum()
            carbon_footprint += source_consumption * self.carbon_factors.get(
                source, 0.5
            )

        self.logger.info(f"Carbon footprint calculated: {carbon_footprint}")
        return carbon_footprint

    def analyze_material_efficiency(
        self, material_data: pd.DataFrame
    ) -> Mapping[str, float]:
        """Analyze material utilization and recycling efficiency."""
        if material_data.empty:
            self.logger.warning("Empty material data provided")
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
        self.logger.info(f"Material efficiency metrics calculated: {metrics}")
        return metrics

    def calculate_sustainability_score(
        self,
        material_efficiency: float,
        recycling_rate: float,
        energy_efficiency: float,
    ) -> float:
        """Calculate overall sustainability score."""
        weights = {"material": 0.4, "recycling": 0.3, "energy": 0.3}
        score = (
            material_efficiency * weights["material"]
            + recycling_rate * weights["recycling"]
            + energy_efficiency * weights["energy"]
        )

        self.logger.info(f"Sustainability score calculated: {score}")
        return score

    def calculate_waste_metrics(self, material_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive waste-related metrics."""
        if material_data.empty:
            return {}

        try:
            material_metrics = self.analyze_material_efficiency(material_data)

            waste_metrics = {
                "total_waste": material_data["waste_generated"].sum(),
                "recycling_rate": material_metrics.get("recycling_rate", 0.0),
                "waste_by_type": material_data.groupby("material_type")[
                    "waste_generated"
                ]
                .sum()
                .to_dict(),
                "recovery_efficiency": material_metrics.get("material_efficiency", 0.0),
            }

            self.logger.info(f"Waste metrics calculated successfully: {waste_metrics}")
            return waste_metrics

        except Exception as e:
            self.logger.error(f"Error calculating waste metrics: {str(e)}")
            return {}

    def calculate_resource_efficiency(self, material_data: pd.DataFrame) -> float:
        """Calculate resource utilization efficiency."""
        metrics = self.analyze_material_efficiency(material_data)
        return metrics.get("material_efficiency", 0.0)

    def calculate_energy_efficiency(self, energy_data: pd.DataFrame) -> float:
        """Calculate comprehensive energy efficiency score."""
        if energy_data.empty:
            return 0.0

        try:
            base_efficiency = (
                energy_data["efficiency_rate"].mean() * 100
                if "efficiency_rate" in energy_data.columns
                else 0.0
            )

            carbon_footprint = self.calculate_carbon_footprint(energy_data)

            carbon_impact_factor = 1.0
            if carbon_footprint > 0:
                carbon_impact_factor = max(0.5, 1 - (carbon_footprint / 10000))

            adjusted_efficiency = base_efficiency * carbon_impact_factor

            self.logger.info(
                f"Energy efficiency calculated: {adjusted_efficiency:.2f}% "
                f"(Base: {base_efficiency:.2f}%, Carbon Impact: {carbon_impact_factor:.2f})"
            )

            return adjusted_efficiency

        except Exception as e:
            self.logger.error(f"Error calculating energy efficiency: {str(e)}")
            return 0.0

    def calculate_sustainability_metrics(
        self, energy_data: pd.DataFrame, material_flow: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive sustainability metrics.

        Args:
            energy_data: DataFrame containing energy consumption data
            material_flow: DataFrame containing material flow data

        Returns:
            Dict containing all sustainability metrics
        """
        try:
            # Calculate carbon footprint
            carbon_footprint = self.calculate_carbon_footprint(energy_data)

            # Analyze material efficiency
            material_metrics = self.analyze_material_efficiency(material_flow)

            # Calculate overall sustainability score
            sustainability_score = self.calculate_sustainability_score(
                material_metrics.get("material_efficiency", 0),
                material_metrics.get("recycling_rate", 0),
                self.calculate_energy_efficiency(energy_data),
            )

            metrics = {
                "carbon_footprint": {
                    "total": carbon_footprint,
                    "per_unit": carbon_footprint / len(energy_data)
                    if not energy_data.empty
                    else 0.0,
                },
                "material_efficiency": {
                    metric: float(value) for metric, value in material_metrics.items()
                },
                "sustainability_score": {"overall": float(sustainability_score)},
            }

            self.logger.info(f"Sustainability metrics calculated: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating sustainability metrics: {str(e)}")
            return {
                "carbon_footprint": {"total": 0.0, "per_unit": 0.0},
                "material_efficiency": {"efficiency": 0.0},
                "sustainability_score": {"overall": 0.0},
            }

    def plot_sustainability_metrics(self, material_data, energy_data, save_path=None):
        """
        Plot sustainability-related metrics.

        Args:
            material_data: DataFrame with material metrics
            energy_data: DataFrame with energy metrics
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Example: Material efficiency plot
        if "material_efficiency" in material_data.columns:
            axes[0].plot(
                material_data["timestamp"], material_data["material_efficiency"]
            )
            axes[0].set_title("Material Efficiency Over Time")
            axes[0].set_ylabel("Efficiency (%)")
            axes[0].grid(True)

            # Fix axis limits
            ymin, ymax = axes[0].get_ylim()
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            axes[0].set_ylim(ymin, ymax)

        # Example: Carbon footprint plot
        if "carbon_footprint" in energy_data.columns:
            axes[1].bar(energy_data["timestamp"], energy_data["carbon_footprint"])
            axes[1].set_title("Carbon Footprint Over Time")
            axes[1].set_ylabel("CO2 Emissions")
            axes[1].grid(True)

            # Fix axis limits
            ymin, ymax = axes[1].get_ylim()
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            axes[1].set_ylim(ymin, ymax)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def test_sustainability_visualization(analyzer, sample_material_data, sample_energy_data, visualizations_dir):  # type: ignore
        """Test sustainability visualization functionality."""
        # Generate visualization path
        viz_path = visualizations_dir / "sustainability_metrics.png"

        # Test visualization generation
        analyzer.plot_sustainability_metrics(
            sample_material_data, sample_energy_data, str(viz_path)
        )

        # Verify file was created
        assert viz_path.exists()

        # Test handling of empty data
        empty_df = pd.DataFrame()
        analyzer.plot_sustainability_metrics(empty_df, empty_df, str(viz_path))
