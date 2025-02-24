# src/circman5/manufacturing/analyzers/sustainability.py

"""Sustainability analysis for PV manufacturing."""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Mapping, Optional
from ...utils.logging_config import setup_logger
from ...utils.results_manager import results_manager
from circman5.manufacturing.visualization_utils import VisualizationConfig
from ...adapters.services.constants_service import ConstantsService


class SustainabilityAnalyzer:
    def __init__(self):
        self.metrics = {}
        self.logger = setup_logger("sustainability_analyzer")
        # Get carbon factors from configuration
        self.constants = ConstantsService()
        self.carbon_factors = {"grid": 0.5, "solar": 0.0, "wind": 0.0}  # Default values
        try:
            self.carbon_factors = self.constants.get_constant(
                "impact_factors", "CARBON_INTENSITY_FACTORS"
            )
        except KeyError:
            self.logger.warning("CARBON_INTENSITY_FACTORS not found, using defaults")

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
        weights = self.constants.get_constant(
            "impact_factors", "SUSTAINABILITY_WEIGHTS"
        )
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

    def plot_sustainability_metrics(
        self,
        material_data: pd.DataFrame,
        energy_data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot sustainability-related metrics.

        Args:
            material_data: DataFrame with material metrics
            energy_data: DataFrame with energy metrics
            save_path: Optional path to save the plot
        """
        try:
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
                # Save to default location using results_manager
                temp_path = "sustainability_metrics.png"
                plt.savefig(temp_path, dpi=300, bbox_inches="tight")
                plt.close()
                results_manager.save_file(temp_path, "visualizations")
                Path(temp_path).unlink()  # Clean up temporary file

        except Exception as e:
            self.logger.error(f"Error plotting sustainability metrics: {str(e)}")
            raise

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

    def plot_metrics(
        self,
        material_data: pd.DataFrame,
        energy_data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot sustainability metrics."""
        try:
            # Apply visualization style
            VisualizationConfig.setup_style()

            fig, axes = plt.subplots(
                2, 2, figsize=VisualizationConfig.DEFAULT_STYLE["figure.figsize"]
            )

            # Material Usage vs Waste Over Time
            axes[0, 0].plot(
                material_data["timestamp"],
                material_data["quantity_used"],
                label="Used",
                color=VisualizationConfig.COLOR_PALETTE[0],
                marker="o",
            )
            axes[0, 0].plot(
                material_data["timestamp"],
                material_data["waste_generated"],
                label="Waste",
                color=VisualizationConfig.COLOR_PALETTE[1],
                marker="o",
            )
            axes[0, 0].set_title("Material Usage vs Waste")
            axes[0, 0].set_xlabel("Time")
            axes[0, 0].set_ylabel("Amount")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Energy Source Distribution
            energy_by_source = energy_data.groupby("energy_source")[
                "energy_consumption"
            ].sum()
            wedges, texts, autotexts = axes[0, 1].pie(
                energy_by_source,
                labels=energy_by_source.index,
                colors=VisualizationConfig.COLOR_PALETTE[: len(energy_by_source)],
                autopct="%1.1f%%",
                explode=[0.05]
                * len(energy_by_source),  # Slight separation for visibility
            )
            axes[0, 1].set_title("Energy Source Distribution")

            # Material Efficiency Metrics
            metrics = self.analyze_material_efficiency(material_data)
            if metrics:
                metrics_to_plot = {
                    "Material\nEfficiency": metrics["material_efficiency"],
                    "Recycling\nRate": metrics["recycling_rate"],
                    "Waste\nReduction": metrics["waste_reduction"],
                }
                bars = axes[1, 0].bar(
                    metrics_to_plot.keys(),
                    metrics_to_plot.values(),
                    color=VisualizationConfig.COLOR_PALETTE[:3],
                )
                axes[1, 0].set_title("Material Efficiency Metrics")
                axes[1, 0].set_ylabel("Percentage (%)")
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 0].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                    )
                axes[1, 0].grid(True, axis="y")
                axes[1, 0].set_ylim(0, 100)  # Set y-axis from 0 to 100%

            # Energy Efficiency Trend
            if "efficiency_rate" in energy_data.columns:
                energy_data.plot(
                    x="timestamp",
                    y="efficiency_rate",
                    ax=axes[1, 1],
                    color=VisualizationConfig.COLOR_PALETTE[3],
                    marker="o",
                )
                axes[1, 1].set_title("Energy Efficiency Trend")
                axes[1, 1].set_xlabel("Time")
                axes[1, 1].set_ylabel("Efficiency Rate")
                axes[1, 1].grid(True)

            plt.tight_layout()

            if save_path:
                VisualizationConfig.save_figure(fig, Path(save_path).name)
            else:
                temp_path = Path("sustainability_metrics.png")
                VisualizationConfig.save_figure(fig, temp_path.name)

            plt.close()
            self.logger.info("Sustainability metrics plot generated successfully")

        except Exception as e:
            self.logger.error(f"Error plotting sustainability metrics: {str(e)}")
            plt.close()
            raise

    def analyze_trends(
        self, material_data: pd.DataFrame, energy_data: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Analyze sustainability trends."""
        daily_metrics = material_data.groupby(
            pd.Grouper(key="timestamp", freq="D")
        ).agg(
            {"quantity_used": "sum", "waste_generated": "sum", "recycled_amount": "sum"}
        )

        return {
            "material_efficiency": (
                (daily_metrics["quantity_used"] - daily_metrics["waste_generated"])
                / daily_metrics["quantity_used"]
                * 100
            ).tolist(),
            "recycling_rate": (
                daily_metrics["recycled_amount"]
                / daily_metrics["waste_generated"]
                * 100
            ).tolist(),
        }
