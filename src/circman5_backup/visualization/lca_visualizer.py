"""
LCA visualization methods for analyzing environmental impacts and material flows.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List, Sequence
from pathlib import Path


class LCAVisualizer:
    """Visualizes Life Cycle Assessment results."""

    def __init__(self):
        """Initialize visualization settings."""
        # Use default matplotlib style
        plt.style.use("default")
        # Configure seaborn without style
        sns.set_theme(style="whitegrid")
        self.colors = sns.color_palette("husl", 8)

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

    def plot_lifecycle_comparison(
        self,
        manufacturing_impact: float,
        use_phase_impact: float,
        end_of_life_impact: float,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create bar chart comparing impacts across lifecycle phases.

        Args:
            manufacturing_impact: Impact from manufacturing phase
            use_phase_impact: Impact from use phase
            end_of_life_impact: Impact from end-of-life phase
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=(10, 6))

        phases = ["Manufacturing", "Use Phase", "End of Life"]
        impacts = [manufacturing_impact, use_phase_impact, end_of_life_impact]

        # Create bar chart
        bars = plt.bar(phases, impacts, color=self.colors[:3])

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        plt.title("Environmental Impact Across Lifecycle Phases")
        plt.ylabel("Impact (kg CO2-eq)")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_material_flow(
        self, material_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Create a material flow visualization showing inputs, waste, and recycling.

        Args:
            material_data: DataFrame containing material flow information
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=(12, 6))

        # Calculate totals for each material
        material_totals = material_data.groupby("material_type").agg(
            {"quantity_used": "sum", "waste_generated": "sum", "recycled_amount": "sum"}
        )

        # Create stacked bar chart
        material_totals.plot(kind="bar", stacked=True, color=self.colors[:3])

        plt.title("Material Flow Analysis")
        plt.xlabel("Material Type")
        plt.ylabel("Amount (kg)")
        plt.legend(title="Flow Type")
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_energy_consumption_trends(
        self, energy_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Create line plot showing energy consumption trends over time.

        Args:
            energy_data: DataFrame containing energy consumption data
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=(12, 6))

        # Calculate daily energy consumption by source
        daily_consumption = (
            energy_data.groupby(
                [pd.Grouper(key="timestamp", freq="D"), "energy_source"]
            )["energy_consumption"]
            .sum()
            .unstack()
        )

        # Create line plot
        daily_consumption.plot(marker="o")

        plt.title("Energy Consumption Trends by Source")
        plt.xlabel("Date")
        plt.ylabel("Energy Consumption (kWh)")
        plt.legend(title="Energy Source")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def create_comprehensive_report(
        self,
        impact_data: Dict[str, float],
        material_data: pd.DataFrame,
        energy_data: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """
        Generate a comprehensive set of LCA visualizations.

        Args:
            impact_data: Dictionary of impact categories and their values
            material_data: DataFrame containing material flow information
            energy_data: DataFrame containing energy consumption data
            output_dir: Directory to save visualization files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate all visualizations
        self.plot_impact_distribution(
            impact_data, save_path=f"{output_dir}/impact_distribution.png"
        )

        manufacturing = impact_data.get("Manufacturing Impact", 0)
        use_phase = impact_data.get("Use Phase Impact", 0)
        end_of_life = impact_data.get("End of Life Impact", 0)

        self.plot_lifecycle_comparison(
            manufacturing,
            use_phase,
            end_of_life,
            save_path=f"{output_dir}/lifecycle_comparison.png",
        )

        self.plot_material_flow(
            material_data, save_path=f"{output_dir}/material_flow.png"
        )

        self.plot_energy_consumption_trends(
            energy_data, save_path=f"{output_dir}/energy_trends.png"
        )
