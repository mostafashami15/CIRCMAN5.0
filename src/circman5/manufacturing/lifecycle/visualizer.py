# src/circman5/manufacturing/lifecycle/visualizer.py

"""
LCA visualization methods for analyzing environmental impacts and material flows.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List, Sequence, Union
from pathlib import Path
from circman5.utils.logging_config import setup_logger
from circman5.utils.results_manager import results_manager


class LCAVisualizer:
    """Visualizes Life Cycle Assessment results."""

    def __init__(self):
        """Initialize visualization settings."""
        self.logger = setup_logger("lca_visualizer")
        # Replace run_dir and viz_dir initialization
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

    def _handle_report_plotting(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        fig = None
        """
        Handle time series plotting for reports with proper axis handling.

        Args:
            df: DataFrame to plot
            save_path: Optional path to save plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            if df.empty:
                self.logger.warning(f"No data to plot for {title}")
                plt.close(fig)
                return

            # Plot data
            df.plot(ax=ax)

            # Handle x-axis limits
            x_min = df.index.min()
            x_max = df.index.max()

            if pd.isna(x_min) or pd.isna(x_max) or x_min == x_max:
                # For single point or invalid data, use date range
                reference_date = pd.Timestamp.now()
                # Convert timestamps to matplotlib numeric dates
                x_min_plt = float(
                    date2num(
                        pd.Timestamp(
                            reference_date - pd.Timedelta(days=1)
                        ).to_pydatetime()
                    )
                )
                x_max_plt = float(
                    date2num(
                        pd.Timestamp(
                            reference_date + pd.Timedelta(days=1)
                        ).to_pydatetime()
                    )
                )
                ax.set_xlim(x_min_plt, x_max_plt)
            else:
                # Add padding to the time range
                delta = pd.Timedelta((x_max - x_min) * 0.1)
                x_min_plt = float(date2num(pd.Timestamp(x_min - delta).to_pydatetime()))
                x_max_plt = float(date2num(pd.Timestamp(x_max + delta).to_pydatetime()))
                if abs(x_max_plt - x_min_plt) < 1e-6:
                    x_min_plt -= 0.5
                    x_max_plt += 0.5

            ax.set_xlim(x_min_plt, x_max_plt)

            # Set labels and title
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Format x-axis with dates
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)

            # Adjust layout
            plt.tight_layout()

            # Save or show plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.close(fig)

        except Exception as e:
            plt.close(fig)
            self.logger.error(f"Error in report plotting: {str(e)}")
            raise

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
        # Check if the provided material_data is empty
        if material_data.empty:
            self.logger.warning(
                "Material flow data is empty. Skipping material flow plot."
            )
            return

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
        """Create line plot showing energy consumption trends over time."""
        try:
            plt.figure(figsize=(12, 6))

            if energy_data.empty:
                self.logger.warning("No energy consumption data to plot")
                plt.close()
                return

            # Calculate daily energy consumption by source
            daily_consumption = (
                energy_data.groupby(
                    [pd.Grouper(key="timestamp", freq="D"), "energy_source"]
                )["energy_consumption"]
                .sum()
                .unstack(fill_value=0)  # Fill NaN values with 0
            )

            if daily_consumption.empty:
                self.logger.warning("No daily consumption data after grouping")
                plt.close()
                return

            # Create line plot
            ax = daily_consumption.plot(marker="o")

            # Handle x-axis limits
            x_min = daily_consumption.index.min()
            x_max = daily_consumption.index.max()

            if x_min == x_max:
                # For single day, add padding
                padding = pd.Timedelta(days=1)
                ax.set_xlim(x_min - padding, x_max + padding)
            else:
                # For multiple days, add small padding
                padding = (x_max - x_min) * 0.05
                ax.set_xlim(x_min - padding, x_max + padding)

            plt.title("Energy Consumption Trends by Source")
            plt.xlabel("Date")
            plt.ylabel("Energy Consumption (kWh)")
            plt.legend(
                title="Energy Source", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
            plt.grid(True)

            # Ensure proper layout with legend
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

        except Exception as e:
            plt.close()
            self.logger.error(f"Error plotting energy consumption trends: {str(e)}")
            raise

    def _get_visualization_path(self, filename: str) -> str:
        """Get the proper path for saving visualizations."""
        return str(self.viz_dir / filename)

    def _ensure_save_path(self, filename: str, batch_id: Optional[str] = None) -> str:
        """
        Ensure visualization is saved in the correct directory with proper naming.

        Args:
            filename: Base filename for the visualization
            batch_id: Optional batch identifier for batch-specific visualizations

        Returns:
            str: Full path to save the visualization
        """
        # Add batch_id to filename if provided
        if batch_id:
            base_name, ext = os.path.splitext(filename)
            filename = f"{base_name}_{batch_id}{ext}"

        return str(self.viz_dir / filename)

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

            # Impact distribution plot (no time series)
            self.plot_impact_distribution(
                impact_data, save_path=str(viz_dir / "impact_distribution.png")
            )

            # Lifecycle comparison (no time series)
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
