# src/circman5/manufacturing/reporting/visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from ...utils.logging_config import setup_logger
from ...utils.errors import DataError, ManufacturingError, ProcessError
from ..visualization_utils import VisualizationConfig
from .visualization_paths import VisualizationPathManager


class ManufacturingVisualizer:
    """Creates visualizations for manufacturing metrics and performance data."""

    def __init__(self):
        """Initialize visualization settings."""
        self.logger = setup_logger("manufacturing_visualizer")
        self.path_manager = VisualizationPathManager()
        VisualizationConfig.setup_style()
        self.colors = VisualizationConfig.COLOR_PALETTE

    def visualize_production_trends(
        self, production_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Enhanced production efficiency and output trends visualization with xlim handling."""
        if production_data.empty:
            self.logger.warning("No production data available for visualization")
            raise DataError("No production data available for visualization")

        # Ensure timestamps are unique and sorted
        production_data = production_data.sort_values("timestamp")

        fig = plt.figure(figsize=(12, 8))

        # Daily output plot with explicit xlim
        plt.subplot(2, 2, 1)
        daily_output = production_data.groupby(pd.Grouper(key="timestamp", freq="D"))[
            "output_amount"
        ].sum()
        if len(daily_output) > 1:  # Only plot if we have multiple points
            daily_output.plot(style=".-", title="Daily Production Output")
            plt.xlim(daily_output.index.min(), daily_output.index.max())
        plt.ylabel("Output Amount")

        # Yield rate distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=production_data, x="yield_rate", bins=20, stat="count")
        plt.title("Yield Rate Distribution")
        plt.xlabel("Yield Rate (%)")

        # Efficiency trend with explicit xlim
        plt.subplot(2, 2, 3)
        efficiency_trend = (
            production_data.set_index("timestamp")["yield_rate"]
            .rolling("7D", min_periods=1)
            .mean()
        )
        if len(efficiency_trend) > 1:  # Only plot if we have multiple points
            efficiency_trend.plot(title="7-Day Rolling Average Efficiency")
            plt.xlim(efficiency_trend.index.min(), efficiency_trend.index.max())
        plt.ylabel("Efficiency (%)")

        # Cycle times by production line
        plt.subplot(2, 2, 4)
        if len(production_data["production_line"].unique()) > 1:
            sns.boxplot(data=production_data, y="cycle_time", x="production_line")
        plt.title("Cycle Times by Production Line")

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()

    def visualize_quality_metrics(
        self, quality_data: pd.DataFrame, analyzer=None, save_path: Optional[str] = None
    ) -> None:
        """Enhanced quality control metrics visualization."""
        if quality_data.empty:
            self.logger.warning("No quality data available for visualization")
            raise DataError("No quality data available for visualization")

        fig = plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.histplot(data=quality_data, x="efficiency", bins=20, stat="count")
        plt.title("Cell Efficiency Distribution")
        plt.xlabel("Efficiency (%)")

        plt.subplot(2, 2, 2)
        daily_defects = quality_data.groupby(
            pd.Grouper(key="test_timestamp", freq="D"), observed=True
        )["defect_rate"].mean()
        daily_defects.plot(style=".-", title="Daily Average Defect Rate")
        plt.ylabel("Defect Rate (%)")

        plt.subplot(2, 2, 3)
        sns.boxplot(data=quality_data, y="thickness_uniformity", orientation="vertical")
        plt.title("Thickness Uniformity Distribution")

        plt.subplot(2, 2, 4)
        if analyzer:
            quality_trends = analyzer.identify_quality_trends(quality_data)
            if quality_trends:
                pd.DataFrame(quality_trends)["efficiency_trend"].plot(
                    title="Efficiency Trend Analysis"
                )

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()

    def visualize_sustainability_indicators(
        self,
        material_data: pd.DataFrame,
        energy_data: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Enhanced sustainability metrics visualization."""
        if material_data.empty:
            self.logger.warning("No material flow data available for visualization")
            raise DataError("No material flow data available for visualization")

        fig = plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        waste_by_type = material_data.groupby("material_type", observed=True)[
            "waste_generated"
        ].sum()
        waste_by_type.plot(kind="bar", title="Waste Generation by Material Type")
        plt.xticks(rotation=45)
        plt.ylabel("Waste Amount")

        plt.subplot(2, 2, 2)
        recycling_rates = material_data.groupby("material_type", observed=True).agg(
            {"recycled_amount": "sum", "waste_generated": "sum"}
        )
        recycling_rates["rate"] = (
            recycling_rates["recycled_amount"]
            / recycling_rates["waste_generated"]
            * 100
        )
        recycling_rates["rate"].plot(
            kind="bar", title="Recycling Rate by Material Type"
        )
        plt.ylabel("Recycling Rate (%)")
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 3)
        material_efficiency_trend = (
            material_data.groupby(pd.Grouper(key="timestamp", freq="W"), observed=True)
            .agg({"recycled_amount": "sum", "waste_generated": "sum"})
            .assign(
                efficiency=lambda x: (x["recycled_amount"] / x["waste_generated"] * 100)
            )["efficiency"]
        )
        material_efficiency_trend.plot(title="Weekly Material Efficiency Trend")
        plt.ylabel("Efficiency (%)")

        plt.subplot(2, 2, 4)
        if energy_data is not None and not energy_data.empty:
            energy_mix = energy_data.groupby("energy_source", observed=True)[
                "energy_consumption"
            ].sum()
            energy_mix.plot(
                kind="pie", autopct="%1.1f%%", title="Energy Source Distribution"
            )

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()

    def visualize_energy_patterns(
        self, energy_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Enhanced energy consumption pattern visualization."""
        if energy_data.empty:
            self.logger.warning("No energy data available for visualization")
            raise DataError("No energy data available for visualization")

        fig = plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        energy_by_source = energy_data.groupby("energy_source")[
            "energy_consumption"
        ].sum()
        energy_by_source.plot(
            kind="pie", autopct="%1.1f%%", title="Energy Consumption by Source"
        )

        plt.subplot(2, 2, 2)
        hourly_consumption = energy_data.groupby(energy_data["timestamp"].dt.hour)[
            "energy_consumption"
        ].mean()
        hourly_consumption.plot(style=".-", title="Average Hourly Energy Consumption")
        plt.xlabel("Hour of Day")
        plt.ylabel("Energy Consumption")

        plt.subplot(2, 2, 3)
        daily_efficiency = energy_data.groupby(pd.Grouper(key="timestamp", freq="D"))[
            "efficiency_rate"
        ].mean()
        daily_efficiency.plot(title="Daily Energy Efficiency Rate")
        plt.ylabel("Efficiency Rate")

        plt.subplot(2, 2, 4)
        weekly_consumption = energy_data.groupby(pd.Grouper(key="timestamp", freq="W"))[
            "energy_consumption"
        ].sum()
        weekly_consumption.plot(kind="bar", title="Weekly Energy Consumption")
        plt.xticks(rotation=45)
        plt.ylabel("Energy Consumption")

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()

    def create_performance_dashboard(
        self, monitor_data: Dict[str, pd.DataFrame], save_path: Optional[str] = None
    ) -> None:
        """Create comprehensive performance dashboard."""
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)

        ax1 = fig.add_subplot(gs[0, 0])  # Efficiency
        ax2 = fig.add_subplot(gs[0, 1])  # Quality
        ax3 = fig.add_subplot(gs[1, :])  # Resource Usage
        ax4 = fig.add_subplot(gs[2, :])  # Combined Trends

        # Efficiency Plot
        efficiency_data = monitor_data["efficiency"]
        sns.lineplot(data=efficiency_data, x="timestamp", y="production_rate", ax=ax1)
        ax1.set_title("Production Efficiency")

        # Quality Plot
        quality_data = monitor_data["quality"]
        sns.lineplot(data=quality_data, x="timestamp", y="quality_score", ax=ax2)
        ax2.set_title("Quality Metrics")

        # Resource Usage
        resource_data = monitor_data["resources"]
        resource_columns = ["material_consumption"]
        resource_data.plot(x="timestamp", y=resource_columns, ax=ax3)
        ax3.set_title("Resource Utilization")

        # Combined Performance Indicators
        combined_metrics = pd.DataFrame(
            {
                "timestamp": efficiency_data["timestamp"],
                "efficiency": efficiency_data["production_rate"]
                / efficiency_data["production_rate"].max(),
                "quality": quality_data["quality_score"] / 100,
                "resource_efficiency": resource_data["resource_efficiency"],
            }
        )

        combined_metrics.plot(x="timestamp", ax=ax4)
        ax4.set_title("Combined Performance Indicators")
        ax4.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()

    def create_kpi_dashboard(
        self, metrics_data: Dict[str, float], save_path: Optional[str] = None
    ) -> None:
        """Create a KPI dashboard with key manufacturing metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Manufacturing KPIs Dashboard", fontsize=16)

        # Efficiency KPIs
        self._create_gauge_chart(
            axes[0, 0], metrics_data.get("efficiency", 0), "Production Efficiency", "%"
        )

        # Quality KPIs
        self._create_gauge_chart(
            axes[0, 1], metrics_data.get("quality_score", 0), "Quality Score", "%"
        )

        # Resource KPIs
        self._create_gauge_chart(
            axes[1, 0],
            metrics_data.get("resource_efficiency", 0),
            "Resource Efficiency",
            "%",
        )

        # Energy KPIs
        self._create_gauge_chart(
            axes[1, 1],
            metrics_data.get("energy_efficiency", 0),
            "Energy Efficiency",
            "%",
        )

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()

    def _create_gauge_chart(self, ax, value: float, title: str, unit: str) -> None:
        """Helper method to create gauge charts for KPIs."""
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # Background
        x_bg = r * np.cos(theta)
        y_bg = r * np.sin(theta)
        ax.plot(x_bg, y_bg, "lightgray")

        # Value
        value_normalized = min(max(value, 0), 100) / 100
        theta_value = np.linspace(0, np.pi * value_normalized, 100)
        x_val = r * np.cos(theta_value)
        y_val = r * np.sin(theta_value)
        ax.plot(x_val, y_val, "blue", linewidth=3)

        # Decorations
        ax.set_title(f"{title}\n{value:.1f}{unit}")
        ax.axis("equal")
        ax.axis("off")

    def generate_visualization(
        self, metric_type: str, data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Create visualizations for different metric types."""
        try:
            # Get proper save path if none provided
            if save_path is None:
                save_path = str(self.path_manager.get_visualization_path(metric_type))

            if metric_type == "production":
                self.visualize_production_trends(data, save_path)
            elif metric_type == "energy":
                self.visualize_energy_patterns(data, save_path)
            elif metric_type == "quality":
                self.visualize_quality_metrics(data, None, save_path)
            elif metric_type == "sustainability":
                self.visualize_sustainability_indicators(data, None, save_path)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

            self.logger.info(
                f"Generated visualization for {metric_type} at {save_path}"
            )

        except Exception as e:
            self.logger.error(f"Error generating visualization: {str(e)}")
            plt.close()
            raise ProcessError(f"Visualization generation failed: {str(e)}")

    def plot_efficiency_trends(
        self, efficiency_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Plot efficiency trends over time."""
        if efficiency_data.empty:
            self.logger.warning("No efficiency data available for visualization")
            raise DataError("No efficiency data available for visualization")

        if save_path is None:
            save_path = str(
                self.path_manager.get_visualization_path(
                    "efficiency", "efficiency_trends.png"
                )
            )

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Production rate trend
        efficiency_data.plot(x="timestamp", y="production_rate", ax=axes[0])
        axes[0].set_title("Production Rate Over Time")
        axes[0].set_ylabel("Production Rate")

        # Energy efficiency trend
        if "energy_efficiency" in efficiency_data.columns:
            efficiency_data.plot(x="timestamp", y="energy_efficiency", ax=axes[1])
            axes[1].set_title("Energy Efficiency Over Time")
            axes[1].set_ylabel("Energy Efficiency (%)")

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()

    def plot_quality_metrics(
        self, quality_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Plot quality control metrics."""
        if quality_data.empty:
            self.logger.warning("No quality data available for visualization")
            raise DataError("No quality data available for visualization")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Quality score distribution
        sns.histplot(data=quality_data, x="quality_score", ax=axes[0, 0])
        axes[0, 0].set_title("Quality Score Distribution")

        # Defect rate over time
        if "defect_rate" in quality_data.columns:
            quality_data.plot(x="timestamp", y="defect_rate", ax=axes[0, 1])
            axes[0, 1].set_title("Defect Rate Over Time")

        # Quality metrics over time
        if "quality_score" in quality_data.columns:
            quality_data.plot(x="timestamp", y="quality_score", ax=axes[1, 0])
            axes[1, 0].set_title("Quality Score Over Time")

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()

    def plot_resource_usage(
        self, resource_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Plot resource usage metrics."""
        if resource_data.empty:
            self.logger.warning("No resource data available for visualization")
            raise DataError("No resource data available for visualization")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Material consumption over time
        if "material_consumption" in resource_data.columns:
            resource_data.plot(x="timestamp", y="material_consumption", ax=axes[0, 0])
            axes[0, 0].set_title("Material Consumption Over Time")

        # Water usage over time
        if "water_usage" in resource_data.columns:
            resource_data.plot(x="timestamp", y="water_usage", ax=axes[0, 1])
            axes[0, 1].set_title("Water Usage Over Time")

        # Resource efficiency trend
        if "resource_efficiency" in resource_data.columns:
            resource_data.plot(x="timestamp", y="resource_efficiency", ax=axes[1, 0])
            axes[1, 0].set_title("Resource Efficiency Over Time")

        plt.tight_layout()

        if save_path:
            VisualizationConfig.save_figure(fig, save_path)
            plt.close()
        else:
            plt.show()
