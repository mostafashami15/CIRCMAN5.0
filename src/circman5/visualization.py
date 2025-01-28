"""
Visualization module for manufacturing metrics and performance analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import pandas as pd


class ManufacturingVisualizer:
    """Creates visualizations for manufacturing metrics and performance data."""

    def __init__(self):
        """Initialize visualization settings."""
        plt.style.use("default")

        self.style_config = {
            "figure.figsize": (12, 8),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
        # plt.style.use('seaborn')
        for key, value in self.style_config.items():
            plt.rcParams[key] = value

    def plot_efficiency_trends(
        self, efficiency_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Plot efficiency metrics trends over time.

        Args:
            efficiency_data: DataFrame containing efficiency metrics
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Production Rate Trend
        sns.lineplot(data=efficiency_data, x="timestamp", y="production_rate", ax=ax1)
        ax1.set_title("Production Rate Over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Production Rate")

        # Energy Efficiency Trend
        sns.lineplot(data=efficiency_data, x="timestamp", y="energy_efficiency", ax=ax2)
        ax2.set_title("Energy Efficiency Over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Energy Efficiency (Output/Energy)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_quality_metrics(
        self, quality_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Visualize quality metrics distribution.

        Args:
            quality_data: DataFrame containing quality metrics
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Defect Rate Distribution
        sns.histplot(data=quality_data, x="defect_rate", ax=ax1)
        ax1.set_title("Defect Rate Distribution")
        ax1.set_xlabel("Defect Rate (%)")

        # Quality Score Trend
        sns.lineplot(data=quality_data, x="timestamp", y="quality_score", ax=ax2)
        ax2.set_title("Quality Score Trend")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Quality Score")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_resource_usage(
        self, resource_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Visualize resource utilization patterns.

        Args:
            resource_data: DataFrame containing resource metrics
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Resource Consumption Over Time
        resource_data.plot(
            x="timestamp",
            y=["material_consumption", "water_usage", "waste_generated"],
            ax=ax1,
        )
        ax1.set_title("Resource Usage Over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amount")

        # Resource Efficiency Trend
        sns.lineplot(data=resource_data, x="timestamp", y="resource_efficiency", ax=ax2)
        ax2.set_title("Resource Efficiency Trend")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Efficiency Score")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def create_performance_dashboard(
        self, monitor_data: Dict[str, pd.DataFrame], save_path: Optional[str] = None
    ) -> None:
        """
        Create comprehensive performance dashboard.

        Args:
            monitor_data: Dictionary containing different types of monitoring data
            save_path: Optional path to save the dashboard
        """
        fig = plt.figure(figsize=(15, 12))

        # Layout setup
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
        resource_data.plot(
            x="timestamp", y=["material_consumption", "water_usage"], ax=ax3
        )
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
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def create_kpi_dashboard(
        self, metrics_data: Dict[str, float], save_path: Optional[str] = None
    ) -> None:
        """
        Create a KPI dashboard with key manufacturing metrics.

        Args:
            metrics_data: Dictionary containing KPI values
            save_path: Optional path to save the dashboard
        """
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
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def _create_gauge_chart(self, ax, value: float, title: str, unit: str):
        """Helper method to create gauge charts for KPIs."""
        import numpy as np

        # Create gauge
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
