# src/circman5/manufacturing/digital_twin/visualization/dashboard.py

"""
Digital Twin Dashboard for CIRCMAN5.0.

This module provides a comprehensive monitoring dashboard for the digital twin system,
enabling real-time monitoring of manufacturing processes and system status.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from pathlib import Path
import datetime
import json

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from ..core.state_manager import StateManager


class TwinDashboard:
    """
    Creates a comprehensive monitoring dashboard for the digital twin.

    The TwinDashboard provides a consolidated view of the digital twin state,
    including key performance indicators, process parameters, and system status.

    Attributes:
        state_manager: Reference to the state manager for accessing state
        constants: Constants service for accessing configuration
        logger: Logger instance for this class
    """

    def __init__(self, state_manager: StateManager):
        """
        Initialize the dashboard.

        Args:
            state_manager: StateManager instance for accessing state
        """
        self.state_manager = state_manager
        self.logger = setup_logger("twin_dashboard")
        self.constants = ConstantsService()
        self.config = self.constants.get_digital_twin_config()
        self.visualizations_dir = results_manager.get_path("visualizations")

        # Set up visualization style
        plt.style.use("default")
        plt.rcParams.update(
            {
                "figure.figsize": (14, 10),
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "axes.grid": True,
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
            }
        )

        self.logger.info("TwinDashboard initialized")

    def generate_dashboard(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> Optional[Figure]:
        """
        Generate a comprehensive dashboard visualization.

        Args:
            save_path: Optional path to save the dashboard visualization

        Returns:
            Optional[Figure]: The generated figure or None if error
        """
        try:
            # Get current state
            current_state = self.state_manager.get_current_state()

            if not current_state:
                self.logger.warning("No current state available for dashboard")
                return None

            # Get some historical data for trends
            history = self.state_manager.get_history(limit=max(50, 20))

            # Create dashboard figure with multiple panels
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle("Digital Twin Manufacturing Dashboard", fontsize=18)

            # Define grid for panels
            grid = GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.4)

            # Status panel (top left)
            ax_status = fig.add_subplot(grid[0, 0])
            self._create_status_panel(ax_status, current_state)

            # Production KPIs (top center-left)
            ax_production = fig.add_subplot(grid[0, 1])
            self._create_production_kpi_panel(ax_production, current_state)

            # Energy KPIs (top center-right)
            ax_energy = fig.add_subplot(grid[0, 2])
            self._create_energy_kpi_panel(ax_energy, current_state)

            # Quality KPIs (top right)
            ax_quality = fig.add_subplot(grid[0, 3])
            self._create_quality_kpi_panel(ax_quality, current_state)

            # Production line trend (middle left, spans 2 columns)
            ax_prod_trend = fig.add_subplot(grid[1, 0:2])
            self._create_production_trend_panel(ax_prod_trend, history)

            # Energy trend (middle right, spans 2 columns)
            ax_energy_trend = fig.add_subplot(grid[1, 2:4])
            self._create_energy_trend_panel(ax_energy_trend, history)

            # Material inventory (bottom left, spans 2 columns)
            ax_materials = fig.add_subplot(grid[2, 0:2])
            self._create_material_panel(ax_materials, current_state)

            # Environmental conditions (bottom right, spans 2 columns)
            ax_environment = fig.add_subplot(grid[2, 2:4])
            self._create_environment_panel(ax_environment, current_state, history)

            # Add timestamp
            timestamp_str = "Last Updated: "
            if "timestamp" in current_state:
                try:
                    timestamp = datetime.datetime.fromisoformat(
                        current_state["timestamp"]
                    )
                    timestamp_str += timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    timestamp_str += str(current_state.get("timestamp", "Unknown"))
            else:
                timestamp_str += datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            fig.text(0.99, 0.01, timestamp_str, ha="right", va="bottom", fontsize=8)

            # Save or show dashboard
            if save_path:
                # Use provided path - make sure directory exists
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"Dashboard saved to {save_path}")
            else:
                # Use results_manager for saving
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"digital_twin_dashboard_{timestamp}.png"

                # Get temp directory
                temp_dir = results_manager.get_path("temp")
                temp_path = temp_dir / filename

                # Make sure temp directory exists
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Save to temp path
                fig.savefig(temp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # Save to visualizations using results_manager
                results_manager.save_file(temp_path, "visualizations")

                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

                self.logger.info(
                    f"Dashboard saved to visualizations directory: {filename}"
                )

            return fig

        except Exception as e:
            self.logger.error(f"Error generating dashboard: {str(e)}")
            plt.close()  # Make sure to close figure in case of error
            return None

    def _create_status_panel(self, ax, state: Dict[str, Any]) -> None:
        """
        Create system status panel.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
        """
        ax.axis("off")  # Turn off axes

        # Get system status
        system_status = state.get("system_status", "Unknown")

        # Show system status with color-coded background
        status_color = "lightgreen" if system_status == "running" else "lightyellow"
        if system_status in ["error", "failure"]:
            status_color = "lightcoral"

        ax.text(
            0.5,
            0.7,
            f"System Status:\n{system_status.upper()}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=status_color, alpha=0.7),
        )

        # Show production line status if available
        if "production_line" in state and "status" in state["production_line"]:
            prod_status = state["production_line"]["status"]
            prod_color = "lightgreen" if prod_status == "running" else "lightyellow"
            if prod_status in ["error", "idle"]:
                prod_color = "lightcoral"

            ax.text(
                0.5,
                0.4,
                f"Production Line:\n{prod_status.upper()}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc=prod_color, alpha=0.7),
            )

        ax.set_title("System Status", fontsize=12)

    def _create_production_kpi_panel(self, ax, state: Dict[str, Any]) -> None:
        """
        Create production KPI panel.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
        """
        ax.axis("off")  # Turn off axes

        # Extract production KPIs
        kpis = {}

        if "production_line" in state:
            prod_line = state["production_line"]
            if "production_rate" in prod_line:
                kpis["Production Rate"] = f"{prod_line['production_rate']:.2f} units/hr"
            if "yield_rate" in prod_line:
                kpis["Yield Rate"] = f"{prod_line['yield_rate']:.2f}%"
            if "cycle_time" in prod_line:
                kpis["Cycle Time"] = f"{prod_line['cycle_time']:.2f} min"

        # Display KPIs
        if kpis:
            y_pos = 0.8
            for kpi, value in kpis.items():
                ax.text(
                    0.5,
                    y_pos,
                    f"{kpi}:\n{value}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.5),
                )
                y_pos -= 0.25
        else:
            ax.text(
                0.5,
                0.5,
                "No production KPIs available",
                horizontalalignment="center",
                verticalalignment="center",
            )

        ax.set_title("Production KPIs", fontsize=12)

    def _create_energy_kpi_panel(self, ax, state: Dict[str, Any]) -> None:
        """
        Create energy KPI panel.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
        """
        ax.axis("off")  # Turn off axes

        # Extract energy KPIs
        kpis = {}

        if "production_line" in state:
            prod_line = state["production_line"]
            if "energy_consumption" in prod_line:
                kpis[
                    "Energy Consumption"
                ] = f"{prod_line['energy_consumption']:.2f} kWh"

            # Calculate energy efficiency if both energy and production rate available
            if "energy_consumption" in prod_line and "production_rate" in prod_line:
                if prod_line["energy_consumption"] > 0:
                    efficiency = (
                        prod_line["production_rate"] / prod_line["energy_consumption"]
                    )
                    kpis["Energy Efficiency"] = f"{efficiency:.2f} units/kWh"

        # Display KPIs
        if kpis:
            y_pos = 0.8
            for kpi, value in kpis.items():
                ax.text(
                    0.5,
                    y_pos,
                    f"{kpi}:\n{value}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.5),
                )
                y_pos -= 0.25
        else:
            ax.text(
                0.5,
                0.5,
                "No energy KPIs available",
                horizontalalignment="center",
                verticalalignment="center",
            )

        ax.set_title("Energy KPIs", fontsize=12)

    def _create_quality_kpi_panel(self, ax, state: Dict[str, Any]) -> None:
        """
        Create quality KPI panel.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
        """
        ax.axis("off")  # Turn off axes

        # Extract quality KPIs
        kpis = {}

        if "production_line" in state:
            prod_line = state["production_line"]
            if "defect_rate" in prod_line:
                kpis["Defect Rate"] = f"{prod_line['defect_rate']:.2f}%"
            if "quality_score" in prod_line:
                kpis["Quality Score"] = f"{prod_line['quality_score']:.2f}/100"

        # Get material quality if available
        if "materials" in state:
            materials = state["materials"]
            quality_sum = 0
            quality_count = 0

            for material, properties in materials.items():
                if "quality" in properties:
                    quality_sum += properties["quality"]
                    quality_count += 1

            if quality_count > 0:
                avg_quality = quality_sum / quality_count
                kpis["Avg Material Quality"] = f"{avg_quality:.2f}%"

        # Display KPIs
        if kpis:
            y_pos = 0.8
            for kpi, value in kpis.items():
                ax.text(
                    0.5,
                    y_pos,
                    f"{kpi}:\n{value}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.5),
                )
                y_pos -= 0.25
        else:
            ax.text(
                0.5,
                0.5,
                "No quality KPIs available",
                horizontalalignment="center",
                verticalalignment="center",
            )

        ax.set_title("Quality KPIs", fontsize=12)

    def _create_production_trend_panel(self, ax, history: List[Dict[str, Any]]) -> None:
        """Create more insightful production trend visualization."""
        if not history:
            ax.text(
                0.5,
                0.5,
                "No historical data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Production Metrics")
            return

        # Extract timestamps, production rates, and energy consumption
        timestamps = []
        production_rates = []
        energy_consumption = []

        for state in history:
            if "timestamp" in state and "production_line" in state:
                prod_line = state["production_line"]
                if "production_rate" in prod_line and "energy_consumption" in prod_line:
                    try:
                        timestamp = datetime.datetime.fromisoformat(state["timestamp"])
                        timestamps.append(timestamp)
                        production_rates.append(prod_line["production_rate"])
                        energy_consumption.append(prod_line["energy_consumption"])
                    except (ValueError, TypeError):
                        continue

        if not timestamps:
            ax.text(
                0.5,
                0.5,
                "No production trend data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Production Metrics")
            return

        # Plot production rate
        line1 = ax.plot(
            timestamps,
            production_rates,
            "o-",
            linewidth=2,
            color="blue",
            label="Production Rate",
        )
        ax.set_ylabel("Production Rate (units/hr)", color="blue")

        # Create secondary y-axis for energy consumption
        ax2 = ax.twinx()
        line2 = ax2.plot(
            timestamps,
            energy_consumption,
            "s-",
            linewidth=2,
            color="red",
            label="Energy Consumption",
        )
        ax2.set_ylabel("Energy Consumption (kWh)", color="red")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left")

        ax.set_title("Production Metrics Over Time")
        ax.grid(True)
        ax.set_xlabel("Time")
        plt.setp(ax.get_xticklabels(), rotation=45)

    def _create_energy_trend_panel(self, ax, history: List[Dict[str, Any]]) -> None:
        """
        Create energy trend panel.

        Args:
            ax: Matplotlib axes to plot on
            history: List of historical states
        """
        if not history:
            ax.text(
                0.5,
                0.5,
                "No historical data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Energy Consumption Trend")
            return

        # Extract timestamps and energy consumption
        timestamps = []
        energy_consumptions = []

        for state in history:
            if (
                "timestamp" in state
                and "production_line" in state
                and "energy_consumption" in state["production_line"]
            ):
                try:
                    timestamp = datetime.datetime.fromisoformat(state["timestamp"])
                    timestamps.append(timestamp)
                    energy_consumptions.append(
                        state["production_line"]["energy_consumption"]
                    )
                except (ValueError, TypeError):
                    continue

        if not timestamps:
            ax.text(
                0.5,
                0.5,
                "No energy trend data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Energy Consumption Trend")
            return

        # Create line plot
        ax.plot(timestamps, energy_consumptions, "o-", linewidth=2, color="green")
        ax.set_title("Energy Consumption Trend")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy Consumption (kWh)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True)

    def _create_material_panel(self, ax, state: Dict[str, Any]) -> None:
        """
        Create material inventory panel.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
        """
        if "materials" not in state:
            ax.text(
                0.5,
                0.5,
                "No materials data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Material Inventory")
            return

        materials = state["materials"]

        # Extract inventory data
        material_names = []
        inventory_levels = []

        for material_name, material_data in materials.items():
            if isinstance(material_data, dict) and "inventory" in material_data:
                material_names.append(material_name)
                inventory_levels.append(material_data["inventory"])

        if not material_names:
            ax.text(
                0.5,
                0.5,
                "No inventory data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Material Inventory")
            return

        # Create bar chart
        x = np.arange(len(material_names))
        ax.bar(x, inventory_levels, width=0.6, color="purple")
        ax.set_xticks(x)
        ax.set_xticklabels(material_names, rotation=45)
        ax.set_title("Material Inventory")
        ax.set_ylabel("Inventory Level")

    def _create_environment_panel(
        self, ax, state: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> None:
        """
        Create environment panel.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
            history: List of historical states
        """
        if "environment" not in state:
            ax.text(
                0.5,
                0.5,
                "No environment data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Environmental Conditions")
            return

        env = state["environment"]

        # Get relevant environment parameters
        parameters = {}
        for key, value in env.items():
            if isinstance(value, (int, float)):
                parameters[key] = value

        if not parameters:
            ax.text(
                0.5,
                0.5,
                "No numeric environment data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Environmental Conditions")
            return

        # Create bar chart
        x = np.arange(len(parameters))
        ax.bar(x, list(parameters.values()), width=0.6, color="orange")
        ax.set_xticks(x)
        ax.set_xticklabels(list(parameters.keys()), rotation=45)
        ax.set_title("Environmental Conditions")
        ax.set_ylabel("Value")

        # Add environment status as text if available
        if "status" in env:
            status_text = f"Status: {env['status']}"
            ax.text(
                0.95,
                0.95,
                status_text,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.5),
            )
