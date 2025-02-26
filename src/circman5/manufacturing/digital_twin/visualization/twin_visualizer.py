# src/circman5/manufacturing/digital_twin/visualization/twin_visualizer.py

"""
Digital Twin Visualization module for CIRCMAN5.0.

This module provides visualization capabilities for the digital twin system,
enabling real-time monitoring and interactive visualization of manufacturing processes.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime
import json


from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from ..core.state_manager import StateManager


class TwinVisualizer:
    """
    Visualizes the digital twin state and provides monitoring capabilities.

    The TwinVisualizer creates visualizations of the digital twin state,
    including production line status, material flow, and process parameters.

    Attributes:
        state_manager: Reference to the state manager for accessing state
        constants: Constants service for accessing configuration
        logger: Logger instance for this class
    """

    def __init__(self, state_manager: StateManager):
        """
        Initialize the twin visualizer.

        Args:
            state_manager: StateManager instance for accessing state
        """
        self.state_manager = state_manager
        self.logger = setup_logger("twin_visualizer")
        self.constants = ConstantsService()
        self.config = self.constants.get_digital_twin_config()
        self.visualizations_dir = results_manager.get_path("visualizations")

        # Set up visualization style
        plt.style.use("default")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "axes.grid": True,
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
            }
        )

        self.logger.info("TwinVisualizer initialized")

    def visualize_current_state(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Visualize the current state of the digital twin.

        Args:
            save_path: Optional path to save the visualization
        """
        try:
            # Get current state
            current_state = self.state_manager.get_current_state()

            if not current_state:
                self.logger.warning("No current state available to visualize")
                return

            # Create figure with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Digital Twin - Current State", fontsize=16)

            # Visualize production line status (top left)
            self._visualize_production_line(axs[0, 0], current_state)

            # Visualize material inventory (top right)
            self._visualize_materials(axs[0, 1], current_state)

            # Visualize environment parameters (bottom left)
            self._visualize_environment(axs[1, 0], current_state)

            # Visualize system status (bottom right)
            self._visualize_system_status(axs[1, 1], current_state)

            # Adjust layout
            plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for title

            # Save or show visualization
            if save_path:
                # Use provided path - make sure directory exists
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"State visualization saved to {save_path}")
            else:
                # Use results_manager for saving
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"digital_twin_state_{timestamp}.png"

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
                    f"State visualization saved to visualizations directory: {filename}"
                )

        except Exception as e:
            self.logger.error(f"Error visualizing current state: {str(e)}")
            plt.close()  # Make sure to close figure in case of error
            raise

    def visualize_historical_states(
        self,
        metrics: List[str],
        limit: int = 20,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Visualize historical states for specified metrics.

        Args:
            metrics: List of metric paths to visualize (e.g., "production_line.temperature")
            limit:  Maximum number of historical states to include
            save_path: Optional path to save the visualization
        """
        try:
            # Retrieve historical states
            history = self.state_manager.get_history(limit=max(50, limit))
            for state in history:
                if "production_line" in state:
                    self.logger.debug(f"Historical state: {state['production_line']}")

            if not history:
                self.logger.warning("No historical data available to visualize")
                return

            # Create figure with one subplot per metric
            fig, axs = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)))
            fig.suptitle("Digital Twin - Historical Data", fontsize=16)

            # If there's only one metric, axs won't be a list by default
            if len(metrics) == 1:
                axs = [axs]

            # Extract timestamps from each state
            timestamps = []
            for st in history:
                if "timestamp" in st:
                    try:
                        dt_obj = datetime.datetime.fromisoformat(st["timestamp"])
                        timestamps.append(dt_obj)
                    except ValueError:
                        timestamps.append(None)
                else:
                    timestamps.append(None)

            # Plot each metric on a separate axis
            for i, metric_path in enumerate(metrics):
                values = []
                for st in history:
                    # Traverse the state via path notation (e.g. "production_line.temperature")
                    parts = metric_path.split(".")
                    val = st
                    try:
                        for p in parts:
                            val = val.get(p, {})
                        if isinstance(val, (int, float)):
                            values.append(val)
                        else:
                            values.append(None)
                    except (AttributeError, TypeError):
                        values.append(None)

                # Keep only the data points that are numeric and have valid timestamps
                valid_data = [
                    (t, v)
                    for t, v in zip(timestamps, values)
                    if t is not None and v is not None
                ]

                if valid_data:
                    plot_timestamps, plot_values = zip(*valid_data)
                    axs[i].plot(plot_timestamps, plot_values, "o-", linewidth=2)
                    axs[i].set_title(f"Metric: {metric_path}")
                    axs[i].set_ylabel("Value")
                    axs[i].grid(True)
                    axs[i].tick_params(axis="x", rotation=45)
                else:
                    axs[i].text(
                        0.5,
                        0.5,
                        f"No data available for {metric_path}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axs[i].transAxes,
                    )

            plt.tight_layout(rect=(0, 0, 1, 0.96))  # Space for suptitle

            # Save or show visualization
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"Historical state visualization saved to {save_path}")
            else:
                # Use results_manager for saving
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"digital_twin_history_{timestamp_str}.png"

                # Save to temp directory first
                temp_dir = results_manager.get_path("temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / filename

                fig.savefig(temp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # Move final file to 'visualizations' via results_manager
                results_manager.save_file(temp_path, "visualizations")

                # Remove local temp file
                if temp_path.exists():
                    temp_path.unlink()

                self.logger.info(
                    f"Historical state visualization saved to 'visualizations' directory as {filename}"
                )

        except Exception as e:
            self.logger.error(f"Error visualizing historical states: {str(e)}")
            plt.close()
            raise

    def _visualize_production_line(self, ax, state: Dict[str, Any]) -> None:
        """
        Visualize production line status.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
        """
        if "production_line" not in state:
            ax.text(
                0.5,
                0.5,
                "No production line data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            return

        prod_line = state["production_line"]

        # Get relevant production line parameters
        parameters = {}
        for key, value in prod_line.items():
            if isinstance(value, (int, float)):
                parameters[key] = value

        if not parameters:
            ax.text(
                0.5,
                0.5,
                "No numeric production line data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            return

        # Create bar chart
        x = np.arange(len(parameters))
        ax.bar(x, list(parameters.values()), width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(list(parameters.keys()), rotation=45)
        ax.set_title("Production Line Parameters")
        ax.set_ylabel("Value")

        # Add status as text if available
        if "status" in prod_line:
            status_text = f"Status: {prod_line['status']}"
            ax.text(
                0.95,
                0.95,
                status_text,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
            )

    def _visualize_materials(self, ax, state: Dict[str, Any]) -> None:
        """
        Visualize material inventory.

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
            return

        # Create bar chart
        x = np.arange(len(material_names))
        ax.bar(x, inventory_levels, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(material_names, rotation=45)
        ax.set_title("Material Inventory")
        ax.set_ylabel("Inventory Level")

    def _visualize_environment(self, ax, state: Dict[str, Any]) -> None:
        """
        Visualize environmental parameters.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
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
            return

        # Create bar chart
        x = np.arange(len(parameters))
        ax.bar(x, list(parameters.values()), width=0.6, color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(list(parameters.keys()), rotation=45)
        ax.set_title("Environmental Parameters")
        ax.set_ylabel("Value")

    def _visualize_system_status(self, ax, state: Dict[str, Any]) -> None:
        """
        Visualize overall system status.

        Args:
            ax: Matplotlib axes to plot on
            state: Current state dictionary
        """
        # Get timestamp
        timestamp = state.get("timestamp", "Unknown")

        # Get system status
        system_status = state.get("system_status", "Unknown")

        # Create a simple text-based visualization
        ax.axis("off")  # Turn off axes

        # Add timestamp
        ax.text(
            0.5,
            0.9,
            f"Timestamp: {timestamp}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

        # Add system status
        ax.text(
            0.5,
            0.7,
            f"System Status: {system_status}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.5),
        )

        # Add additional information if available
        info_y = 0.5
        for key, value in state.items():
            if key not in [
                "timestamp",
                "system_status",
                "production_line",
                "materials",
                "environment",
            ] and not isinstance(value, dict):
                ax.text(
                    0.5,
                    info_y,
                    f"{key}: {value}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                )
                info_y -= 0.1

    def visualize_material_flow(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Visualize material flow through the manufacturing process."""
        try:
            # Get current state and history
            current_state = self.state_manager.get_current_state()
            history = self.state_manager.get_history(20)

            if not current_state or not history:
                self.logger.warning("Insufficient data for material flow visualization")
                return

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle("Material Flow Analysis", fontsize=16)

            # Material inventory levels (left)
            if "materials" in current_state:
                materials = current_state["materials"]
                material_names = []
                inventory_levels = []

                for name, data in materials.items():
                    if isinstance(data, dict) and "inventory" in data:
                        material_names.append(name)
                        inventory_levels.append(data["inventory"])

                if material_names:
                    colors = cm.get_cmap("viridis")(
                        np.linspace(0, 1, len(material_names))
                    )
                    ax1.bar(material_names, inventory_levels, color=colors)
                    ax1.set_title("Current Material Inventory")
                    ax1.set_ylabel("Inventory Level")
                    ax1.tick_params(axis="x", rotation=45)

                    # Add value labels on bars
                    for i, v in enumerate(inventory_levels):
                        ax1.text(i, v + 0.1, f"{v:.1f}", ha="center")
                else:
                    ax1.text(
                        0.5,
                        0.5,
                        "No material inventory data available",
                        ha="center",
                        va="center",
                        transform=ax1.transAxes,
                    )

            # Material consumption over time (right)
            material_consumption = {}
            timestamps = []

            for state in history:
                if "timestamp" in state and "materials" in state:
                    try:
                        timestamp = datetime.datetime.fromisoformat(state["timestamp"])
                        timestamps.append(timestamp)

                        # Track inventory changes over time
                        for name, data in state["materials"].items():
                            if isinstance(data, dict) and "inventory" in data:
                                if name not in material_consumption:
                                    material_consumption[name] = []
                                material_consumption[name].append(data["inventory"])
                    except (ValueError, TypeError):
                        continue

            if timestamps and material_consumption:
                for name, values in material_consumption.items():
                    # Ensure lists are the same length
                    if len(values) == len(timestamps):
                        ax2.plot(timestamps, values, "o-", linewidth=2, label=name)

                ax2.set_title("Material Inventory Over Time")
                ax2.set_ylabel("Inventory Level")
                ax2.set_xlabel("Time")
                ax2.legend(loc="upper right")
                ax2.tick_params(axis="x", rotation=45)
                ax2.grid(True)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No historical material data available",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )

            plt.tight_layout(rect=(0, 0, 1, 0.95))

            # Save or return visualization
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"Material flow visualization saved to {save_path}")
            else:
                # Save using results_manager
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"material_flow_{timestamp_str}.png"
                temp_dir = results_manager.get_path("temp")
                temp_path = temp_dir / filename

                temp_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(temp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                results_manager.save_file(temp_path, "visualizations")
                if temp_path.exists():
                    temp_path.unlink()

                self.logger.info(
                    f"Material flow visualization saved to visualizations directory"
                )

        except Exception as e:
            self.logger.error(f"Error visualizing material flow: {str(e)}")
            plt.close()
