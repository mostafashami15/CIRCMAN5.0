# src/circman5/manufacturing/digital_twin/visualization/twin_visualizer.py

"""
Digital Twin Visualization module for CIRCMAN5.0.

This module provides visualization capabilities for the digital twin system,
enabling real-time monitoring and interactive visualization of manufacturing processes.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

    def visualize_material_flow_sankey(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create a Sankey diagram visualization of material flow in the manufacturing process.

        Args:
            save_path: Optional path to save the visualization
        """
        try:
            # Get current state and history
            current_state = self.state_manager.get_current_state()
            history = self.state_manager.get_history(limit=20)

            if not current_state or not history:
                self.logger.warning(
                    "Insufficient data for material flow Sankey visualization"
                )
                return

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.suptitle("Material Flow Sankey Diagram", fontsize=16)

            # Extract material flow data
            material_inputs = {}
            material_outputs = {}
            waste_generated = {}

            # Process material data from state
            if "materials" in current_state:
                materials = current_state["materials"]

                # Extract material names and quantities
                for material_name, material_data in materials.items():
                    if isinstance(material_data, dict):
                        # Get inventory and incoming/outgoing flows
                        if "inventory" in material_data:
                            material_inputs[material_name] = material_data.get(
                                "input_rate", 10.0
                            )
                            material_outputs[material_name] = material_data.get(
                                "output_rate", 8.0
                            )
                            waste_generated[material_name] = material_data.get(
                                "waste_rate", 2.0
                            )

            # If no material data in state, try to estimate from history
            if not material_inputs and history:
                for state in history:
                    if "materials" in state:
                        for name, data in state.get("materials", {}).items():
                            if isinstance(data, dict) and "inventory" in data:
                                # Estimate flows based on inventory changes
                                material_inputs[name] = (
                                    material_inputs.get(name, 0) + 10
                                )
                                material_outputs[name] = (
                                    material_outputs.get(name, 0) + 8
                                )
                                waste_generated[name] = waste_generated.get(name, 0) + 2

                # Average over history length if we extracted data
                if material_inputs:
                    for name in material_inputs:
                        material_inputs[name] /= len(history)
                        material_outputs[name] /= len(history)
                        waste_generated[name] /= len(history)

            # If still no data, use example data
            if not material_inputs:
                material_inputs = {
                    "silicon_wafer": 100.0,
                    "solar_glass": 80.0,
                    "metallization_paste": 40.0,
                    "eva_sheet": 60.0,
                    "backsheet": 60.0,
                }
                material_outputs = {
                    "silicon_wafer": 90.0,
                    "solar_glass": 75.0,
                    "metallization_paste": 35.0,
                    "eva_sheet": 57.0,
                    "backsheet": 58.0,
                }
                waste_generated = {
                    "silicon_wafer": 10.0,
                    "solar_glass": 5.0,
                    "metallization_paste": 5.0,
                    "eva_sheet": 3.0,
                    "backsheet": 2.0,
                }

            # Create Sankey diagram
            from matplotlib.sankey import Sankey

            # Initialize Sankey diagram
            sankey = Sankey(
                ax=ax,
                unit="kg",
                scale=0.01,
                offset=0.2,
                head_angle=120,
                format="%.1f",
                gap=0.5,
            )

            # Add flows for each material
            material_colors = {
                "silicon_wafer": "royalblue",
                "solar_glass": "skyblue",
                "metallization_paste": "silver",
                "eva_sheet": "lightgreen",
                "backsheet": "lightgray",
                "aluminum_frame": "gray",
                "copper_wiring": "orange",
            }

            # Starting position for flows
            pos_y = 0

            # Add Sankey flows for each material
            for material_name in material_inputs:
                input_value = material_inputs[material_name]
                output_value = material_outputs[material_name]
                waste_value = waste_generated[material_name]

                # Use default color if material not in colors dict
                color = material_colors.get(material_name, "blue")

                # Add material flow
                sankey.add(
                    flows=[input_value, -output_value, -waste_value],
                    labels=[f"{material_name}", "Product", "Waste"],
                    orientations=[0, 0, 1],
                    pathlengths=[0.4, 0.4, 0.4],
                    trunklength=1.0,
                    color=color,
                    alpha=0.8,
                )

                # Increment vertical position for next material
                pos_y += 1

            # Finish the diagram
            diagrams = sankey.finish()

            # Add title and additional info
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.figtext(0.5, 0.02, f"Generated: {timestamp}", ha="center", fontsize=10)

            # Adjust layout
            plt.tight_layout(rect=(0.01, 0.05, 0.99, 0.95))

            # Save or show the visualization
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"Material flow Sankey diagram saved to {save_path}")
            else:
                # Save using results_manager
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"material_flow_sankey_{timestamp_str}.png"

                # Get temp directory from results_manager
                temp_dir = results_manager.get_path("temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / filename

                # Save to temporary path
                plt.savefig(temp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # Save to visualizations directory using results_manager
                results_manager.save_file(temp_path, "visualizations")

                # Remove temporary file
                if temp_path.exists():
                    temp_path.unlink()

                self.logger.info(
                    f"Material flow Sankey diagram saved to visualizations directory"
                )

        except Exception as e:
            self.logger.error(f"Error creating material flow Sankey diagram: {str(e)}")
            plt.close()  # Make sure to close any open figures

    def visualize_efficiency_heatmap(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create a heatmap visualization of efficiency metrics.

        Args:
            save_path: Optional path to save the visualization
        """
        try:
            # Get history data
            history = self.state_manager.get_history(limit=30)

            if not history:
                self.logger.warning(
                    "No historical data available for efficiency heatmap"
                )
                return

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.suptitle("Manufacturing Efficiency Metrics Heatmap", fontsize=16)

            # Extract efficiency metrics from history
            timestamps = []
            metrics_data = {}
            metrics_to_extract = [
                "production_rate",
                "energy_consumption",
                "defect_rate",
                "yield_rate",
                "efficiency",
            ]

            for state in history:
                if "timestamp" in state:
                    try:
                        # Extract timestamp
                        timestamp = datetime.datetime.fromisoformat(state["timestamp"])
                        timestamps.append(timestamp)

                        # Extract metrics from production_line
                        if "production_line" in state:
                            prod_line = state["production_line"]
                            for metric in metrics_to_extract:
                                if metric in prod_line:
                                    if metric not in metrics_data:
                                        metrics_data[metric] = []
                                    metrics_data[metric].append(prod_line[metric])
                    except (ValueError, TypeError):
                        continue

            # If no data could be extracted, return
            if not timestamps or not metrics_data:
                self.logger.warning(
                    "Could not extract sufficient data for efficiency heatmap"
                )
                return

            # Normalize data for heatmap
            for metric in metrics_data:
                if len(metrics_data[metric]) < len(timestamps):
                    # Pad with NaN if needed
                    metrics_data[metric].extend(
                        [float("nan")] * (len(timestamps) - len(metrics_data[metric]))
                    )

                # Ensure all values are numeric
                metrics_data[metric] = [
                    float(val) if val is not None else float("nan")
                    for val in metrics_data[metric]
                ]

                # Normalize values between 0 and 1
                metric_values = np.array(metrics_data[metric])
                if not np.isnan(metric_values).all():
                    min_val = np.nanmin(metric_values)
                    max_val = np.nanmax(metric_values)
                    if max_val > min_val:
                        metrics_data[metric] = (metric_values - min_val) / (
                            max_val - min_val
                        )

            # Create data for heatmap
            heatmap_data = []
            metric_names = []

            for metric, values in metrics_data.items():
                if len(values) == len(timestamps):
                    heatmap_data.append(values)
                    # Format metric name for display
                    metric_names.append(metric.replace("_", " ").title())

            # If not enough data for heatmap, return
            if not heatmap_data:
                self.logger.warning("Insufficient data for heatmap visualization")
                return

            # Format timestamps for display
            tick_labels = [ts.strftime("%H:%M:%S") for ts in timestamps]

            # Create heatmap
            heatmap = ax.imshow(
                heatmap_data, aspect="auto", cmap="viridis", interpolation="nearest"
            )

            # Add colorbar
            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label("Normalized Value")

            # Configure axes
            ax.set_yticks(np.arange(len(metric_names)))
            ax.set_yticklabels(metric_names)

            # Display every nth tick to avoid crowding
            n = max(1, len(timestamps) // 10)
            ax.set_xticks(np.arange(0, len(timestamps), n))
            ax.set_xticklabels(
                [tick_labels[i] for i in range(0, len(timestamps), n)], rotation=45
            )

            ax.set_xlabel("Time")
            ax.set_title("Efficiency Metrics Over Time")

            # Add grid
            ax.grid(False)

            # Add timestamp
            generation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.figtext(
                0.5, 0.01, f"Generated: {generation_time}", ha="center", fontsize=10
            )

            # Adjust layout
            plt.tight_layout(rect=(0.01, 0.05, 0.99, 0.95))

            # Save or show the visualization
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"Efficiency heatmap saved to {save_path}")
            else:
                # Save using results_manager
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"efficiency_heatmap_{timestamp_str}.png"

                # Get temp directory from results_manager
                temp_dir = results_manager.get_path("temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / filename

                # Save to temporary path
                plt.savefig(temp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # Save to visualizations directory using results_manager
                results_manager.save_file(temp_path, "visualizations")

                # Remove temporary file
                if temp_path.exists():
                    temp_path.unlink()

                self.logger.info(
                    f"Efficiency heatmap saved to visualizations directory"
                )

        except Exception as e:
            self.logger.error(f"Error creating efficiency heatmap: {str(e)}")
            plt.close()  # Make sure to close any open figures

    def visualize_state_comparison(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
        labels: Tuple[str, str] = ("State 1", "State 2"),
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Create a comparison visualization between two digital twin states.

        Args:
            state1: First state to compare
            state2: Second state to compare
            labels: Labels for the two states
            save_path: Optional path to save the visualization
        """
        try:
            if not state1 or not state2:
                self.logger.warning("Missing state data for comparison")
                return

            # Create figure
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle("Digital Twin State Comparison", fontsize=16)

            # Define grid layout
            gs = GridSpec(3, 2, figure=fig, wspace=0.4, hspace=0.4)

            # Add timestamps
            timestamp1 = "Unknown"
            timestamp2 = "Unknown"

            if "timestamp" in state1:
                try:
                    dt1 = datetime.datetime.fromisoformat(state1["timestamp"])
                    timestamp1 = dt1.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    timestamp1 = str(state1.get("timestamp", "Unknown"))

            if "timestamp" in state2:
                try:
                    dt2 = datetime.datetime.fromisoformat(state2["timestamp"])
                    timestamp2 = dt2.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    timestamp2 = str(state2.get("timestamp", "Unknown"))

            # Add subtitle with timestamps
            fig.text(
                0.5,
                0.95,
                f"{labels[0]}: {timestamp1} vs {labels[1]}: {timestamp2}",
                ha="center",
                fontsize=12,
            )

            # Production line parameters (top row)
            ax_prod = fig.add_subplot(gs[0, :])
            self._compare_production_parameters(ax_prod, state1, state2, labels)

            # Material inventory (middle left)
            ax_materials = fig.add_subplot(gs[1, 0])
            self._compare_materials(ax_materials, state1, state2, labels)

            # Energy metrics (middle right)
            ax_energy = fig.add_subplot(gs[1, 1])
            self._compare_energy(ax_energy, state1, state2, labels)

            # Quality metrics (bottom row)
            ax_quality = fig.add_subplot(gs[2, :])
            self._compare_quality(ax_quality, state1, state2, labels)

            # Add generation timestamp
            generation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.figtext(
                0.5, 0.01, f"Generated: {generation_time}", ha="center", fontsize=10
            )

            # Save or show visualization
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"State comparison saved to {save_path}")
            else:
                # Use results_manager
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"state_comparison_{timestamp_str}.png"

                # Get temp directory
                temp_dir = results_manager.get_path("temp")
                temp_path = temp_dir / filename
                temp_dir.mkdir(parents=True, exist_ok=True)

                plt.savefig(temp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # Save to visualizations directory using results_manager
                results_manager.save_file(temp_path, "visualizations")

                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

                self.logger.info(f"State comparison saved to visualizations directory")

        except Exception as e:
            self.logger.error(f"Error creating state comparison: {str(e)}")
            plt.close()

    def _compare_production_parameters(
        self,
        ax,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
        labels: Tuple[str, str],
    ) -> None:
        """Compare production line parameters between two states."""
        # Extract production line parameters
        params = ["temperature", "production_rate", "energy_consumption"]
        values1 = []
        values2 = []

        for param in params:
            if "production_line" in state1 and param in state1["production_line"]:
                values1.append(state1["production_line"][param])
            else:
                values1.append(0)

            if "production_line" in state2 and param in state2["production_line"]:
                values2.append(state2["production_line"][param])
            else:
                values2.append(0)

        # Format parameter names for display
        param_labels = [p.replace("_", " ").title() for p in params]

        # Set up bar positions
        x = np.arange(len(params))
        width = 0.35

        # Create grouped bar chart
        ax.bar(x - width / 2, values1, width, label=labels[0], color="royalblue")
        ax.bar(x + width / 2, values2, width, label=labels[1], color="lightgreen")

        # Add percentage change labels
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            if v1 > 0:  # Avoid division by zero
                pct_change = (v2 - v1) / v1 * 100
                color = "green" if pct_change >= 0 else "red"
                ax.text(
                    i,
                    max(v1, v2) + 0.05 * max(v1, v2),
                    f"{pct_change:+.1f}%",
                    ha="center",
                    color=color,
                )

        # Configure axis
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels)
        ax.legend()
        ax.set_title("Production Line Parameters")
        ax.grid(True, linestyle="--", alpha=0.7)

    def _compare_materials(
        self,
        ax,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
        labels: Tuple[str, str],
    ) -> None:
        """Compare material inventory between two states."""
        # Extract material inventory
        materials1 = {}
        materials2 = {}

        if "materials" in state1:
            for name, data in state1["materials"].items():
                if isinstance(data, dict) and "inventory" in data:
                    materials1[name] = data["inventory"]

        if "materials" in state2:
            for name, data in state2["materials"].items():
                if isinstance(data, dict) and "inventory" in data:
                    materials2[name] = data["inventory"]

        # Get common materials
        common_materials = set(materials1.keys()) & set(materials2.keys())

        if not common_materials:
            ax.text(
                0.5,
                0.5,
                "No common materials for comparison",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Material Inventory")
            return

        # Prepare data for plotting
        material_names = list(common_materials)
        inventory1 = [materials1.get(m, 0) for m in material_names]
        inventory2 = [materials2.get(m, 0) for m in material_names]

        # Format material names for display
        display_names = [m.replace("_", " ").title() for m in material_names]

        # Set up bar positions
        x = np.arange(len(material_names))
        width = 0.35

        # Create grouped bar chart
        ax.bar(x - width / 2, inventory1, width, label=labels[0], color="royalblue")
        ax.bar(x + width / 2, inventory2, width, label=labels[1], color="lightgreen")

        # Configure axis
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        ax.legend()
        ax.set_title("Material Inventory")
        ax.grid(True, linestyle="--", alpha=0.7)

    def _compare_energy(
        self,
        ax,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
        labels: Tuple[str, str],
    ) -> None:
        """Compare energy metrics between two states."""
        # Extract energy metrics
        metrics = ["energy_consumption", "energy_efficiency"]
        values1 = []
        values2 = []

        for metric in metrics:
            if "production_line" in state1 and metric in state1["production_line"]:
                values1.append(state1["production_line"][metric])
            else:
                values1.append(0)

            if "production_line" in state2 and metric in state2["production_line"]:
                values2.append(state2["production_line"][metric])
            else:
                values2.append(0)

        # Format metric names for display
        metric_labels = [m.replace("_", " ").title() for m in metrics]

        # Set up bar positions
        x = np.arange(len(metrics))
        width = 0.35

        # Create grouped bar chart
        ax.bar(x - width / 2, values1, width, label=labels[0], color="royalblue")
        ax.bar(x + width / 2, values2, width, label=labels[1], color="lightgreen")

        # Add percentage change labels
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            if v1 > 0:  # Avoid division by zero
                pct_change = (v2 - v1) / v1 * 100
                # For energy consumption, lower is better, so invert color logic
                if i == 0:  # energy_consumption
                    color = "green" if pct_change <= 0 else "red"
                else:  # energy_efficiency
                    color = "green" if pct_change >= 0 else "red"
                ax.text(
                    i,
                    max(v1, v2) + 0.05 * max(v1, v2),
                    f"{pct_change:+.1f}%",
                    ha="center",
                    color=color,
                )

        # Configure axis
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.set_title("Energy Metrics")
        ax.grid(True, linestyle="--", alpha=0.7)

    def _compare_quality(
        self,
        ax,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
        labels: Tuple[str, str],
    ) -> None:
        """Compare quality metrics between two states."""
        # Extract quality metrics
        quality_metrics = []
        values1 = []
        values2 = []

        # Direct quality metrics in production_line
        if "production_line" in state1 and "production_line" in state2:
            prod1 = state1["production_line"]
            prod2 = state2["production_line"]

            potential_metrics = [
                "quality_score",
                "defect_rate",
                "yield_rate",
                "efficiency",
            ]
            for metric in potential_metrics:
                if metric in prod1 or metric in prod2:
                    quality_metrics.append(metric)
                    values1.append(prod1.get(metric, 0))
                    values2.append(prod2.get(metric, 0))

        # Material quality
        avg_quality1 = 0
        avg_quality2 = 0
        quality_count1 = 0
        quality_count2 = 0

        if "materials" in state1:
            for material, data in state1["materials"].items():
                if isinstance(data, dict) and "quality" in data:
                    avg_quality1 += data["quality"]
                    quality_count1 += 1

        if "materials" in state2:
            for material, data in state2["materials"].items():
                if isinstance(data, dict) and "quality" in data:
                    avg_quality2 += data["quality"]
                    quality_count2 += 1

        if quality_count1 > 0 and quality_count2 > 0:
            quality_metrics.append("avg_material_quality")
            values1.append(avg_quality1 / quality_count1)
            values2.append(avg_quality2 / quality_count2)

        # If no quality metrics found, show message
        if not quality_metrics:
            ax.text(
                0.5,
                0.5,
                "No quality metrics available for comparison",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Quality Metrics")
            return

        # Format metric names for display
        metric_labels = [m.replace("_", " ").title() for m in quality_metrics]

        # Set up bar positions
        x = np.arange(len(quality_metrics))
        width = 0.35

        # Create grouped bar chart
        ax.bar(x - width / 2, values1, width, label=labels[0], color="royalblue")
        ax.bar(x + width / 2, values2, width, label=labels[1], color="lightgreen")

        # Add percentage change labels
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            if v1 > 0:  # Avoid division by zero
                pct_change = (v2 - v1) / v1 * 100
                # For defect_rate, lower is better, so invert color logic
                if quality_metrics[i] == "defect_rate":
                    color = "green" if pct_change <= 0 else "red"
                else:
                    color = "green" if pct_change >= 0 else "red"
                ax.text(
                    i,
                    max(v1, v2) + 0.05 * max(v1, v2),
                    f"{pct_change:+.1f}%",
                    ha="center",
                    color=color,
                )

        # Configure axis
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.set_title("Quality Metrics")
        ax.grid(True, linestyle="--", alpha=0.7)

    def visualize_parameter_sensitivity(
        self,
        parameter: str,
        value_range: List[float],
        target_metrics: List[str] = [
            "production_rate",
            "energy_consumption",
            "quality_score",
        ],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Create a parameter sensitivity visualization showing how changes to a parameter
        affect different metrics in the digital twin.

        Args:
            parameter: The parameter to adjust
            value_range: List of values to test for the parameter
            target_metrics: List of metrics to track
            save_path: Optional path to save the visualization
        """
        try:
            # Get current state
            base_state = self.state_manager.get_current_state()

            if not base_state:
                self.logger.warning(
                    "No current state available for parameter sensitivity analysis"
                )
                return

            # Create figure
            fig, axes = plt.subplots(
                len(target_metrics), 1, figsize=(12, 4 * len(target_metrics))
            )
            fig.suptitle(
                f"Parameter Sensitivity Analysis: {parameter.replace('_', ' ').title()}",
                fontsize=16,
            )

            # If only one metric, axes won't be an array
            if len(target_metrics) == 1:
                axes = [axes]

            # Prepare data storage for results
            results = {metric: [] for metric in target_metrics}

            # Import the Digital Twin to run simulations
            from ..core.twin_core import DigitalTwin

            digital_twin = DigitalTwin()  # a new instance
            digital_twin.state_manager = (
                self.state_manager
            )  # Set the state manager after initialization

            # Run simulations for each parameter value
            for value in value_range:
                # Prepare parameters for this simulation
                # We need to create the right parameter structure based on the parameter name
                simulation_params = {}

                # Handle nested parameters with dot notation
                if "." in parameter:
                    parts = parameter.split(".")
                    current = simulation_params
                    for part in parts[:-1]:
                        current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    simulation_params[parameter] = value

                # Run a simulation with this parameter value
                simulated_states = digital_twin.simulate(
                    steps=3, parameters=simulation_params
                )

                # Extract relevant metrics from final state
                final_state = simulated_states[-1]

                # Process each target metric
                for metric in target_metrics:
                    # Handle nested metrics with dot notation
                    if "." in metric:
                        parts = metric.split(".")
                        current = final_state
                        try:
                            for part in parts:
                                current = current[part]
                            metric_value = current
                        except (KeyError, TypeError):
                            metric_value = 0
                    else:
                        # Try to find it at the top level or in production_line
                        if metric in final_state:
                            metric_value = final_state[metric]
                        elif (
                            "production_line" in final_state
                            and metric in final_state["production_line"]
                        ):
                            metric_value = final_state["production_line"][metric]
                        else:
                            metric_value = 0

                    results[metric].append(metric_value)

            # Plot the results for each metric
            for i, metric in enumerate(target_metrics):
                # Format metric name for display
                display_metric = metric.replace("_", " ").title()
                if "." in metric:
                    display_metric = metric.split(".")[-1].replace("_", " ").title()

                # Create line plot
                axes[i].plot(value_range, results[metric], "o-", linewidth=2)
                axes[i].set_xlabel(f"{parameter.replace('_', ' ').title()} Value")
                axes[i].set_ylabel(display_metric)
                axes[i].set_title(f"Effect on {display_metric}")
                axes[i].grid(True, linestyle="--", alpha=0.7)

                # Highlight current value if it's in the range
                current_value = None
                if "." in parameter:
                    parts = parameter.split(".")
                    current = base_state
                    try:
                        for part in parts:
                            current = current[part]
                        current_value = current
                    except (KeyError, TypeError):
                        pass
                else:
                    if parameter in base_state:
                        current_value = base_state[parameter]
                    elif (
                        "production_line" in base_state
                        and parameter in base_state["production_line"]
                    ):
                        current_value = base_state["production_line"][parameter]

                if current_value is not None and isinstance(
                    current_value, (int, float)
                ):
                    current_value_float = float(
                        current_value
                    )  # Explicitly convert to float once

                    if min(value_range) <= current_value_float <= max(value_range):
                        # Find closest value in the range using the explicitly converted value
                        closest_idx = min(
                            range(len(value_range)),
                            key=lambda i: abs(value_range[i] - current_value_float),
                        )

                        axes[i].plot(
                            [value_range[closest_idx]],
                            [results[metric][closest_idx]],
                            "ro",
                            markersize=8,
                        )
                        axes[i].axvline(
                            value_range[closest_idx],
                            color="red",
                            linestyle="--",
                            alpha=0.5,
                        )
                        axes[i].text(
                            value_range[closest_idx],
                            results[metric][closest_idx] * 0.95,
                            f"Current: {current_value_float:.2f}",
                            ha="center",
                            color="red",
                        )

            # Adjust layout
            plt.tight_layout(rect=(0, 0, 1, 0.95))  # Make room for suptitle

            # Add timestamp
            generation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.figtext(
                0.5, 0.01, f"Generated: {generation_time}", ha="center", fontsize=10
            )

            # Save or show visualization
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"Parameter sensitivity analysis saved to {save_path}")
            else:
                # Use results_manager
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"parameter_sensitivity_{parameter.replace('.', '_')}_{timestamp_str}.png"

                # Get temp directory
                temp_dir = results_manager.get_path("temp")
                temp_path = temp_dir / filename
                temp_dir.mkdir(parents=True, exist_ok=True)

                plt.savefig(temp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # Save to visualizations directory using results_manager
                results_manager.save_file(temp_path, "visualizations")

                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

                self.logger.info(
                    f"Parameter sensitivity analysis saved to visualizations directory"
                )

        except Exception as e:
            self.logger.error(f"Error in parameter sensitivity analysis: {str(e)}")
            plt.close()
