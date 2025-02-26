# src/circman5/manufacturing/digital_twin/visualization/process_visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from matplotlib.patches import Circle, Arc
from matplotlib.lines import Line2D
from pathlib import Path
import datetime
import json

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from ..core.state_manager import StateManager
from circman5.adapters.services.constants_service import ConstantsService


class ProcessVisualizer:
    """
    Create visualizations specific to manufacturing processes.

    This class provides tailored visualizations for different manufacturing
    process stages like wafer cutting, cell production, and module assembly.
    """

    def __init__(self, state_manager: StateManager):
        """Initialize the process visualizer."""
        self.state_manager = state_manager
        self.logger = setup_logger("process_visualizer")
        self.constants = ConstantsService()
        self.config = self.constants.get_digital_twin_config()

        # Get manufacturing configuration for process stages
        self.manufacturing_config = self.constants.get_manufacturing_constants()
        self.manufacturing_stages = self.manufacturing_config.get(
            "MANUFACTURING_STAGES", {}
        )

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

        self.logger.info("ProcessVisualizer initialized")

    def visualize_manufacturing_stages(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create a visualization showing all manufacturing stages with their metrics.

        Args:
            save_path: Optional path to save the visualization
        """
        fig = None  # Initialize fig to None at the start
        error_message = ""  # Initialize error message

        try:
            # Get current state and history
            current_state = self.state_manager.get_current_state()
            history = self.state_manager.get_history(limit=20)

            if not current_state:
                self.logger.warning("No state data available for process visualization")
                return

            # Create figure with subplots per manufacturing stage
            num_stages = len(self.manufacturing_stages)
            if num_stages == 0:
                # Fallback if no stages are defined in configuration
                stages = ["silicon_purification", "wafer_production", "cell_production"]
                num_stages = len(stages)
            else:
                stages = list(self.manufacturing_stages.keys())

            # Create figure with stages as rows and metrics as columns
            fig, axes = plt.subplots(num_stages, 3, figsize=(16, 4 * num_stages))
            fig.suptitle("Manufacturing Process Stages Performance", fontsize=16)

            # If only one stage, axes won't be 2D, make it 2D
            if num_stages == 1:
                axes = [axes]

            # Extract metrics for each stage
            for i, stage in enumerate(stages):
                stage_data = self._extract_stage_data(stage, current_state, history)

                # Plot yield rate (column 1)
                self._plot_stage_yield(axes[i][0], stage, stage_data)

                # Plot energy efficiency (column 2)
                self._plot_stage_energy(axes[i][1], stage, stage_data)

                # Plot quality metrics (column 3)
                self._plot_stage_quality(axes[i][2], stage, stage_data)

            plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for title

        except Exception as e:
            self.logger.error(f"Error creating process visualization: {str(e)}")
            error_message = str(e)  # Save the error message
            plt.close() if "plt" in locals() else None  # Close if plt exists

        # This try block handles saving the figure, regardless of whether the main visualization succeeded
        try:
            # If fig is None or we encountered an error, create a simple error figure
            if fig is None:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(
                    0.5,
                    0.5,
                    f"Error creating visualization: {error_message}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Process Visualization Error")
                ax.axis("off")

            # Save or show visualization
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.logger.info(f"Process visualization saved to {save_path}")
            else:
                # Use results_manager
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"manufacturing_processes_{timestamp_str}.png"

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
                    f"Process visualization saved to visualizations directory"
                )

        except Exception as nested_e:
            self.logger.error(f"Error saving process visualization: {str(nested_e)}")
            plt.close() if "plt" in locals() else None  # Close if plt exists

    def _extract_stage_data(
        self, stage: str, state: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract metrics specific to a manufacturing stage from state and history.

        Args:
            stage: Manufacturing stage name
            state: Current state dictionary
            history: State history

        Returns:
            Dict with stage-specific metrics
        """
        # Initialize with default values
        stage_data = {
            "yield_rate": 90.0,  # Default yield rate
            "energy_efficiency": 0.7,  # Default energy efficiency
            "defect_rate": 5.0,  # Default defect rate
            "cycle_time": 60.0,  # Default cycle time
            "production_rate": 100.0,  # Default production rate
            "historical_yield": [],
            "historical_energy": [],
            "historical_defect": [],
            "timestamps": [],
        }

        # Check if process data exists in state
        if (
            "manufacturing_processes" in state
            and stage in state["manufacturing_processes"]
        ):
            process_data = state["manufacturing_processes"][stage]
            stage_data.update(process_data)

        # Process historical data
        for h_state in history:
            if "timestamp" in h_state:
                try:
                    timestamp = datetime.datetime.fromisoformat(h_state["timestamp"])
                    stage_data["timestamps"].append(timestamp)

                    # Extract historical metrics if available
                    if (
                        "manufacturing_processes" in h_state
                        and stage in h_state["manufacturing_processes"]
                    ):
                        h_process = h_state["manufacturing_processes"][stage]
                        stage_data["historical_yield"].append(
                            h_process.get("yield_rate", 90.0)
                        )
                        stage_data["historical_energy"].append(
                            h_process.get("energy_efficiency", 0.7)
                        )
                        stage_data["historical_defect"].append(
                            h_process.get("defect_rate", 5.0)
                        )
                    else:
                        # Use placeholder values if no data
                        stage_data["historical_yield"].append(90.0)
                        stage_data["historical_energy"].append(0.7)
                        stage_data["historical_defect"].append(5.0)
                except (ValueError, TypeError):
                    continue

        return stage_data

    def _plot_stage_yield(
        self, ax, stage_name: str, stage_data: Dict[str, Any]
    ) -> None:
        """Plot yield rate metrics for a manufacturing stage."""
        # Format stage name for display
        display_name = stage_name.replace("_", " ").title()

        # Create yield gauge chart
        yield_rate = stage_data.get("yield_rate", 90.0)
        self._create_gauge_chart(
            ax,
            value=yield_rate,
            title=f"{display_name}\nYield Rate",
            min_val=0,
            max_val=100,
            units="%",
            colors=["red", "yellow", "green"],  # Explicit list instead of None
            thresholds=[70, 85],  # Explicit list instead of None
        )

        # Add expected yield for comparison if available
        if stage_name in self.manufacturing_stages:
            expected_yield = (
                self.manufacturing_stages[stage_name].get("expected_yield", 0.0) * 100
            )
            if expected_yield > 0:
                ax.axhline(expected_yield, color="blue", linestyle="--", alpha=0.7)
                ax.text(
                    0.5,
                    expected_yield + 2,
                    f"Expected: {expected_yield:.1f}%",
                    ha="center",
                    color="blue",
                )

    def _plot_stage_energy(
        self, ax, stage_name: str, stage_data: Dict[str, Any]
    ) -> None:
        """Plot energy efficiency metrics for a manufacturing stage."""
        # Format stage name for display
        display_name = stage_name.replace("_", " ").title()

        # Create energy efficiency gauge chart
        energy_efficiency = stage_data.get("energy_efficiency", 0.7)
        self._create_gauge_chart(
            ax,
            value=energy_efficiency,
            title=f"{display_name}\nEnergy Efficiency",
            min_val=0,
            max_val=1.0,
            units="ratio",
            colors=["red", "yellow", "green"],
            thresholds=[0.4, 0.6],
        )

    def _plot_stage_quality(
        self, ax, stage_name: str, stage_data: Dict[str, Any]
    ) -> None:
        """Plot quality metrics for a manufacturing stage."""
        # Format stage name for display
        display_name = stage_name.replace("_", " ").title()

        # Defect rate (inverse of quality)
        defect_rate = stage_data.get("defect_rate", 5.0)
        quality_score = 100 - defect_rate  # Convert to quality score

        # Create quality gauge chart
        self._create_gauge_chart(
            ax,
            value=quality_score,
            title=f"{display_name}\nQuality Score",
            min_val=0,
            max_val=100,
            units="%",
            colors=["red", "yellow", "green"],
            thresholds=[80, 90],
        )

    def _create_gauge_chart(
        self,
        ax,
        value: float,
        title: str,
        min_val: float,
        max_val: float,
        units: str = "",
        colors: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
    ) -> None:
        """
        Create a gauge chart visualization.

        Args:
            ax: Matplotlib axis to plot on
            value: Value to display
            title: Chart title
            min_val: Minimum scale value
            max_val: Maximum scale value
            units: Units label
            colors: List of colors for different value ranges
            thresholds: List of threshold values for color changes
        """
        from matplotlib.patches import Arc, Circle
        from matplotlib.lines import Line2D

        # Default colors and thresholds
        if colors is None:
            colors = ["red", "yellow", "green"]
        if thresholds is None:
            range_size = max_val - min_val
            thresholds = [min_val + range_size * 0.33, min_val + range_size * 0.66]

        # Clear axis
        ax.clear()

        # Draw the gauge background as an arc
        radius = 0.8  # Size of the gauge
        center = (0.5, 0)  # Center at bottom middle

        # Ensure value is within range
        value = max(min_val, min(max_val, value))

        # Calculate angle for the value (180° to 0° mapping to min_val to max_val)
        angle = 180 * (1 - (value - min_val) / (max_val - min_val))

        # Draw colored arc segments
        if len(thresholds) + 1 == len(colors):
            # Create segments based on thresholds
            prev_angle = 180
            prev_threshold = min_val

            for i, threshold in enumerate(thresholds):
                # Calculate angle for this threshold
                threshold_angle = 180 * (
                    1 - (threshold - min_val) / (max_val - min_val)
                )

                # Draw arc segment
                arc = Arc(
                    center,
                    2 * radius,
                    2 * radius,
                    theta1=threshold_angle,
                    theta2=prev_angle,
                    lw=20,
                    color=colors[i],
                    zorder=0,
                )
                ax.add_patch(arc)

                prev_angle = threshold_angle
                prev_threshold = threshold

            # Draw final segment
            arc = Arc(
                center,
                2 * radius,
                2 * radius,
                theta1=0,
                theta2=prev_angle,
                lw=20,
                color=colors[-1],
                zorder=0,
            )
            ax.add_patch(arc)

        # Draw needle
        # Convert angle from degrees to radians
        angle_rad = np.deg2rad(angle)

        # Calculate needle endpoint
        x_needle = center[0] + radius * np.cos(angle_rad)
        y_needle = center[1] + radius * np.sin(angle_rad)

        # Draw needle line
        ax.add_line(
            Line2D(
                [center[0], x_needle],
                [center[1], y_needle],
                color="black",
                linewidth=2,
                zorder=2,
            )
        )

        # Add circular pivot at needle base
        pivot = Circle(center, radius=0.05, color="black", zorder=3)
        ax.add_patch(pivot)

        # Add value text
        ax.text(
            center[0],
            center[1] - 0.3,
            f"{value:.1f}{units}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        # Set title
        ax.set_title(title, fontsize=12)

        # Set equal aspect ratio and remove frame
        ax.set_aspect("equal")

        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Set limits
        ax.set_xlim(center[0] - radius - 0.1, center[0] + radius + 0.1)
        ax.set_ylim(center[1] - 0.1, center[1] + radius + 0.1)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
