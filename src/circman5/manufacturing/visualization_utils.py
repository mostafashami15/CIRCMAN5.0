# src/circman5/manufacturing/visualization_utils.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from pathlib import Path
from typing import Optional
from ..utils.results_manager import results_manager


class VisualizationConfig:
    """Shared configuration for all visualization modules."""

    DEFAULT_STYLE = {
        "figure.figsize": (12, 8),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    }

    COLOR_PALETTE = sns.color_palette("husl", 8)

    @classmethod
    def setup_style(cls):
        """Apply consistent style across all visualizations."""
        plt.style.use("default")
        sns.set_theme(style="whitegrid")

        for key, value in cls.DEFAULT_STYLE.items():
            plt.rcParams[key] = value

    @staticmethod
    def save_figure(
        fig: Figure,
        filename: str,
        save_path: Optional[str] = None,
        subdir: str = "visualizations",
    ) -> str:
        """
        Save figure to standardized location.

        Args:
            fig: matplotlib figure to save
            filename: name of the file
            save_path: optional explicit save path
            subdir: subdirectory under run directory for organization

        Returns:
            str: Path where figure was saved
        """
        if save_path is None:
            viz_dir = results_manager.get_path("visualizations")
            save_path = str(viz_dir / filename)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return save_path

    @staticmethod
    def get_visualization_path(filename: str, subdir: str = "visualizations") -> Path:
        """Get standardized path for visualizations."""
        return results_manager.get_path("visualizations") / filename
