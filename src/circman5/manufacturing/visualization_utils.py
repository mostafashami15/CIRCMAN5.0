# src/circman5/manufacturing/visualization_utils.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from pathlib import Path
from typing import Optional
from ..utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService


class VisualizationConfig:
    """Shared configuration for all visualization modules."""

    # Initialize with constants service
    _constants = ConstantsService()
    _viz_config = _constants.get_visualization_config()

    DEFAULT_STYLE = _viz_config.get(
        "DEFAULT_STYLE",
        {
            "figure.figsize": (12, 8),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        },
    )

    # Get color palette from config
    COLOR_PALETTE = sns.color_palette(
        _viz_config.get("COLOR_PALETTE", "husl"),
        _viz_config.get("COLOR_PALETTE_SIZE", 8),
    )

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
        try:
            from circman5.adapters.services.constants_service import ConstantsService

            constants = ConstantsService()
            viz_config = constants.get_visualization_config()
        except (ImportError, ValueError):
            # Default configuration if service is not available
            viz_config = {
                "DEFAULT_STYLE": {
                    "figure.figsize": [12, 8],
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "axes.grid": True,
                    "grid.linestyle": "--",
                    "grid.alpha": 0.7,
                },
                "COLOR_PALETTE": "husl",
                "COLOR_PALETTE_SIZE": 8,
                "DEFAULT_DPI": 300,
            }

        if save_path is None:
            viz_dir = results_manager.get_path("visualizations")
            save_path = str(viz_dir / filename)

        fig.savefig(
            save_path, dpi=viz_config.get("DEFAULT_DPI", 300), bbox_inches="tight"
        )
        return save_path

    @staticmethod
    def get_visualization_path(filename: str, subdir: str = "visualizations") -> Path:
        """Get standardized path for visualizations."""
        return results_manager.get_path("visualizations") / filename
