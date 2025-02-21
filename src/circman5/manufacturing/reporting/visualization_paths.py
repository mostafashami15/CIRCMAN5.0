# src/circman5/manufacturing/reporting/visualization_paths.py
"""Visualization path management module."""

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union
from ...utils.logging_config import setup_logger
from ...utils.results_manager import results_manager


class VisualizationPathManager:
    """Manages paths for visualization outputs."""

    def __init__(self):
        self.logger = setup_logger("visualization_path_manager")

    def get_visualization_path(
        self, metric_type: str, filename: Optional[str] = None
    ) -> Path:
        """Get the full path for saving visualizations.

        Args:
            metric_type: Type of metric being visualized
            filename: Optional specific filename to use

        Returns:
            Path: Full path for saving visualization
        """
        if filename is None:
            filename = f"{metric_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        return results_manager.get_path("visualizations") / filename

    def ensure_visualization_directory(
        self, run_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """Ensure visualization directory exists and return its path.

        Args:
            run_dir: Optional path to run directory (deprecated, kept for backwards compatibility)

        Returns:
            Path: Path to visualization directory
        """
        return results_manager.get_path("visualizations")
