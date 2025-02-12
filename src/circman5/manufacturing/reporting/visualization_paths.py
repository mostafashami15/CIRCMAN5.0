# src/circman5/manufacturing/reporting/visualization_paths.py

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union
from ...config.project_paths import project_paths
from ...utils.logging_config import setup_logger


class VisualizationPathManager:
    """Manages paths for visualization outputs."""

    def __init__(self):
        self.logger = setup_logger("visualization_path_manager")
        self.base_dir = project_paths.get_path("RESULTS_RUNS")

    def get_visualization_path(
        self, metric_type: str, filename: Optional[str] = None
    ) -> Path:
        """Get the full path for saving visualizations."""
        # Get current run directory
        run_dir = project_paths.get_run_directory()
        viz_dir = run_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{metric_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        return viz_dir / filename

    def ensure_visualization_directory(
        self, run_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Ensure visualization directory exists and return its path.

        Args:
            run_dir: Optional path to run directory. If None, creates new run directory.

        Returns:
            Path: Path to visualization directory
        """
        if run_dir is None:
            run_dir = project_paths.get_run_directory()
        else:
            run_dir = Path(run_dir) if isinstance(run_dir, str) else run_dir

        viz_dir = run_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        return viz_dir
