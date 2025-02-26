# src/circman5/utils/results_manager.py

import os
from pathlib import Path
import shutil
from datetime import datetime
import logging
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class ResultsManager:
    """Centralized results and path management."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResultsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Get project root
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent

        # Define core paths
        self.paths = {
            # Data directories
            "DATA_DIR": self.project_root / "data",
            "SYNTHETIC_DATA": self.project_root / "data" / "synthetic",
            "PROCESSED_DATA": self.project_root / "data" / "processed",
            "RAW_DATA": self.project_root / "data" / "raw",
            # Results directories
            "RESULTS_BASE": self.project_root / "tests" / "results",
            "RESULTS_ARCHIVE": self.project_root / "tests" / "results" / "archive",
            "RESULTS_RUNS": self.project_root / "tests" / "results" / "runs",
            # Logs
            "LOGS_DIR": self.project_root / "logs",
            "LOGS_ARCHIVE": self.project_root / "logs" / "archive",
        }

        # Create base directories
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # Setup current run
        self._setup_run_directory()

        self._initialized = True

    def _setup_run_directory(self):
        """Create new run directory with standardized structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run = self.paths["RESULTS_RUNS"] / f"run_{timestamp}"

        # Create standard subdirectories
        self.run_dirs = {
            "input_data": self.current_run / "input_data",
            "visualizations": self.current_run / "visualizations",
            "reports": self.current_run / "reports",
            "lca_results": self.current_run / "lca_results",
            "metrics": self.current_run / "metrics",
            "temp": self.current_run / "temp",
            "digital_twin": self.current_run / "digital_twin",
        }

        for dir in self.run_dirs.values():
            dir.mkdir(parents=True, exist_ok=True)

        self._update_latest_symlink()

    def _update_latest_symlink(self):
        """Update 'latest' symlink to current run."""
        latest_link = self.paths["RESULTS_BASE"] / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)

        try:
            os.symlink(self.current_run, latest_link, target_is_directory=True)
        except OSError as e:
            logger.warning(f"Could not create symlink: {e}")

    def get_run_dir(self) -> Path:
        """Get current run directory."""
        return self.current_run

    def get_path(self, key: str) -> Path:
        """Get path by key."""
        if key not in self.paths and key not in self.run_dirs:
            raise KeyError(f"Invalid path key: {key}")

        # First check run_dirs
        if key in self.run_dirs:
            return self.run_dirs[key]

        # Then check paths
        return self.paths[key]

    def save_file(self, file_path: Union[str, Path], target_dir: str) -> Path:
        """Save file to specified target directory."""
        if target_dir not in self.run_dirs:
            raise ValueError(f"Invalid target directory: {target_dir}")

        dest_dir = self.run_dirs[target_dir]
        source = Path(file_path)
        dest = dest_dir / source.name

        # Only copy if source and destination are different paths
        if source != dest and source.exists():
            shutil.copy2(source, dest)
        return dest

    def cleanup_old_runs(self, keep_last: int = 5) -> None:
        """Archive old run directories."""
        runs = sorted(self.paths["RESULTS_RUNS"].glob("run_*"))
        if len(runs) > keep_last:
            for old_run in runs[:-keep_last]:
                archive_path = self.paths["RESULTS_ARCHIVE"] / old_run.name
                shutil.move(str(old_run), str(archive_path))

    def save_to_path(self, file_path: Union[str, Path], target_path_key: str) -> Path:
        """Save file to a specified path from self.paths."""
        if target_path_key not in self.paths:
            raise ValueError(f"Invalid path key: {target_path_key}")

        dest_dir = self.paths[target_path_key]
        source = Path(file_path)
        dest = dest_dir / source.name

        if source.exists():
            shutil.copy2(source, dest)
        return dest


# Create global instance
results_manager = ResultsManager()
