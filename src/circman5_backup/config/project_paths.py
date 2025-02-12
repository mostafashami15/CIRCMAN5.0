"""Project paths configuration and management."""

import os
from pathlib import Path
from datetime import datetime
import shutil
from typing import Dict, Optional


class ProjectPaths:
    """Manages project paths and directory structures."""

    def __init__(self):
        # Get absolute project root
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

        # Define core directory structure
        self.paths: Dict[str, Path] = {
            # Data directories
            "DATA_DIR": self.PROJECT_ROOT / "data",
            "SYNTHETIC_DATA": self.PROJECT_ROOT / "data" / "synthetic",
            "PROCESSED_DATA": self.PROJECT_ROOT / "data" / "processed",
            "RAW_DATA": self.PROJECT_ROOT / "data" / "raw",
            # Test results
            "RESULTS_DIR": self.PROJECT_ROOT / "tests" / "results",
            "RESULTS_ARCHIVE": self.PROJECT_ROOT / "tests" / "results" / "archive",
            "RESULTS_LATEST": self.PROJECT_ROOT / "tests" / "results" / "latest",
            "RESULTS_RUNS": self.PROJECT_ROOT / "tests" / "results" / "runs",
            # Logs
            "LOGS_DIR": self.PROJECT_ROOT / "logs",
            "LOGS_ARCHIVE": self.PROJECT_ROOT / "logs" / "archive",
        }

        # Create directory structure
        self.create_directories()

    def create_directories(self) -> None:
        """Create all necessary directories."""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_run_directory(self) -> Path:
        """
        Create and return a new timestamped run directory.
        Updates 'latest' symlink to point to newest run.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_run = self.paths["RESULTS_RUNS"] / f"run_{timestamp}"

        # Create run directory structure
        current_run.mkdir(parents=True, exist_ok=True)
        (current_run / "input_data").mkdir(exist_ok=True)
        (current_run / "visualizations").mkdir(exist_ok=True)
        (current_run / "reports").mkdir(exist_ok=True)

        # Update 'latest' symlink
        latest_link = self.paths["RESULTS_LATEST"]
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)
        os.symlink(current_run, latest_link)

        return current_run

    def get_path(self, key: str) -> str:
        """Get path as string by key."""
        if key not in self.paths:
            raise KeyError(
                f"Path key '{key}' not found. Available keys: {list(self.paths.keys())}"
            )
        return str(self.paths[key])

    def cleanup_old_runs(self, keep_last: int = 5) -> None:
        """
        Clean up old run directories, keeping only the specified number of most recent runs.

        Args:
            keep_last: Number of most recent runs to keep
        """
        runs = sorted(self.paths["RESULTS_RUNS"].glob("run_*"))
        for old_run in runs[:-keep_last]:
            if old_run.exists():
                shutil.rmtree(old_run)

    def archive_old_logs(self, days: int = 7) -> None:
        """
        Archive logs older than specified number of days.

        Args:
            days: Age in days after which logs should be archived
        """
        for log in self.paths["LOGS_DIR"].glob("*.log"):
            if (
                datetime.now() - datetime.fromtimestamp(log.stat().st_mtime)
            ).days > days:
                shutil.move(str(log), str(self.paths["LOGS_ARCHIVE"] / log.name))

    def get_synthetic_data_path(self, filename: str) -> str:
        """Get path for synthetic data files."""
        return str(self.paths["SYNTHETIC_DATA"] / filename)


# Create global instance
project_paths = ProjectPaths()
