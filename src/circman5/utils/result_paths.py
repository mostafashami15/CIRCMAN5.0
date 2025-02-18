# src/circman5/utils/result_paths.py
from pathlib import Path
import os
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_run_directory() -> Path:
    """
    Create a new timestamped run directory inside tests/results/runs.

    Returns:
        Path: Directory path for the current test run

    Directory Structure:
        tests/
        └── results/
            ├── archive/           # For archived test results
            ├── latest/ -> runs/latest_run/  # Symlink to current run
            └── runs/
                └── run_TIMESTAMP/ # e.g. run_20250212_134527/
                    ├── input_data/
                    ├── visualizations/
                    └── reports/

    """
    try:
        # Get project root (4 levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent.parent

        # Setup directory structure
        results_base = project_root / "tests" / "results"
        runs_dir = results_base / "runs"
        archive_dir = results_base / "archive"

        # Create directories
        for dir_path in [results_base, runs_dir, archive_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create new timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_run = runs_dir / f"run_{timestamp}"

        # Create standard subdirectories
        subdirs = ["input_data", "visualizations", "reports", "lca_results"]
        for subdir in subdirs:
            (current_run / subdir).mkdir(parents=True, exist_ok=True)

        # Update 'latest' symlink
        latest_link = results_base / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)

        try:
            os.symlink(current_run, latest_link, target_is_directory=True)
        except OSError as e:
            logger.warning(f"Could not create symlink: {e}")

        return current_run

    except Exception as e:
        logger.error(f"Error creating run directory: {e}")
        raise RuntimeError(f"Failed to create test run directory: {e}")
