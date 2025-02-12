from pathlib import Path
import os
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_run_directory() -> Path:
    """
    Create a new timestamped run directory inside tests/results/runs and update the 'latest' symlink.

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
        # Get absolute path to project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        logger.debug(f"Project root directory: {project_root}")

        # Setup base directory structure
        results_base = project_root / "tests" / "results"
        results_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results base directory: {results_base}")

        # Ensure required subdirectories exist
        archive_dir = results_base / "archive"
        archive_dir.mkdir(exist_ok=True)

        runs_dir = results_base / "runs"
        runs_dir.mkdir(exist_ok=True)
        logger.debug("Created base directory structure")

        # Create new timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_run = runs_dir / f"run_{timestamp}"
        current_run.mkdir(exist_ok=True)
        logger.info(f"Created new run directory: {current_run}")

        # Create standard subdirectories
        subdirs = ["input_data", "visualizations", "reports"]
        for subdir in subdirs:
            subdir_path = current_run / subdir
            subdir_path.mkdir(exist_ok=True)
            logger.debug(f"Created subdirectory: {subdir_path}")

        # Manage 'latest' symlink
        latest_link = results_base / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.is_dir():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
            logger.debug("Removed existing 'latest' link")

        # Create new symlink
        try:
            os.symlink(current_run, latest_link, target_is_directory=True)
            logger.info(f"Created 'latest' symlink: {latest_link} -> {current_run}")
        except OSError as e:
            logger.warning(f"Could not create symlink: {e}")

        # Archive old runs (keep last 5)
        old_runs = sorted(list(runs_dir.glob("run_*")))[:-5]
        for old_run in old_runs:
            archive_path = archive_dir / old_run.name
            shutil.move(str(old_run), str(archive_path))
            logger.debug(f"Archived old run: {old_run.name}")

        # Verify directory structure
        logger.info("Directory structure created successfully")
        logger.debug(f"Run directory contents: {list(current_run.iterdir())}")

        return current_run

    except Exception as e:
        logger.error(f"Error creating run directory: {e}")
        raise RuntimeError(f"Failed to create test run directory: {e}")
