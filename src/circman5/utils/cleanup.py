# src/circman5/utils/cleanup.py
import os
import shutil
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def cleanup_test_results(keep_last: int = 5, max_log_age_days: int = 7):
    """
    Clean up old test results and log files.

    Args:
        keep_last: Number of recent test runs to keep
        max_log_age_days: Maximum age of log files in days
    """
    try:
        # Get project root path
        project_root = Path(__file__).resolve().parent.parent.parent.parent

        # Clean up test results
        results_dir = project_root / "tests" / "results"
        runs_dir = results_dir / "runs"
        archive_dir = results_dir / "archive"

        # Ensure archive directory exists
        archive_dir.mkdir(parents=True, exist_ok=True)

        if runs_dir.exists():
            # Get all run directories sorted by name (which includes timestamp)
            runs = sorted(list(runs_dir.glob("run_*")))

            # Move old runs to archive
            if len(runs) > keep_last:
                for old_run in runs[:-keep_last]:
                    archive_path = archive_dir / old_run.name
                    if old_run.exists():
                        shutil.move(str(old_run), str(archive_path))
                        logger.info(f"Archived old run: {old_run.name}")

        # Clean up logs
        logs_dir = project_root / "logs"
        logs_archive_dir = logs_dir / "archive"
        logs_archive_dir.mkdir(parents=True, exist_ok=True)

        if logs_dir.exists():
            current_time = datetime.now()
            for log_file in logs_dir.glob("*.log"):
                file_age = current_time - datetime.fromtimestamp(
                    log_file.stat().st_mtime
                )
                if file_age.days > max_log_age_days:
                    archive_path = logs_archive_dir / log_file.name
                    shutil.move(str(log_file), str(archive_path))
                    logger.info(f"Archived old log: {log_file.name}")

        logger.info("Cleanup completed successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise


def cleanup_old_runs(runs_dir: Path, keep_last: int = 5) -> None:
    """
    Clean up old test runs, keeping only the specified number of most recent runs.

    Args:
        runs_dir: Directory containing test runs
        keep_last: Number of most recent runs to keep
    """
    if not runs_dir.exists():
        return

    runs = sorted(list(runs_dir.glob("run_*")))
    if len(runs) > keep_last:
        for old_run in runs[:-keep_last]:
            try:
                if old_run.exists():
                    shutil.rmtree(old_run)
            except Exception as e:
                print(f"Warning: Could not remove old run {old_run}: {e}")
