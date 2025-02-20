"""Cleanup utility for managing test results and logs."""

import os
import shutil
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional
from ..utils.results_manager import results_manager

logger = logging.getLogger(__name__)


def cleanup_test_results(keep_last: int = 5, max_log_age_days: int = 7) -> None:
    """Clean up old test results and log files.

    Args:
        keep_last: Number of recent test runs to keep
        max_log_age_days: Maximum age of log files in days
    """
    try:
        # Get directories from results_manager
        archive_dir = results_manager.get_path("RESULTS_ARCHIVE")
        runs_dir = results_manager.get_path("RESULTS_RUNS")
        logs_dir = results_manager.get_path("LOGS_DIR")
        logs_archive = results_manager.get_path("LOGS_ARCHIVE")

        # Clean up test results
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
        if logs_dir.exists():
            current_time = datetime.now()
            for log_file in logs_dir.glob("*.log"):
                file_age = current_time - datetime.fromtimestamp(
                    log_file.stat().st_mtime
                )
                if file_age.days > max_log_age_days:
                    archive_path = logs_archive / log_file.name
                    shutil.move(str(log_file), str(archive_path))
                    logger.info(f"Archived old log: {log_file.name}")

        logger.info("Cleanup completed successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise


def cleanup_old_runs(runs_dir: Optional[Path] = None, keep_last: int = 5) -> None:
    """Clean up old test runs, keeping only the specified number of most recent runs.

    Args:
        runs_dir: Optional directory containing test runs. If None, uses results_manager
        keep_last: Number of most recent runs to keep
    """
    try:
        if runs_dir is None:
            runs_dir = results_manager.get_path("RESULTS_RUNS")

        if not runs_dir.exists():
            return

        runs = sorted(list(runs_dir.glob("run_*")))
        if len(runs) > keep_last:
            archive_dir = results_manager.get_path("RESULTS_ARCHIVE")
            for old_run in runs[:-keep_last]:
                try:
                    if old_run.exists():
                        archive_path = archive_dir / old_run.name
                        shutil.move(str(old_run), str(archive_path))
                        logger.info(f"Archived old run: {old_run.name}")
                except Exception as e:
                    logger.warning(f"Could not archive old run {old_run}: {e}")

    except Exception as e:
        logger.error(f"Error during runs cleanup: {e}")
        raise
