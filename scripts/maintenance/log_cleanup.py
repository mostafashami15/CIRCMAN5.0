# scripts/maintenance/log_cleanup.py

from pathlib import Path
from datetime import datetime, timedelta
import shutil
import os
import re
from circman5.utils.logging_config import setup_logger
from circman5.utils.results_manager import results_manager


def cleanup_logs():
    logger = setup_logger("log_cleanup")

    try:
        # Get logs directory
        logs_dir = results_manager.project_root / "logs"
        archive_dir = logs_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Get current time
        now = datetime.now()

        # Group logs by date using improved pattern matching
        log_groups = {}
        unmatched_logs = []

        for log_file in logs_dir.glob("*"):
            if not log_file.is_file() or log_file.name == "cleanup_report.txt":
                continue

            try:
                # Try different patterns to match dates in filenames
                patterns = [
                    r".*_(\d{8})_.*\.log$",  # Standard format YYYYMMDD
                    r".*_(\d{6})\.l.*$",  # Alternative format YYMMDD
                ]

                date_str = None
                for pattern in patterns:
                    match = re.match(pattern, log_file.name)
                    if match:
                        date_str = match.group(1)
                        # Ensure we have full year format
                        if len(date_str) == 6:  # YYMMDD format
                            date_str = "20" + date_str  # Assume 20xx for year
                        break

                if date_str:
                    log_groups.setdefault(date_str, []).append(log_file)
                else:
                    unmatched_logs.append(log_file)

            except Exception as e:
                logger.warning(f"Could not process filename {log_file.name}: {e}")
                unmatched_logs.append(log_file)

        # Archive old logs (keep last 7 days)
        total_archived = 0
        total_size_saved = 0

        # Process dated logs
        for date_str, files in log_groups.items():
            try:
                log_date = datetime.strptime(date_str, "%Y%m%d")
                if now - log_date > timedelta(days=7):
                    # Create archive directory for this date
                    date_archive = archive_dir / date_str
                    date_archive.mkdir(exist_ok=True)

                    # Move files to archive
                    for file in files:
                        file_size = file.stat().st_size
                        shutil.move(str(file), str(date_archive / file.name))
                        total_archived += 1
                        total_size_saved += file_size

                    logger.info(f"Archived {len(files)} logs from {date_str}")
            except Exception as e:
                logger.error(f"Error processing logs from {date_str}: {e}")
                continue

        # Process unmatched logs (put them in a special archive folder)
        if unmatched_logs:
            unknown_archive = archive_dir / "unknown_date"
            unknown_archive.mkdir(exist_ok=True)

            for file in unmatched_logs:
                try:
                    file_size = file.stat().st_size
                    shutil.move(str(file), str(unknown_archive / file.name))
                    total_archived += 1
                    total_size_saved += file_size
                except Exception as e:
                    logger.error(f"Error archiving unmatched log {file.name}: {e}")

        # Create cleanup report
        report_path = logs_dir / "cleanup_report.txt"
        with open(report_path, "a") as f:
            f.write(f"\nLog Cleanup Report - {datetime.now()}\n")
            f.write(f"Total files archived: {total_archived}\n")
            f.write(f"Total space saved: {total_size_saved / (1024*1024):.2f} MB\n")
            f.write(f"Unmatched logs processed: {len(unmatched_logs)}\n")
            f.write("-" * 50 + "\n")

        logger.info(f"Log cleanup completed. Total files archived: {total_archived}")
        logger.info(
            f"Unmatched logs moved to archive/unknown_date: {len(unmatched_logs)}"
        )
        return total_archived

    except Exception as e:
        logger.error(f"Log cleanup failed: {str(e)}")
        raise


if __name__ == "__main__":
    cleanup_logs()
