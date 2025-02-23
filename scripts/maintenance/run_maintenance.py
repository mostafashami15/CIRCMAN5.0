# scripts/maintenance/run_maintenance.py

from pathlib import Path
import sys
from backup.backup_project import create_backup
from maintenance.log_cleanup import cleanup_logs
from circman5.utils.logging_config import setup_logger


def run_maintenance():
    logger = setup_logger("maintenance")

    try:
        # Run backup
        logger.info("Starting project backup...")
        backup_path = create_backup()
        logger.info(f"Backup completed at: {backup_path}")

        # Run log cleanup
        logger.info("Starting log cleanup...")
        files_archived = cleanup_logs()
        logger.info(f"Log cleanup completed. {files_archived} files archived.")

    except Exception as e:
        logger.error(f"Maintenance failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_maintenance()
