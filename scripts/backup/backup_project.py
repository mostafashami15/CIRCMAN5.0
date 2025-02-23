# scripts/backup/backup_project.py

from pathlib import Path
from datetime import datetime
import shutil
import os
from circman5.utils.results_manager import results_manager
from circman5.utils.logging_config import setup_logger


def create_backup():
    logger = setup_logger("backup_manager")

    try:
        # Get project root using existing utility
        project_root = results_manager.project_root

        # Create backup name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"CIRCMAN5.0_backup_{timestamp}"

        # Create backup directory in parent directory
        backup_dir = project_root.parent / backup_name

        # Define exclusion patterns
        exclude = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "tests/results",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
        }

        def ignore_patterns(path, names):
            return {
                n
                for n in names
                if n in exclude or any(n.endswith(ext) for ext in [".pyc", ".pyo"])
            }

        # Create backup
        shutil.copytree(project_root, backup_dir, ignore=ignore_patterns)

        # Create backup info file
        with open(backup_dir / "backup_info.txt", "w") as f:
            f.write(f"Backup created: {datetime.now()}\n")
            f.write(f"Original path: {project_root}\n")
            f.write(f"Backup path: {backup_dir}\n")

        logger.info(f"Backup successfully created at: {backup_dir}")
        return str(backup_dir)

    except Exception as e:
        logger.error(f"Backup creation failed: {str(e)}")
        raise


if __name__ == "__main__":
    create_backup()
