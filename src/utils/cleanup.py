import os
import shutil
from datetime import datetime
from pathlib import Path


def cleanup_test_results():
    # Get absolute project root path
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    results_dir = project_root / "tests" / "results"

    runs_dir = results_dir / "runs"
    if runs_dir.exists():
        runs = sorted(list(runs_dir.glob("run_*")))
        # Keep only last 5 runs
        for old_run in runs[:-5]:
            if old_run.exists():
                shutil.rmtree(old_run)

    logs_dir = project_root / "logs"
    archive_dir = logs_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Archive old logs
    for log in logs_dir.glob("*.log"):
        if (datetime.now() - datetime.fromtimestamp(log.stat().st_mtime)).days > 7:
            shutil.move(str(log), str(archive_dir / log.name))

    print("Cleanup completed")
