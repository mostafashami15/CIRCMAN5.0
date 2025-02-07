from pathlib import Path
import os
import shutil
from datetime import datetime


def get_run_directory():
    """
    Create a new timestamped run directory inside tests/results/runs and update the 'latest' symlink.
    Returns:
        Path to the current run directory.
    """
    project_root = Path(__file__).resolve().parent.parent
    results_base = project_root / "tests" / "results"
    runs_dir = results_base / "runs"

    # Ensure runs directory exists
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create a timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run = runs_dir / f"run_{timestamp}"
    current_run.mkdir()

    # Create required subdirectories
    (current_run / "input_data").mkdir(parents=True, exist_ok=True)
    (current_run / "visualizations").mkdir(parents=True, exist_ok=True)
    (current_run / "reports").mkdir(parents=True, exist_ok=True)

    # Update 'latest' symlink
    latest_link = results_base / "latest"
    if latest_link.exists():
        if latest_link.is_symlink():
            latest_link.unlink()
        else:
            shutil.rmtree(latest_link)
    os.symlink(current_run, latest_link)

    return current_run
