import os
import shutil
from datetime import datetime
from pathlib import Path
import sys

# Set the project root to two levels up (assuming this file is in CIRCMAN5.0/scripts/)
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def cleanup_main_root():
    """Clean up irrelevant output files in the main project root."""
    # Define the file types to clean up
    irrelevant_files = [".png", ".xlsx", ".csv", "test_log.txt"]

    for file in project_root.glob("*"):
        if file.suffix in irrelevant_files and file.is_file():
            file.unlink()  # Delete the file
            print(f"Removed: {file}")

    print("Main root directory cleaned up.")


def cleanup_test_results():
    """Clean up and reorganize test results directory."""
    # Use the global project_root, not a local one.
    results_dir = project_root / "tests" / "results"
    print(f"Cleaning up test results in: {results_dir}")

    # Remove the 'latest' symlink if it exists
    latest_link = results_dir / "latest"
    if latest_link.exists():
        if latest_link.is_symlink():
            latest_link.unlink()
        else:
            shutil.rmtree(latest_link)

    # Define base directories for test results
    base_dirs = {
        "latest": results_dir / "latest",
        "runs": results_dir / "runs",
        "archive": results_dir / "archive",
    }

    # Clear and recreate base directories
    for dir_path in base_dirs.values():
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)
        print(f"Created directory: {dir_path}")

    # Function to organize a single run directory
    def organize_run_directory(run_path):
        """Reorganizes and structures a test run directory properly."""
        parent_dir = run_path.parent  # Parent directory where runs are stored
        temp_run_path = parent_dir / f"temp_{run_path.name}"
        temp_run_path.mkdir()

        # Create subdirectories
        (temp_run_path / "input_data").mkdir()
        (temp_run_path / "reports").mkdir()
        (temp_run_path / "visualizations").mkdir()

        # Move files into the correct locations
        for root, _, files in os.walk(run_path):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                if file.endswith(".csv"):
                    shutil.move(
                        str(file_path), str(temp_run_path / "input_data" / file)
                    )
                elif file.endswith(".png"):
                    shutil.move(
                        str(file_path), str(temp_run_path / "visualizations" / file)
                    )
                elif file.endswith(".xlsx"):
                    shutil.move(str(file_path), str(temp_run_path / "reports" / file))
                elif file == "test_log.txt":
                    shutil.move(str(file_path), str(temp_run_path / file))

        # Remove the old directory and replace it with the organized structure
        shutil.rmtree(run_path)
        shutil.move(str(temp_run_path), str(run_path))
        print(f"Organized run directory: {run_path.name}")

    # Process each run directory under "runs"
    run_dirs = list(base_dirs["runs"].glob("run_*"))
    for run_dir in run_dirs:
        organize_run_directory(run_dir)

    # Update the 'latest' symlink to point to the most recent run directory
    if run_dirs:
        latest_run = max(run_dirs, key=lambda x: x.name)
        os.symlink(latest_run, base_dirs["latest"])
        print(f"Updated latest symlink to: {latest_run.name}")

    print("\nCleanup Summary:")
    print(f"Organized {len(run_dirs)} run directories")
    print("Directory structure has been corrected and reorganized.")


if __name__ == "__main__":
    try:
        # Clean up irrelevant files in the main project root
        cleanup_main_root()

        # Clean up and reorganize the test results directory
        cleanup_test_results()
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
