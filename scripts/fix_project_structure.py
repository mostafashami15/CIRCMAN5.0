# scripts/fix_project_structure.py
import os
import shutil
from datetime import datetime
from pathlib import Path
import sys
import re

# Set the project root to two levels up (assuming this file is in CIRCMAN5.0/scripts/)
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def cleanup_old_structure():
    """Remove outdated directories and fix project structure."""
    base_dir = src_path / "circman5"

    # Directories that should be removed (they've been moved)
    dirs_to_remove = [
        "analysis",  # Moved to manufacturing/lifecycle
        "visualization",  # Should be under manufacturing
    ]

    for dir_name in dirs_to_remove:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"Removing outdated directory: {dir_path}")
            shutil.rmtree(dir_path)


def fix_imports():
    """Fix import statements throughout the project."""
    patterns_to_fix = [
        # Fix analysis imports to point to new location
        (r"from circman5\.analysis\.lca", r"from circman5.manufacturing.lifecycle"),
        (
            r"from circman5\.analysis\.efficiency",
            r"from circman5.manufacturing.analyzers.efficiency",
        ),
        (
            r"from circman5\.analysis\.quality",
            r"from circman5.manufacturing.analyzers.quality",
        ),
        (
            r"from circman5\.analysis\.sustainability",
            r"from circman5.manufacturing.analyzers.sustainability",
        ),
        # Fix visualization imports
        (r"from circman5\.visualization", r"from circman5.manufacturing"),
        # Fix any remaining src.circman5 imports
        (r"from src\.circman5\.", r"from circman5."),
        # Fix relative imports in test_data_generator
        (r"from \.\.config\.project_paths", r"from circman5.config.project_paths"),
    ]

    def fix_file_imports(file_path):
        with open(file_path, "r") as f:
            content = f.read()

        modified = False
        for old_pattern, new_pattern in patterns_to_fix:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                modified = True

        if modified:
            print(f"Fixing imports in {file_path}")
            backup_path = file_path.with_suffix(".py.bak")
            shutil.copy2(file_path, backup_path)
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Backup created at {backup_path}")

    # Fix imports in src directory
    for root, _, files in os.walk(src_path):
        for file in files:
            if file.endswith(".py"):
                fix_file_imports(Path(root) / file)

    # Fix imports in tests directory
    tests_dir = project_root / "tests"
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py"):
                fix_file_imports(Path(root) / file)


def verify_project_structure():
    """Verify that all required directories and files exist."""
    base_dir = src_path / "circman5"
    required_dirs = [
        "manufacturing/analyzers",
        "manufacturing/lifecycle",
        "manufacturing/reporting",
        "manufacturing/optimization",
        "config",
        "utils",
    ]

    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if not full_path.exists():
            print(f"Creating missing directory: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()


def main():
    """Main function to run all project structure fixes."""
    print("Starting project structure cleanup...")
    cleanup_old_structure()
    print("\nVerifying project structure...")
    verify_project_structure()
    print("\nFixing imports...")
    fix_imports()
    print("\nProject structure cleanup completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during project structure cleanup: {str(e)}")
