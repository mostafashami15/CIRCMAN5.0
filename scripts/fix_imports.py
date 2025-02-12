# fix_imports.py
import os
from pathlib import Path
import re


def fix_imports():
    """Fix import statements in the project."""
    src_dir = Path("src/circman5")
    tests_dir = Path("tests")

    patterns_to_fix = [
        # Fix relative imports
        (r"from \.\.(config|utils|analysis)\.", r"from circman5.\1."),
        (r"from src\.circman5\.", "from circman5."),
        # Fix direct imports
        (r"from \.logging_config import", "from circman5.utils.logging_config import"),
        (
            r"from \.config\.project_paths import",
            "from circman5.config.project_paths import",
        ),
        # Fix visualization imports
        (r"from circman5\.visualization\.", "from circman5.visualization."),
    ]

    def fix_file_imports(file_path):
        print(f"Checking {file_path}")
        with open(file_path, "r") as f:
            content = f.read()

        modified = False
        for old_pattern, new_pattern in patterns_to_fix:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                modified = True

        if modified:
            print(f"Fixing imports in {file_path}")
            with open(file_path, "w") as f:
                f.write(content)

    # Process src directory
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                fix_file_imports(Path(root) / file)

    # Process tests directory
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py"):
                fix_file_imports(Path(root) / file)


if __name__ == "__main__":
    fix_imports()
