# scripts/fix_project_structure.py

import shutil
from pathlib import Path


def fix_project_structure():
    """Fix project structure to match desired organization."""

    src_root = Path("src/circman5")

    # Fix imports in manufacturing core
    core_file = src_root / "manufacturing/core.py"
    if core_file.exists():
        content = core_file.read_text()
        # Update relative imports to use absolute imports
        content = content.replace("from ..utils", "from circman5.utils")
        content = content.replace(
            "from .analyzers", "from circman5.manufacturing.analyzers"
        )
        core_file.write_text(content)

    # Clean up backup files
    backup_files = src_root.rglob("*.bak")
    for file in backup_files:
        file.unlink()

    # Remove circman5_backup if it exists
    backup_dir = Path("src/circman5_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    # Create required __init__.py files
    init_locations = [
        "src/circman5/manufacturing/analyzers",
        "src/circman5/manufacturing/reporting",
        "src/circman5/manufacturing/lifecycle",
        "src/circman5/manufacturing/optimization",
        "src/circman5/utils",
        "src/circman5/config",
    ]

    for loc in init_locations:
        init_file = Path(loc) / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch()


if __name__ == "__main__":
    fix_project_structure()
