# verify_structure.py
from pathlib import Path


def create_directory_structure():
    """Create and verify required directory structure."""
    base_dir = Path("src/circman5")

    required_dirs = [
        "analysis",
        "analysis/lca",
        "utils",
        "config",
        "visualization",
        "manufacturing",
        "manufacturing/analyzers",
    ]

    # Create directories
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        init_file = full_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"Created {init_file}")


if __name__ == "__main__":
    create_directory_structure()
