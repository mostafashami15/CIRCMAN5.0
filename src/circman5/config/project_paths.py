# src/circman5/config/project_paths.py
from pathlib import Path
from ..utils.results_manager import results_manager


class ProjectPaths:
    """Legacy wrapper around ResultsManager."""

    def __init__(self):
        self.PROJECT_ROOT = results_manager.project_root

    def get_run_directory(self) -> Path:
        return results_manager.get_run_dir()

    def get_path(self, key: str) -> str:
        return str(results_manager.get_path(key))

    def get_synthetic_data_path(self, filename: str) -> str:
        synthetic_dir = results_manager.get_path("SYNTHETIC_DATA")
        return str(synthetic_dir / filename)


# Keep global instance for backward compatibility
project_paths = ProjectPaths()
