"""Base class for manufacturing optimization."""

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from circman5.config.project_paths import project_paths
from ..logging_config import setup_logger


class OptimizerBase:
    """Base class containing shared attributes for optimization."""

    def __init__(self):
        # Initialize models
        self.efficiency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.quality_model = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )

        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # State tracking
        self.is_trained = False

        # Setup logging and paths
        self.logger = setup_logger("manufacturing_optimizer")
        self.run_dir = project_paths.get_run_directory()
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Default model configuration
        self.config = {
            "feature_columns": [
                "input_amount",
                "energy_used",
                "cycle_time",
                "efficiency",
                "defect_rate",
                "thickness_uniformity",
            ],
            "target_column": "output_amount",
            "test_size": 0.2,
            "random_state": 42,
        }
