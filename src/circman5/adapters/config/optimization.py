# src/circman5/adapters/config/optimization.py

from pathlib import Path
from typing import Dict, Any, Optional
import json

from ..base.adapter_base import ConfigAdapterBase


class OptimizationAdapter(ConfigAdapterBase):
    """Adapter for optimization parameters and configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize optimization adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "optimization.json"
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load optimization configuration.

        Returns:
            Dict[str, Any]: Optimization configuration

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        if not self.config_path.exists():
            self.logger.warning(
                f"Config file not found: {self.config_path}, using defaults"
            )
            return self.get_defaults()

        return self._load_json_config(self.config_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate optimization configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        required_sections = {
            "MODEL_CONFIG",
            "FEATURE_COLUMNS",
            "OPTIMIZATION_CONSTRAINTS",
            "TRAINING_PARAMETERS",
        }

        # Add validation for new sections if they exist
        if "ADVANCED_MODELS" in config:
            advanced_models = config.get("ADVANCED_MODELS", {})
            if not all(
                model in advanced_models for model in ["deep_learning", "ensemble"]
            ):
                self.logger.warning("Missing required advanced model configurations")
                # Don't fail validation, just warn

        if "ONLINE_LEARNING" in config:
            online_learning = config.get("ONLINE_LEARNING", {})
            required_online_params = {
                "window_size",
                "learning_rate",
                "update_frequency",
                "forgetting_factor",
            }
            if not all(param in online_learning for param in required_online_params):
                self.logger.warning("Missing required online learning parameters")
                # Don't fail validation, just warn

        if "VALIDATION" in config:
            validation = config.get("VALIDATION", {})
            required_validation_sections = {"cross_validation", "uncertainty"}
            if not all(
                section in validation for section in required_validation_sections
            ):
                self.logger.warning("Missing required validation sections")
                # Don't fail validation, just warn

        # Check required top-level sections
        if not all(section in config for section in required_sections):
            self.logger.error(
                f"Missing required sections: {required_sections - set(config.keys())}"
            )
            return False

        # Validate model configuration
        model_config = config.get("MODEL_CONFIG", {})
        required_model_params = {
            "test_size",
            "random_state",
            "cv_folds",
            "model_params",
        }

        if not all(param in model_config for param in required_model_params):
            self.logger.error("Invalid model configuration")
            return False

        # Validate feature columns
        feature_cols = config.get("FEATURE_COLUMNS", [])
        required_features = {
            "input_amount",
            "energy_used",
            "cycle_time",
            "efficiency",
            "defect_rate",
            "thickness_uniformity",
        }

        if not all(feature in feature_cols for feature in required_features):
            self.logger.error("Missing required feature columns")
            return False

        # Validate optimization constraints
        constraints = config.get("OPTIMIZATION_CONSTRAINTS", {})
        required_constraints = {
            "min_yield_rate",
            "max_cycle_time",
            "min_efficiency",
            "max_defect_rate",
        }

        if not all(constraint in constraints for constraint in required_constraints):
            self.logger.error("Missing required optimization constraints")
            return False

        # Validate training parameters
        training_params = config.get("TRAINING_PARAMETERS", {})
        required_training_params = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "learning_rate",
        }

        if not all(param in training_params for param in required_training_params):
            self.logger.error("Missing required training parameters")
            return False

        return True

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default optimization configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "MODEL_CONFIG": {
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 5,
                "model_params": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "subsample": 0.8,
                },
            },
            "FEATURE_COLUMNS": [
                "input_amount",
                "energy_used",
                "cycle_time",
                "efficiency",
                "defect_rate",
                "thickness_uniformity",
            ],
            "OPTIMIZATION_CONSTRAINTS": {
                "min_yield_rate": 92.0,
                "max_cycle_time": 60.0,
                "min_efficiency": 18.0,
                "max_defect_rate": 5.0,
                "min_thickness_uniformity": 90.0,
                "max_energy_consumption": 160.0,
            },
            "TRAINING_PARAMETERS": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "tol": 1e-4,
            },
            "ADVANCED_MODELS": {
                "deep_learning": {
                    "model_type": "lstm",
                    "hidden_layers": [64, 32],
                    "activation": "relu",
                    "dropout_rate": 0.2,
                    "l2_regularization": 0.001,
                    "batch_size": 32,
                    "epochs": 100,
                    "early_stopping_patience": 10,
                },
                "ensemble": {
                    "base_models": [
                        "random_forest",
                        "gradient_boosting",
                        "extra_trees",
                    ],
                    "meta_model": "linear",
                    "cv_folds": 5,
                    "use_probabilities": True,
                    "voting_strategy": "soft",
                },
            },
            "ONLINE_LEARNING": {
                "window_size": 100,
                "learning_rate": 0.01,
                "update_frequency": 10,
                "regularization": 0.001,
                "forgetting_factor": 0.95,
                "max_model_age": 24,
                "model_persistence_interval": 60,
            },
            "VALIDATION": {
                "cross_validation": {
                    "method": "stratified_kfold",
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": 42,
                    "metrics": ["accuracy", "precision", "recall", "f1", "r2", "mse"],
                },
                "uncertainty": {
                    "method": "monte_carlo_dropout",
                    "samples": 30,
                    "confidence_level": 0.95,
                    "calibration_method": "temperature_scaling",
                },
            },
        }
