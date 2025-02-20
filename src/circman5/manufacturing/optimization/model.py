# src/circman5/manufacturing/optimization/model.py

from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Union, List, Tuple, Optional, cast
import joblib

from .types import PredictionDict, MetricsDict
from ...utils.logging_config import setup_logger
from ...utils.results_manager import results_manager


class ManufacturingModel:
    def __init__(self):
        """Initialize the manufacturing model with improved configuration."""
        self.logger = setup_logger("manufacturing_model")

        # Use more robust model configuration
        self.model = RandomForestRegressor(
            n_estimators=200,  # More trees
            max_depth=10,  # Control overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,  # Use all available cores
            random_state=42,
        )

        # Use RobustScaler for better handling of outliers
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.is_trained = False

        # Enhanced configuration
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
            "cv_folds": 5,  # For cross-validation
            "model_params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
        }

    def train_optimization_model(
        self, production_data: pd.DataFrame, quality_data: pd.DataFrame
    ) -> MetricsDict:
        """Train the AI optimization model with enhanced validation."""
        if production_data.empty or quality_data.empty:
            raise ValueError("No data available for training optimization model")

        try:
            # Prepare data with feature engineering
            features = self._prepare_features(production_data, quality_data)

            # Add engineered features
            features = cast(pd.DataFrame, self._add_engineered_features(features))

            X = features[self.config["feature_columns"]].to_numpy()
            y = features[self.config["target_column"]].to_numpy().reshape(-1, 1)

            # Scale data
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y)

            # Cross-validation
            cv_scores = cross_val_score(
                self.model,
                X_scaled,
                y_scaled.ravel(),
                cv=self.config["cv_folds"],
                scoring="r2",
            )

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y_scaled,
                test_size=self.config["test_size"],
                random_state=self.config["random_state"],
            )

            # Train model
            self.model.fit(X_train, y_train.ravel())

            # Predictions for metrics
            y_pred = self.model.predict(X_test)

            # Calculate comprehensive metrics
            metrics: MetricsDict = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
                "cv_r2_mean": float(np.mean(cv_scores)),
                "cv_r2_std": float(np.std(cv_scores)),
                "feature_importance": {
                    feat: float(imp)
                    for feat, imp in zip(
                        self.config["feature_columns"],
                        self.model.feature_importances_.tolist(),
                    )
                },
            }

            self.is_trained = True

            # Save enhanced metrics
            metrics_file = "training_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            results_manager.save_file(metrics_file, "metrics")
            Path(metrics_file).unlink()

            self.logger.info(
                f"Model trained successfully with enhanced metrics: {metrics}"
            )
            return metrics

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def _prepare_features(
        self, production_data: pd.DataFrame, quality_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare and validate features with enhanced checks."""
        features = pd.merge(
            production_data, quality_data, on="batch_id", suffixes=("_prod", "_qual")
        )

        # Validate all required columns are present
        missing_cols = set(self.config["feature_columns"]) - set(features.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove rows with invalid values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()

        return features

    def _add_engineered_features(
        self, params: Union[Dict[str, float], pd.DataFrame]
    ) -> Union[Dict[str, float], pd.DataFrame]:
        """Add engineered features to parameters."""

        def validate_required_params(
            p: Union[Dict[str, float], pd.DataFrame], required: List[str]
        ) -> None:
            """Validate required parameters exist in either dict or DataFrame."""
            if isinstance(p, pd.DataFrame):
                missing = [key for key in required if key not in p.columns]
            else:
                missing = [key for key in required if key not in p]
            if missing:
                raise ValueError(
                    f"Missing required parameters for feature calculation: {missing}"
                )

        if isinstance(params, pd.DataFrame):
            df = params.copy()
            validate_required_params(
                df,
                [
                    "output_amount",
                    "input_amount",
                    "energy_used",
                    "efficiency",
                    "defect_rate",
                ],
            )
            df["efficiency_rate"] = df["output_amount"] / df["input_amount"]
            df["energy_efficiency"] = df["output_amount"] / df["energy_used"]
            df["efficiency_quality"] = df["efficiency"] * (1 - df["defect_rate"])
            return df
        else:
            # Handle dictionary input
            params_dict = params.copy()
            validate_required_params(
                params_dict,
                ["input_amount", "energy_used", "efficiency", "defect_rate"],
            )

            output_amount = params_dict.get(
                "output_amount", 0.9 * params_dict["input_amount"]
            )
            params_dict["output_amount"] = output_amount

            params_dict["efficiency_rate"] = output_amount / params_dict["input_amount"]
            params_dict["energy_efficiency"] = (
                output_amount / params_dict["energy_used"]
            )
            params_dict["efficiency_quality"] = params_dict["efficiency"] * (
                1 - params_dict["defect_rate"]
            )
            return params_dict

    def predict_batch_outcomes(
        self, process_params: Dict[str, float]
    ) -> PredictionDict:
        """Predict manufacturing outcomes with enhanced validation."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Add engineered features before validation
            process_params = cast(
                Dict[str, float], self._add_engineered_features(process_params)
            )

            # Validate parameters
            self._validate_prediction_params(process_params)

            # Prepare input array
            input_array = np.array(
                [[process_params[col] for col in self.config["feature_columns"]]]
            )

            # Scale and predict
            scaled_params = self.feature_scaler.transform(input_array)
            prediction_scaled = self.model.predict(scaled_params)
            prediction_actual = float(
                self.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[
                    0
                ][0]
            )

            # Calculate confidence
            confidence_score = self._calculate_prediction_confidence(process_params)

            predictions: PredictionDict = {
                "predicted_output": prediction_actual,
                "predicted_quality": prediction_actual * confidence_score,
                "confidence_score": confidence_score,
            }

            # Save predictions to file
            predictions_file = "latest_predictions.json"
            with open(predictions_file, "w") as f:
                json.dump(predictions, f, indent=2)
            results_manager.save_file(predictions_file, "lca_results")
            Path(predictions_file).unlink()

            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def _validate_prediction_params(self, params: Dict[str, float]) -> None:
        """Validate prediction parameters."""
        if not params:
            raise ValueError("Empty parameters dictionary")

        required_params = set(self.config["feature_columns"])
        missing_params = required_params - set(params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        if any(not isinstance(v, (int, float)) for v in params.values()):
            raise ValueError("All parameter values must be numeric")

    def _process_prediction_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Process and engineer features for prediction."""
        processed = params.copy()

        # Add engineered features
        processed["efficiency_rate"] = params["output_amount"] / params["input_amount"]
        processed["energy_efficiency"] = params["output_amount"] / params["energy_used"]
        processed["efficiency_quality"] = params["efficiency"] * (
            1 - params["defect_rate"]
        )

        return processed

    def _calculate_prediction_confidence(self, params: Dict[str, float]) -> float:
        """Calculate confidence score based on feature importance and input ranges."""
        if not hasattr(self.model, "feature_importances_"):
            return 0.9  # Default confidence if feature importance not available

        importance_dict = dict(
            zip(self.config["feature_columns"], self.model.feature_importances_)
        )

        # Weight parameters by their importance
        confidence = sum(
            importance_dict[param] * (0.9 + 0.1 * np.random.random())
            for param in params.keys()
            if param in importance_dict
        )

        return min(max(confidence, 0.5), 0.99)  # Bound between 0.5 and 0.99

    def save_model(self, model_name: str) -> None:
        """Save the trained model and its configuration.

        Args:
            model_name: Base name for saved model files
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        try:
            # Save model
            model_file = Path(f"{model_name}_model.joblib")
            joblib.dump(self.model, model_file)

            # Save scalers
            scalers_file = Path(f"{model_name}_scalers.joblib")
            scalers = {
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
            }
            joblib.dump(scalers, scalers_file)

            # Save config
            config_file = Path(f"{model_name}_config.json")
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)

            # Save to results using results_manager
            try:
                results_manager.save_file(model_file, "lca_results")
                results_manager.save_file(scalers_file, "lca_results")
                results_manager.save_file(config_file, "lca_results")

                # Cleanup temporary files
                model_file.unlink()
                scalers_file.unlink()
                config_file.unlink()

                self.logger.info(f"Model '{model_name}' saved successfully")

            except Exception as e:
                self.logger.error(
                    f"Error saving model files using results_manager: {str(e)}"
                )
                raise

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
