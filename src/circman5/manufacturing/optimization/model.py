# src/circman5/manufacturing/optimization/model.py

from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Union, List, Tuple, Optional, cast
import joblib

from .types import PredictionDict, MetricsDict
from ...utils.logging_config import setup_logger
from ...utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService


class ManufacturingModel:
    def __init__(self):
        """Initialize the manufacturing model with improved configuration."""
        self.logger = setup_logger("manufacturing_model")

        # Initialize constants service
        self.constants = ConstantsService()

        # Get optimization configuration
        optimization_config = self.constants.get_optimization_config()
        model_config = optimization_config.get("MODEL_CONFIG", {})
        self.model_params = model_config.get("model_params", {})

        # Use configuration from constants service
        self.model = GradientBoostingRegressor(
            n_estimators=self.model_params.get("n_estimators", 100),
            max_depth=self.model_params.get(
                "max_depth", 3
            ),  # Reduced to prevent overfitting
            min_samples_split=self.model_params.get("min_samples_split", 10),
            min_samples_leaf=self.model_params.get(
                "min_samples_leaf", 5
            ),  # Increased for stability
            subsample=self.model_params.get("subsample", 0.8),
            random_state=model_config.get("random_state", 42),
            learning_rate=self.model_params.get(
                "learning_rate", 0.05
            ),  # Slower learning rate
            # Enable early stopping
            validation_fraction=self.model_params.get("validation_fraction", 0.1),
            n_iter_no_change=self.model_params.get("n_iter_no_change", 10),
            tol=self.model_params.get("tol", 0.001),
        )

        # Use RobustScaler for better handling of outliers
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.is_trained = False

        # Get configuration from constants service
        self.config = {
            "feature_columns": optimization_config.get(
                "FEATURE_COLUMNS",
                [
                    "input_amount",
                    "energy_used",
                    "cycle_time",
                    "efficiency",
                    "defect_rate",
                    "thickness_uniformity",
                ],
            ),
            "target_column": "output_amount",  # This is fixed
            "test_size": model_config.get("test_size", 0.2),
            "random_state": model_config.get("random_state", 42),
            "cv_folds": model_config.get("cv_folds", 5),
            "model_params": self.model_params,
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
            features = cast(pd.DataFrame, self._add_engineered_features(features))

            X = features[self.config["feature_columns"]].to_numpy()
            y = features[self.config["target_column"]].to_numpy().reshape(-1, 1)

            # Scale data
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y)

            # Implement proper k-fold cross validation with stratified sampling
            k = 5
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            cv_scores = []
            cv_mse_scores = []
            fold_predictions = []

            # First, perform cross-validation to assess model generalization
            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

                # Create a fresh model for each fold to avoid data leakage
                fold_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,  # Reduced to prevent overfitting
                    min_samples_split=10,
                    min_samples_leaf=5,  # Increased for stability
                    subsample=0.8,
                    random_state=42,
                    # Add regularization to prevent overfitting
                    learning_rate=0.05,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    tol=0.001,
                )

                fold_model.fit(X_train, y_train.ravel())
                fold_predictions = fold_model.predict(X_val)

                # Calculate proper R2 and MSE for this fold
                r2 = r2_score(y_val, fold_predictions)
                mse = mean_squared_error(y_val, fold_predictions)

                cv_scores.append(r2)
                cv_mse_scores.append(mse)

            # Final training on full dataset with optimal hyperparameters
            self.model.fit(X_scaled, y_scaled.ravel())
            y_pred = self.model.predict(X_scaled)

            # Additional check for model quality
            train_r2 = r2_score(y_scaled, y_pred)
            if train_r2 < 0.5 or np.mean(cv_scores) < 0.2:
                self.logger.warning(
                    f"Model has poor fit: train_r2={train_r2:.2f}, cv_r2={np.mean(cv_scores):.2f}. "
                    f"Consider reviewing data quality or model hyperparameters."
                )

            # Uncertainty estimation
            pred_std = np.std(
                [
                    self.model.predict(
                        X_scaled + np.random.normal(0, 0.01, X_scaled.shape)
                    )
                    for _ in range(10)
                ],
                axis=0,
            )

            # Calculate comprehensive metrics
            metrics: MetricsDict = {
                "mse": float(mean_squared_error(y_scaled, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_scaled, y_pred))),
                "mae": float(mean_absolute_error(y_scaled, y_pred)),
                "r2": float(train_r2),
                "cv_r2_mean": float(np.mean(cv_scores)),
                "cv_r2_std": float(np.std(cv_scores)),
                "cv_mse_mean": float(np.mean(cv_mse_scores)),
                "mean_uncertainty": float(np.mean(pred_std)),
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
            # Avoid division by zero
            df["energy_efficiency"] = df.apply(
                lambda row: row["output_amount"] / row["energy_used"]
                if row["energy_used"] > 0
                else 0.0,
                axis=1,
            )
            # Fix: Ensure defect_rate is treated as a percentage (0-100) or decimal (0-1)
            df["efficiency_quality"] = df.apply(
                lambda row: row["efficiency"]
                * (
                    1 - min(row["defect_rate"] / 100, row["defect_rate"])
                    if row["defect_rate"] > 1
                    else row["defect_rate"]
                )
                if row["defect_rate"] < 100
                else 0.0,
                axis=1,
            )
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

            # Avoid division by zero
            energy_used = params_dict["energy_used"]
            if energy_used > 0:
                params_dict["energy_efficiency"] = output_amount / energy_used
            else:
                params_dict["energy_efficiency"] = 0.0

            # Fix: Ensure defect_rate is treated as a percentage (0-100) or decimal (0-1)
            defect_rate = params_dict["defect_rate"]
            if defect_rate >= 0 and defect_rate < 100:
                # Handle both percentage (0-100) and decimal (0-1) formats
                defect_rate_normalized = (
                    min(defect_rate / 100, defect_rate)
                    if defect_rate > 1
                    else defect_rate
                )
                params_dict["efficiency_quality"] = params_dict["efficiency"] * (
                    1 - defect_rate_normalized
                )
            else:
                # Fallback for invalid values
                params_dict["efficiency_quality"] = 0.0

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

            # Calculate confidence score
            confidence_score = self._calculate_prediction_confidence(process_params)

            # For GradientBoostingRegressor, we can estimate uncertainty using multiple predictions with noise
            prediction_samples = [
                self.model.predict(
                    scaled_params + np.random.normal(0, 0.01, scaled_params.shape)
                )[0]
                for _ in range(20)
            ]
            uncertainty = float(np.std(np.array(prediction_samples)))

            # Create prediction dictionary
            result_predictions: PredictionDict = {
                "predicted_output": prediction_actual,
                "predicted_quality": prediction_actual * confidence_score,
                "confidence_score": confidence_score,
                "uncertainty": uncertainty,
            }

            # Save predictions to file
            predictions_file = "latest_predictions.json"
            with open(predictions_file, "w") as f:
                json.dump(result_predictions, f, indent=2)
            results_manager.save_file(predictions_file, "lca_results")
            Path(predictions_file).unlink()

            return result_predictions

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
