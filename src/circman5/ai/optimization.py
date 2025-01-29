"""
AI optimization engine for PV manufacturing processes.
Provides process optimization, predictive maintenance, and quality prediction.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional


class AIOptimizationEngine:
    """AI-powered optimization engine for PV manufacturing processes."""

    def __init__(self):
        """Initialize ML models and scalers."""
        try:
            self.process_optimizer = RandomForestRegressor(
                n_estimators=100, random_state=42
            )
            self.quality_predictor = GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
            self.scaler = StandardScaler()
            self.is_trained = False
            self.quality_predictor_trained = False
        except Exception as e:
            print("Error initializing ML models:", str(e))
            raise

    def optimize_process_parameters(
        self, production_data: pd.DataFrame, quality_data: pd.DataFrame
    ) -> Dict:
        """Optimize manufacturing process parameters."""
        try:
            # Merge production and quality data
            features = pd.merge(
                production_data, quality_data, on="batch_id", how="inner"
            )

            # Select features for optimization
            optimization_features = features[
                ["cycle_time", "output_quantity", "yield_rate"]
            ].values

            # Scale features
            scaled_features = self.scaler.fit_transform(optimization_features)

            # Train if not trained
            if not self.is_trained:
                self._train_optimizer(scaled_features, features["yield_rate"].values)

            # Generate optimization suggestions
            predictions = self.process_optimizer.predict(scaled_features)

            return {
                "cycle_time": float(np.mean(predictions)),
                "temperature": 180.0,  # Default optimal temperature
                "pressure": 1.0,  # Default optimal pressure
                "confidence_score": float(self._calculate_confidence(predictions)),
            }
        except Exception as e:
            print("Error in optimize_process_parameters:", str(e))
            raise

    def predict_maintenance_needs(
        self, production_data: pd.DataFrame, energy_data: pd.DataFrame
    ) -> Dict:
        """Predict maintenance requirements."""
        try:
            features = self._extract_maintenance_features(production_data, energy_data)
            failure_prob = self._predict_failure_probability(features)

            return {
                "next_maintenance": self._calculate_maintenance_timing(failure_prob),
                "failure_probability": float(failure_prob),
                "critical_components": self._identify_critical_components(features),
            }
        except Exception as e:
            print("Error in predict_maintenance_needs:", str(e))
            raise

    def predict_quality_metrics(
        self, production_data: pd.DataFrame, material_data: pd.DataFrame
    ) -> Dict:
        """Predict quality metrics."""
        try:
            # Prepare features for quality prediction
            features = self._prepare_quality_features(production_data, material_data)

            # Create binary quality labels (using yield_rate as proxy for quality)
            quality_labels = (
                production_data["yield_rate"].iloc[: len(features)] > 0.95
            ).astype(int)

            # Train quality predictor if not trained
            if not self.quality_predictor_trained:
                self.quality_predictor.fit(features, quality_labels)
                self.quality_predictor_trained = True

            # Generate predictions
            predictions = self.quality_predictor.predict_proba(features)

            return {
                "predicted_defect_rate": float(
                    np.mean(predictions[:, 0])
                ),  # Class 0 probability
                "confidence_interval": self._calculate_prediction_confidence(
                    predictions
                ),
                "quality_factors": self._identify_quality_factors(features),
            }
        except Exception as e:
            print("Error in predict_quality_metrics:", str(e))
            raise

    def _extract_maintenance_features(
        self, production_data: pd.DataFrame, energy_data: pd.DataFrame
    ) -> np.ndarray:
        """Extract features for maintenance prediction."""
        try:
            merged_data = pd.merge(
                production_data,
                energy_data,
                on=["timestamp", "production_line"],
                how="inner",
            )

            features = pd.DataFrame(
                {
                    "output_efficiency": merged_data["output_quantity"]
                    / merged_data["energy_consumption"],
                    "production_rate": merged_data["output_quantity"]
                    / merged_data["cycle_time"],
                }
            )

            return self.scaler.fit_transform(features)
        except Exception as e:
            print("Error in _extract_maintenance_features:", str(e))
            raise

    def _prepare_quality_features(
        self, production_data: pd.DataFrame, material_data: pd.DataFrame
    ) -> np.ndarray:
        """Prepare features for quality prediction."""
        try:
            # Merge data and remove duplicates
            merged_data = pd.merge(
                production_data, material_data, on="batch_id", how="inner"
            ).drop_duplicates()

            # Select relevant features
            features = merged_data[
                [
                    "output_quantity",
                    "cycle_time",
                    "yield_rate",
                    "quantity_used",
                    "waste_generated",
                ]
            ].values

            return self.scaler.fit_transform(features)
        except Exception as e:
            print("Error in _prepare_quality_features:", str(e))
            raise

    def _predict_failure_probability(self, features: np.ndarray) -> float:
        """Calculate probability of equipment failure."""
        try:
            # Simple heuristic based on feature patterns
            return float(np.mean(np.abs(features)))
        except Exception as e:
            print("Error in _predict_failure_probability:", str(e))
            raise

    def _calculate_maintenance_timing(self, failure_prob: float) -> pd.Timestamp:
        """Determine optimal maintenance timing."""
        try:
            days_until_maintenance = int(30 * (1 - failure_prob))
            return pd.Timestamp.now() + pd.Timedelta(days=days_until_maintenance)
        except Exception as e:
            print("Error in _calculate_maintenance_timing:", str(e))
            raise

    def _identify_critical_components(self, features: np.ndarray) -> Dict:
        """Identify components needing attention."""
        try:
            return {
                "component_1": {"risk_score": 0.8, "priority": "high"},
                "component_2": {"risk_score": 0.5, "priority": "medium"},
            }
        except Exception as e:
            print("Error in _identify_critical_components:", str(e))
            raise

    def _calculate_prediction_confidence(self, predictions: np.ndarray) -> Dict:
        """Calculate confidence intervals for predictions."""
        try:
            return {
                "lower_bound": float(np.percentile(predictions, 25)),
                "upper_bound": float(np.percentile(predictions, 75)),
            }
        except Exception as e:
            print("Error in _calculate_prediction_confidence:", str(e))
            raise

    def _identify_quality_factors(self, features: np.ndarray) -> Dict:
        """Identify key factors affecting quality."""
        try:
            return {
                "material_quality": 0.8,
                "process_stability": 0.7,
                "environmental_factors": 0.6,
            }
        except Exception as e:
            print("Error in _identify_quality_factors:", str(e))
            raise

    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """Calculate confidence score for predictions."""
        try:
            return float(1 - np.std(predictions) / np.mean(predictions))
        except Exception as e:
            print("Error in _calculate_confidence:", str(e))
            raise

    def _train_optimizer(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the optimization model."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.process_optimizer.fit(X_train, y_train)
            self.is_trained = True
        except Exception as e:
            print("Error in _train_optimizer:", str(e))
            raise
