# src/circman5/manufacturing/analyzers/efficiency.py
"""
Efficiency analyzer for manufacturing processes.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


class EfficiencyAnalyzer:
    """Analyzes manufacturing efficiency metrics."""

    def __init__(self):
        """Initialize efficiency analyzer."""
        self.metrics = {}

    def analyze_batch_efficiency(
        self, production_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Analyze batch production efficiency.

        Args:
            production_data: DataFrame containing production metrics

        Returns:
            Dict containing efficiency metrics
        """
        if production_data.empty:
            return {}

        try:
            # Use output_amount consistently
            required_columns = ["input_amount", "output_amount"]
            if not all(col in production_data.columns for col in required_columns):
                return {}

            # Ensure values are non-negative
            valid_output = max(0, production_data["output_amount"].mean())
            valid_input = max(0, production_data["input_amount"].mean())

            # Calculate yield rate
            yield_rate = (valid_output / valid_input * 100) if valid_input > 0 else 0

            metrics = {"yield_rate": yield_rate}

            # Calculate cycle time efficiency if data available
            if "cycle_time" in production_data.columns:
                valid_cycle_time = max(
                    0.1, production_data["cycle_time"].mean()
                )  # Avoid division by zero
                cycle_time_efficiency = max(0, valid_output / valid_cycle_time)
                metrics["cycle_time_efficiency"] = cycle_time_efficiency

            # Calculate energy efficiency if data available
            if "energy_used" in production_data.columns:
                valid_energy = max(
                    0.1, production_data["energy_used"].mean()
                )  # Avoid division by zero
                energy_efficiency = max(0, valid_output / valid_energy)
                metrics["energy_efficiency"] = energy_efficiency

            # Store metrics
            self.metrics.update(metrics)
            return metrics

        except Exception as e:
            print(f"Error in efficiency analysis: {str(e)}")
            return {}

    def calculate_overall_efficiency(self, production_data: pd.DataFrame) -> float:
        """
        Calculate comprehensive manufacturing efficiency.

        Args:
            production_data: DataFrame containing production metrics

        Returns:
            Overall efficiency as a percentage
        """
        if production_data.empty:
            return 0.0

        return (
            production_data["output_amount"] / production_data["input_amount"]
        ).mean() * 100

    def identify_efficiency_trends(
        self, production_data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Identify efficiency trends over time.

        Args:
            production_data: DataFrame containing production metrics

        Returns:
            Dictionary of efficiency trend series
        """
        if production_data.empty:
            return {}

        # Ensure timestamp is set as index
        df = production_data.set_index("timestamp")

        trends = {
            "daily_efficiency": df["output_amount"].resample("D").mean(),
            "weekly_efficiency": df["output_amount"].resample("W").mean(),
            "monthly_efficiency": df["output_amount"].resample("M").mean(),
        }

        return trends

    def detect_efficiency_anomalies(
        self, production_data: pd.DataFrame, threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect efficiency anomalies using statistical methods.

        Args:
            production_data: DataFrame containing production metrics
            threshold: Number of standard deviations for anomaly detection

        Returns:
            DataFrame containing efficiency anomalies
        """
        if production_data.empty:
            return pd.DataFrame()

        # Calculate efficiency
        production_data["efficiency"] = (
            production_data["output_amount"] / production_data["input_amount"]
        ) * 100

        # Calculate mean and standard deviation
        mean_efficiency = production_data["efficiency"].mean()
        std_efficiency = production_data["efficiency"].std()

        # Detect anomalies
        anomalies = production_data[
            np.abs(production_data["efficiency"] - mean_efficiency)
            > (threshold * std_efficiency)
        ]

        return anomalies
