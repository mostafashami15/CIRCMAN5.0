"""
Efficiency analyzer for manufacturing processes.
"""

import pandas as pd
from typing import Dict


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
