"""
Manufacturing efficiency analysis for PV production.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class EfficiencyAnalyzer:
    def __init__(self):
        """Initialize efficiency analyzer."""
        self.metrics = {}

    def calculate_production_efficiency(
        self, input_amount: float, output_amount: float
    ) -> float:
        """Calculate production efficiency ratio."""
        return (output_amount / input_amount * 100) if input_amount > 0 else 0

    def analyze_batch_efficiency(
        self, production_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze batch production efficiency."""
        if production_data.empty:
            return {}

        metrics = {
            "yield_rate": (
                production_data["output_amount"] / production_data["input_amount"]
            ).mean()
            * 100,
            "cycle_time_efficiency": (
                production_data["output_amount"] / production_data["cycle_time"]
            ).mean(),
            "energy_efficiency": (
                production_data["output_amount"] / production_data["energy_used"]
            ).mean(),
        }

        self.metrics.update(metrics)
        return metrics
