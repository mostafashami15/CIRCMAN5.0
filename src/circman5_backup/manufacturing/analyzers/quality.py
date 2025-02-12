# src/circman5/manufacturing/analyzers/quality.py
"""
Quality analysis for PV manufacturing.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class QualityAnalyzer:
    def __init__(self):
        self.metrics = {}

    def analyze_defect_rates(self, quality_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze defect patterns and rates.

        Args:
            quality_data: DataFrame containing quality metrics

        Returns:
            Dictionary of quality metrics
        """
        if quality_data.empty:
            return {}

        metrics = {
            "avg_defect_rate": quality_data["defect_rate"].mean(),
            "efficiency_score": quality_data["efficiency"].mean(),
            "uniformity_score": quality_data["thickness_uniformity"].mean(),
        }
        self.metrics.update(metrics)
        return metrics

    def calculate_quality_score(
        self, defect_rate: float, efficiency: float, uniformity: float
    ) -> float:
        """
        Calculate overall quality score.

        Args:
            defect_rate: Percentage of defective products
            efficiency: Production efficiency
            uniformity: Product uniformity measure

        Returns:
            Weighted quality score
        """
        weights = {"defect": 0.4, "efficiency": 0.4, "uniformity": 0.2}
        score = (
            (100 - defect_rate) * weights["defect"]
            + efficiency * weights["efficiency"]
            + uniformity * weights["uniformity"]
        )
        return score

    def identify_quality_trends(
        self, quality_data: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """
        Identify quality trends over time.

        Args:
            quality_data: DataFrame containing quality metrics

        Returns:
            Dictionary of quality trend lists
        """
        if quality_data.empty:
            return {}

        daily_metrics = quality_data.groupby(
            pd.Grouper(key="test_timestamp", freq="D")
        ).agg(
            {
                "defect_rate": "mean",
                "efficiency": "mean",
                "thickness_uniformity": "mean",
            }
        )

        return {
            "defect_trend": daily_metrics["defect_rate"].tolist(),
            "efficiency_trend": daily_metrics["efficiency"].tolist(),
            "uniformity_trend": daily_metrics["thickness_uniformity"].tolist(),
        }

    def detect_quality_anomalies(
        self, quality_data: pd.DataFrame, threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect quality anomalies using statistical methods.

        Args:
            quality_data: DataFrame containing quality metrics
            threshold: Number of standard deviations for anomaly detection

        Returns:
            DataFrame of quality anomalies
        """
        if quality_data.empty:
            return pd.DataFrame()

        # Combine multiple quality metrics into a single composite score
        quality_data["quality_score"] = (
            (100 - quality_data["defect_rate"]) * 0.4
            + quality_data["efficiency"] * 0.4
            + quality_data["thickness_uniformity"] * 0.2
        )

        # Calculate mean and standard deviation
        mean_quality = quality_data["quality_score"].mean()
        std_quality = quality_data["quality_score"].std()

        # Detect anomalies
        anomalies = quality_data[
            np.abs(quality_data["quality_score"] - mean_quality)
            > (threshold * std_quality)
        ]

        return anomalies
