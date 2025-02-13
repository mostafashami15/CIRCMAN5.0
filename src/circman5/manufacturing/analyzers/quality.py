# src/circman5/manufacturing/analyzers/quality.py

"""Quality analysis for PV manufacturing."""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ...utils.logging_config import setup_logger


class QualityAnalyzer:
    def __init__(self):
        self.metrics = {}
        self.logger = setup_logger("quality_analyzer")

    def analyze_defect_rates(self, quality_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze defect patterns and rates."""
        if quality_data.empty:
            self.logger.warning("Empty quality data provided")
            return {}

        metrics = {
            "avg_defect_rate": quality_data["defect_rate"].mean(),
            "efficiency_score": quality_data["efficiency"].mean(),
            "uniformity_score": quality_data["thickness_uniformity"].mean(),
        }

        self.metrics.update(metrics)
        self.logger.info(f"Quality metrics calculated: {metrics}")
        return metrics

    def calculate_quality_score(
        self, quality_data: Optional[pd.DataFrame] = None
    ) -> float:
        """Calculate overall quality score."""
        if quality_data is None and self.metrics:
            defect_rate = self.metrics.get("avg_defect_rate", 0)
            efficiency = self.metrics.get("efficiency_score", 0)
            uniformity = self.metrics.get("uniformity_score", 0)
        elif quality_data is not None and not quality_data.empty:
            defect_rate = quality_data["defect_rate"].mean()
            efficiency = quality_data["efficiency"].mean()
            uniformity = quality_data["thickness_uniformity"].mean()
        else:
            return 0.0

        weights = {"defect": 0.4, "efficiency": 0.4, "uniformity": 0.2}
        score = (
            (100 - defect_rate) * weights["defect"]
            + efficiency * weights["efficiency"]
            + uniformity * weights["uniformity"]
        )
        self.logger.debug(f"Quality score calculated: {score}")
        return score

    def identify_quality_trends(
        self, quality_data: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Identify quality trends over time."""
        if quality_data.empty:
            self.logger.warning("Empty quality data provided for trend analysis")
            return {}

        daily_metrics = quality_data.groupby(pd.Grouper(key="timestamp", freq="D")).agg(
            {
                "defect_rate": "mean",
                "efficiency": "mean",
                "thickness_uniformity": "mean",
            }
        )

        trends = {
            "defect_trend": daily_metrics["defect_rate"].tolist(),
            "efficiency_trend": daily_metrics["efficiency"].tolist(),
            "uniformity_trend": daily_metrics["thickness_uniformity"].tolist(),
        }

        self.logger.info("Quality trends analyzed successfully")
        return trends

    def plot_trends(
        self, trends: Dict[str, List[float]], save_path: Optional[str] = None
    ) -> None:
        """
        Plot quality trends visualization.

        Args:
            trends: Dictionary containing trend data
            save_path: Optional path to save visualization
        """
        try:
            plt.figure(figsize=(12, 8))

            # Plot each trend line
            for trend_name, trend_data in trends.items():
                plt.plot(trend_data, label=trend_name.replace("_", " ").title())

            plt.title("Quality Metrics Trends")
            plt.xlabel("Time Period")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

            self.logger.info(f"Quality trends plot generated successfully")

        except Exception as e:
            self.logger.error(f"Error plotting quality trends: {str(e)}")
            raise
