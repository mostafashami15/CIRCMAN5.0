"""
Manufacturing monitoring and metrics tracking system.
Tracks key performance indicators (KPIs) for PV manufacturing processes.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from .config.project_paths import project_paths
from .logging_config import setup_logger


class ManufacturingMonitor:
    """Monitors and tracks manufacturing performance metrics."""

    def __init__(self):
        """Initialize monitoring system with empty metrics storage."""
        self.metrics_history = {
            "efficiency": pd.DataFrame(),
            "quality": pd.DataFrame(),
            "resources": pd.DataFrame(),
        }
        self.current_batch: Optional[str] = None
        self.logger = setup_logger("manufacturing_monitor")

    def start_batch_monitoring(self, batch_id: str) -> None:
        """
        Start monitoring a new manufacturing batch.

        Args:
            batch_id: Unique identifier for the batch
        """
        self.current_batch = batch_id
        self._record_batch_start(batch_id)

    def record_efficiency_metrics(
        self, output_quantity: float, cycle_time: float, energy_consumption: float
    ) -> Dict:
        """
        Record efficiency-related metrics for current batch.

        Args:
            output_quantity: Amount of product produced
            cycle_time: Production cycle duration
            energy_consumption: Energy used in production

        Returns:
            Dict containing calculated efficiency metrics
        """
        if not self.current_batch:
            raise ValueError("No active batch being monitored")

        metrics = {
            "batch_id": self.current_batch,
            "timestamp": datetime.now(),
            "output_quantity": output_quantity,
            "cycle_time": cycle_time,
            "energy_consumption": energy_consumption,
            "production_rate": output_quantity / cycle_time if cycle_time > 0 else 0,
            "energy_efficiency": output_quantity / energy_consumption
            if energy_consumption > 0
            else 0,
        }

        self.metrics_history["efficiency"] = pd.concat(
            [self.metrics_history["efficiency"], pd.DataFrame([metrics])],
            ignore_index=True,
        )

        return metrics

    def record_quality_metrics(
        self, defect_rate: float, yield_rate: float, uniformity_score: float
    ) -> Dict:
        """
        Record quality-related metrics for current batch.

        Args:
            defect_rate: Percentage of defective products
            yield_rate: Production yield percentage
            uniformity_score: Product uniformity measure

        Returns:
            Dict containing calculated quality metrics
        """
        metrics = {
            "batch_id": self.current_batch,
            "timestamp": datetime.now(),
            "defect_rate": defect_rate,
            "yield_rate": yield_rate,
            "uniformity_score": uniformity_score,
            "quality_score": self._calculate_quality_score(
                defect_rate, yield_rate, uniformity_score
            ),
        }

        self.metrics_history["quality"] = pd.concat(
            [self.metrics_history["quality"], pd.DataFrame([metrics])],
            ignore_index=True,
        )

        return metrics

    def record_resource_metrics(
        self, material_consumption: float, water_usage: float, waste_generated: float
    ) -> Dict:
        """
        Record resource utilization metrics for current batch.

        Args:
            material_consumption: Amount of raw materials used
            water_usage: Volume of water consumed
            waste_generated: Amount of waste produced

        Returns:
            Dict containing calculated resource metrics
        """
        metrics = {
            "batch_id": self.current_batch,
            "timestamp": datetime.now(),
            "material_consumption": material_consumption,
            "water_usage": water_usage,
            "waste_generated": waste_generated,
            "resource_efficiency": self._calculate_resource_efficiency(
                material_consumption, waste_generated
            ),
        }

        self.metrics_history["resources"] = pd.concat(
            [self.metrics_history["resources"], pd.DataFrame([metrics])],
            ignore_index=True,
        )

        return metrics

    def save_metrics(self, metric_type: str, save_path: Optional[Path] = None) -> None:
        """Save metrics to appropriate location.

        Args:
            metric_type: Type of metrics to save ('efficiency', 'quality', etc.)
            save_path: Optional explicit path to save metrics file
        """
        if metric_type not in self.metrics_history:
            raise ValueError(f"Invalid metric type: {metric_type}")

        if save_path is None:
            run_dir = project_paths.get_run_directory()
            save_path = run_dir / "reports" / f"{metric_type}_metrics.csv"

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metrics
        self.metrics_history[metric_type].to_csv(save_path, index=False)
        self.logger.info(f"Saved {metric_type} metrics to: {save_path}")

    def get_batch_summary(self, batch_id: str) -> Dict:
        """Generate batch summary and save to reports directory."""
        summary = {
            "efficiency": self._summarize_efficiency(batch_id),
            "quality": self._summarize_quality(batch_id),
            "resources": self._summarize_resources(batch_id),
        }

        run_dir = project_paths.get_run_directory()
        summary_file = run_dir / "reports" / f"batch_{batch_id}_summary.xlsx"

        with pd.ExcelWriter(summary_file) as writer:
            for metric_type, data in summary.items():
                pd.DataFrame([data]).to_excel(writer, sheet_name=metric_type)

        return summary

    def _record_batch_start(self, batch_id: str) -> None:
        """Record the start of a new batch monitoring session."""
        # Implementation for recording batch start time and initial conditions
        pass

    def _calculate_quality_score(
        self, defect_rate: float, yield_rate: float, uniformity_score: float
    ) -> float:
        """Calculate composite quality score."""
        # Weighted average of quality metrics
        weights = {"defect": 0.4, "yield": 0.4, "uniformity": 0.2}
        return (
            (100 - defect_rate) * weights["defect"]
            + yield_rate * weights["yield"]
            + uniformity_score * weights["uniformity"]
        )

    def _calculate_resource_efficiency(
        self, material_input: float, waste_output: float
    ) -> float:
        """Calculate resource utilization efficiency."""
        return (
            (material_input - waste_output) / material_input
            if material_input > 0
            else 0
        )

    def _summarize_efficiency(self, batch_id: str) -> Dict:
        """Generate efficiency metrics summary for a batch."""
        batch_data = self.metrics_history["efficiency"][
            self.metrics_history["efficiency"]["batch_id"] == batch_id
        ]
        return {
            "avg_production_rate": batch_data["production_rate"].mean(),
            "total_energy_consumption": batch_data["energy_consumption"].sum(),
            "avg_energy_efficiency": batch_data["energy_efficiency"].mean(),
        }

    def _summarize_quality(self, batch_id: str) -> Dict:
        """Generate quality metrics summary for a batch."""
        batch_data = self.metrics_history["quality"][
            self.metrics_history["quality"]["batch_id"] == batch_id
        ]
        return {
            "avg_defect_rate": batch_data["defect_rate"].mean(),
            "final_yield_rate": batch_data["yield_rate"].iloc[-1]
            if not batch_data.empty
            else 0,
            "avg_quality_score": batch_data["quality_score"].mean(),
        }

    def _summarize_resources(self, batch_id: str) -> Dict:
        """Generate resource utilization summary for a batch."""
        batch_data = self.metrics_history["resources"][
            self.metrics_history["resources"]["batch_id"] == batch_id
        ]
        return {
            "total_material_consumption": batch_data["material_consumption"].sum(),
            "total_waste_generated": batch_data["waste_generated"].sum(),
            "avg_resource_efficiency": batch_data["resource_efficiency"].mean(),
        }
