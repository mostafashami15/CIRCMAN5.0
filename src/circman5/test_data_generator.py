"""Test data generator for manufacturing analysis system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List

from .analysis.lca.impact_factors import (
    MATERIAL_IMPACT_FACTORS,
    ENERGY_IMPACT_FACTORS,
    RECYCLING_BENEFIT_FACTORS,
    PROCESS_IMPACT_FACTORS,
    GRID_CARBON_INTENSITIES,
)

# Explicitly tell pytest not to collect this module for tests
__test__ = False


class ManufacturingDataGenerator:  # Renamed class to remove "Test" prefix
    """Generates synthetic manufacturing data for testing.
    This is a utility class for generating test data, not a test class.
    """

    def __init__(self, start_date: str = "2024-01-01", days: int = 30):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.days = days
        self.batch_ids = [f"BATCH_{i:04d}" for i in range(1, 101)]
        self.production_lines = ["LINE_A", "LINE_B", "LINE_C"]
        self.product_types = ["Mono_PERC_60", "Mono_PERC_72", "Bifacial_72"]
        self.material_types = ["Silicon_Wafer", "Glass", "EVA", "Backsheet", "Frame"]
        self.energy_sources = ["grid", "solar", "wind"]

        # LCA-specific parameters
        self.panel_sizes = {
            "Mono_PERC_60": 1.6,  # m²
            "Mono_PERC_72": 2.0,  # m²
            "Bifacial_72": 2.0,  # m²
        }

        self.material_composition = {
            "silicon_wafer": 0.5,  # kg/m²
            "solar_glass": 10.0,  # kg/m²
            "eva_sheet": 1.0,  # kg/m²
            "backsheet": 0.5,  # kg/m²
            "aluminum_frame": 2.0,  # kg/m²
            "copper_wiring": 0.2,  # kg/m²
        }

    def generate_production_data(self) -> pd.DataFrame:
        """Generate production data with all required columns."""
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            for hour in range(8, 16):
                for line in self.production_lines:
                    timestamp = current_date + timedelta(hours=hour)

                    input_amount = random.uniform(90, 110)
                    output_amount = input_amount * random.uniform(0.85, 0.98)

                    data.append(
                        {
                            "timestamp": timestamp,
                            "batch_id": random.choice(self.batch_ids),
                            "product_type": random.choice(self.product_types),
                            "production_line": line,
                            "input_amount": input_amount,
                            "output_amount": output_amount,
                            "energy_used": random.uniform(140, 160),
                            "cycle_time": random.uniform(45, 55),
                            "yield_rate": (output_amount / input_amount) * 100,
                        }
                    )
            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_complete_lca_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Generate a complete set of LCA-related test data.

        Returns:
            Dictionary containing all required LCA datasets
        """
        # Generate each type of data
        lca_data = {
            "material_flow": self.generate_lca_material_data(),
            "energy_consumption": self.generate_lca_energy_data(),
            "process_data": self.generate_lca_process_data(),
            "production_data": self.generate_production_data(),
        }

        return lca_data

    def generate_lca_material_data(self) -> pd.DataFrame:
        """Generate material flow data for LCA calculations."""
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            for hour in range(8, 16):  # 8-hour production day
                timestamp = current_date + timedelta(hours=hour)

                for line in self.production_lines:
                    batch_id = random.choice(self.batch_ids)
                    for material in self.material_types:
                        quantity = random.uniform(80, 120)  # Base quantity
                        waste = quantity * random.uniform(0.02, 0.08)  # 2-8% waste
                        recycled = waste * random.uniform(0.7, 0.9)  # 70-90% recycling

                        data.append(
                            {
                                "timestamp": timestamp,
                                "batch_id": batch_id,
                                "material_type": material,
                                "quantity_used": quantity,
                                "waste_generated": waste,
                                "recycled_amount": recycled,
                                "production_line": line,
                            }
                        )

            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_lca_energy_data(self) -> pd.DataFrame:
        """Generate energy consumption data for LCA calculations."""
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            for hour in range(8, 16):
                timestamp = current_date + timedelta(hours=hour)

                for line in self.production_lines:
                    for source in self.energy_sources:
                        data.append(
                            {
                                "timestamp": timestamp,
                                "batch_id": random.choice(self.batch_ids),
                                "energy_source": source,
                                "energy_consumption": random.uniform(40, 60),
                                "production_line": line,
                            }
                        )

            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_lca_process_data(self) -> pd.DataFrame:
        """Generate process-specific data for LCA calculations."""
        data = []
        current_date = self.start_date
        process_steps = [
            "wafer_cutting",
            "cell_processing",
            "module_assembly",
            "testing",
        ]

        for _ in range(self.days):
            for hour in range(8, 16):
                timestamp = current_date + timedelta(hours=hour)

                for line in self.production_lines:
                    batch_id = random.choice(self.batch_ids)
                    for step in process_steps:
                        data.append(
                            {
                                "timestamp": timestamp,
                                "batch_id": batch_id,
                                "process_step": step,
                                "process_time": random.uniform(45, 75),
                                "production_line": line,
                            }
                        )

            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_quality_data(self) -> pd.DataFrame:
        """Generate quality data with updated column names."""
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            for hour in range(8, 16):
                timestamp = current_date + timedelta(hours=hour)

                for _ in range(3):  # 3 quality checks per hour
                    data.append(
                        {
                            "batch_id": random.choice(self.batch_ids),
                            "test_timestamp": timestamp,
                            "efficiency": random.uniform(20, 22),
                            "defect_rate": random.uniform(1, 3),
                            "thickness_uniformity": random.uniform(94, 96),
                            "contamination_level": random.uniform(0.1, 0.5),
                        }
                    )
            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_energy_data(self) -> pd.DataFrame:
        """Generate energy consumption data."""
        production_data = self.generate_production_data()
        data = []

        for _, production_row in production_data.iterrows():
            data.append(
                {
                    "timestamp": production_row["timestamp"],
                    "production_line": production_row["production_line"],
                    "energy_consumption": random.uniform(40, 60),
                    "energy_source": random.choice(self.energy_sources),
                    "efficiency_rate": random.uniform(0.85, 0.95),
                }
            )

        return pd.DataFrame(data)

    def generate_material_flow_data(self) -> pd.DataFrame:
        """Generate material flow data."""
        production_data = self.generate_production_data()
        data = []

        for _, production_row in production_data.iterrows():
            data.append(
                {
                    "timestamp": production_row["timestamp"],
                    "batch_id": production_row["batch_id"],
                    "material_type": random.choice(self.material_types),
                    "quantity_used": random.uniform(900, 1100),
                    "waste_generated": random.uniform(10, 50),
                    "recycled_amount": random.uniform(5, 40),
                }
            )

        return pd.DataFrame(data)


# For backwards compatibility
TestDataGenerator = ManufacturingDataGenerator  # Alias for existing code
