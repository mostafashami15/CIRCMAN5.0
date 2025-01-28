"""Test data generator for AI optimization testing."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class TestDataGenerator:
    """Generates synthetic manufacturing data for testing."""

    def __init__(self, start_date: str = "2024-01-01", days: int = 30):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.days = days
        self.batch_ids = [f"BATCH_{i:04d}" for i in range(1, 101)]
        self.production_lines = ["LINE_A", "LINE_B", "LINE_C"]
        self.product_types = ["Mono_PERC_60", "Mono_PERC_72", "Bifacial_72"]

    def generate_production_data(self) -> pd.DataFrame:
        """Generate production metrics data."""
        data = []
        current_date = self.start_date
        batch_id = 0

        for _ in range(self.days):
            for hour in range(8, 16):  # 8-hour production day
                for line in self.production_lines:
                    batch_id = (batch_id + 1) % len(self.batch_ids)

                    data.append(
                        {
                            "timestamp": current_date + timedelta(hours=hour),
                            "batch_id": self.batch_ids[batch_id],
                            "product_type": random.choice(self.product_types),
                            "production_line": line,
                            "output_quantity": random.uniform(85, 95),
                            "cycle_time": random.uniform(45, 55),
                            "yield_rate": random.uniform(0.92, 0.98),
                        }
                    )
            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_quality_data(self) -> pd.DataFrame:
        """Generate quality control data."""
        production_data = self.generate_production_data()
        data = []

        for _, production_row in production_data.iterrows():
            data.append(
                {
                    "batch_id": production_row["batch_id"],
                    "test_time": production_row["timestamp"],
                    "efficiency": random.uniform(20, 22),
                    "defect_rate": random.uniform(1, 3),
                    "thickness_uniformity": random.uniform(95, 99),
                    "contamination_level": random.uniform(0.1, 0.5),
                }
            )

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
                    "energy_source": random.choice(["grid", "solar", "wind"]),
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
                    "material_type": random.choice(
                        ["Silicon", "Glass", "EVA", "Frame"]
                    ),
                    "quantity_used": random.uniform(900, 1100),
                    "waste_generated": random.uniform(10, 50),
                    "recycled_amount": random.uniform(5, 40),
                }
            )

        return pd.DataFrame(data)
