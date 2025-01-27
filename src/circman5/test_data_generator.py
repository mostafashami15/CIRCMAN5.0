import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class TestDataGenerator:
    """
    Generates realistic test data for SoliTek's PV manufacturing analysis.
    Creates synthetic but plausible data for production, energy, quality, and material flow metrics.
    """

    def __init__(self, start_date: str = "2024-01-01", days: int = 30):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.days = days
        self.batch_ids = [f"BATCH_{i:04d}" for i in range(1, 101)]
        self.production_lines = ["LINE_A", "LINE_B", "LINE_C"]
        self.product_types = ["Mono_PERC_60", "Mono_PERC_72", "Bifacial_72"]
        self.material_types = ["Silicon_Wafer", "Glass", "EVA", "Backsheet", "Frame"]
        self.energy_sources = ["grid", "solar", "wind"]

    def generate_production_data(self) -> pd.DataFrame:
        """
        Generates production data with realistic patterns including:
        - Daily and weekly cycles
        - Random variations in output
        - Plausible yield rates and cycle times
        """
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            # Generate 8 batches per day (typical production shift)
            for hour in range(8, 16):  # 8 AM to 4 PM shift
                for line in self.production_lines:
                    timestamp = current_date + timedelta(hours=hour)

                    # Create realistic variations in production metrics
                    base_output = (
                        100 if line == "LINE_A" else 90 if line == "LINE_B" else 85
                    )
                    variation = random.uniform(-5, 5)

                    data.append(
                        {
                            "timestamp": timestamp,
                            "batch_id": random.choice(self.batch_ids),
                            "product_type": random.choice(self.product_types),
                            "production_line": line,
                            "output_quantity": max(0, base_output + variation),
                            "cycle_time": random.uniform(45, 55),  # minutes per batch
                            "yield_rate": random.uniform(
                                0.92, 0.98
                            ),  # typical PV manufacturing yield
                        }
                    )
            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_energy_data(self) -> pd.DataFrame:
        """
        Creates energy consumption data incorporating:
        - Time-of-day variations
        - Different energy sources
        - Realistic efficiency patterns
        """
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            for hour in range(24):
                timestamp = current_date + timedelta(hours=hour)

                # Create time-based energy consumption patterns
                base_consumption = (
                    100 if 8 <= hour <= 16 else 50
                )  # Higher during production hours

                for line in self.production_lines:
                    data.append(
                        {
                            "timestamp": timestamp,
                            "production_line": line,
                            "energy_consumption": base_consumption
                            * random.uniform(0.9, 1.1),
                            "energy_source": random.choice(self.energy_sources),
                            "efficiency_rate": random.uniform(0.85, 0.95),
                        }
                    )
            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_quality_data(self) -> pd.DataFrame:
        """
        Produces quality control data with:
        - Realistic efficiency distributions
        - Correlated quality metrics
        - Typical defect patterns
        """
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            for hour in range(8, 16):
                timestamp = current_date + timedelta(hours=hour)

                # Generate 3 quality checks per hour per line
                for line in self.production_lines:
                    for _ in range(3):
                        base_efficiency = random.uniform(
                            0.20, 0.22
                        )  # Typical PV cell efficiency

                        data.append(
                            {
                                "batch_id": random.choice(self.batch_ids),
                                "test_timestamp": timestamp,
                                "efficiency": base_efficiency,
                                "defect_rate": random.uniform(0.01, 0.03),
                                "thickness_uniformity": random.uniform(95, 99),
                                "visual_inspection": random.choice(
                                    ["Pass", "Minor_Issues", "Fail"]
                                ),
                            }
                        )
            current_date += timedelta(days=1)

        return pd.DataFrame(data)

    def generate_material_flow_data(self) -> pd.DataFrame:
        """
        Creates material flow data including:
        - Material consumption rates
        - Waste generation
        - Recycling metrics
        """
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            for hour in range(8, 16):
                timestamp = current_date + timedelta(hours=hour)

                for material in self.material_types:
                    # Base material quantities depend on type
                    base_quantity = {
                        "Silicon_Wafer": 1000,
                        "Glass": 800,
                        "EVA": 500,
                        "Backsheet": 400,
                        "Frame": 300,
                    }.get(material, 500)

                    quantity_used = base_quantity * random.uniform(0.9, 1.1)
                    waste_rate = random.uniform(0.02, 0.05)
                    waste_generated = quantity_used * waste_rate

                    data.append(
                        {
                            "timestamp": timestamp,
                            "material_type": material,
                            "quantity_used": quantity_used,
                            "waste_generated": waste_generated,
                            "recycled_amount": waste_generated
                            * random.uniform(0.7, 0.9),
                            "batch_id": random.choice(self.batch_ids),
                        }
                    )
            current_date += timedelta(days=1)

        return pd.DataFrame(data)


def main():
    """
    Demonstrates the test data generation and saves files for testing.
    """
    # Initialize the generator
    generator = TestDataGenerator(start_date="2024-01-01", days=30)

    # Generate all datasets
    production_data = generator.generate_production_data()
    energy_data = generator.generate_energy_data()
    quality_data = generator.generate_quality_data()
    material_data = generator.generate_material_flow_data()

    # Save to CSV files
    production_data.to_csv("test_production_data.csv", index=False)
    energy_data.to_csv("test_energy_data.csv", index=False)
    quality_data.to_csv("test_quality_data.csv", index=False)
    material_data.to_csv("test_material_data.csv", index=False)

    print("Test datasets generated successfully!")
    print(f"Production data shape: {production_data.shape}")
    print(f"Energy data shape: {energy_data.shape}")
    print(f"Quality data shape: {quality_data.shape}")
    print(f"Material flow data shape: {material_data.shape}")


if __name__ == "__main__":
    main()
