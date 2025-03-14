# src/circman5/test_data_generator.py

"""Test data generator for manufacturing analysis system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from circman5.utils.results_manager import results_manager

from circman5.manufacturing.lifecycle.impact_factors import (
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

    def generate_production_data(self):
        """Generate production data with all required columns."""
        data = []
        current_date = self.start_date

        stages = ["silicon_purification", "wafer_production", "cell_production"]

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
                            "stage": random.choice(stages),  # Added stage column
                            "product_type": random.choice(self.product_types),
                            "production_line": line,
                            "input_amount": input_amount,
                            "output_amount": output_amount,
                            "energy_used": random.uniform(140, 160),
                            "cycle_time": random.uniform(45, 55),
                            "yield_rate": (output_amount / input_amount) * 100,
                            "status": "completed",  # Added status column
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
        """Generate energy consumption data with all required fields."""
        data = []
        current_date = self.start_date

        for _ in range(self.days):
            for hour in range(8, 16):
                for line in self.production_lines:
                    timestamp = current_date + timedelta(hours=hour)
                    for source in self.energy_sources:
                        data.append(
                            {
                                "timestamp": timestamp,
                                "batch_id": random.choice(self.batch_ids),
                                "energy_source": source,
                                "energy_consumption": random.uniform(40, 60),
                                "production_line": line,
                                "efficiency_rate": random.uniform(
                                    0.80, 0.95
                                ),  # Added this field
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
                            "timestamp": timestamp,
                            "efficiency": random.uniform(20, 22),
                            "defect_rate": random.uniform(1, 3),
                            "thickness_uniformity": random.uniform(94, 96),
                            "contamination_level": random.uniform(0.1, 0.5),
                            "visual_inspection": "pass",
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
        """Generate material flow data with realistic waste and recycling ratios."""
        production_data = self.generate_production_data()
        data = []

        for _, production_row in production_data.iterrows():
            # Generate base quantity
            quantity_used = random.uniform(900, 1100)

            # Generate waste as percentage of quantity used (5-15%)
            waste_ratio = random.uniform(0.05, 0.15)
            waste_generated = quantity_used * waste_ratio

            # Generate recycled amount as percentage of waste (60-90%)
            recycling_ratio = random.uniform(0.60, 0.90)
            recycled_amount = waste_generated * recycling_ratio

            data.append(
                {
                    "timestamp": production_row["timestamp"],
                    "batch_id": production_row["batch_id"],
                    "material_type": random.choice(self.material_types),
                    "quantity_used": quantity_used,
                    "waste_generated": waste_generated,
                    "recycled_amount": recycled_amount,
                }
            )

        return pd.DataFrame(data)

    def save_generated_data(self) -> None:
        """Save all generated test data to the synthetic data directory."""
        try:
            # Get temporary directory from results_manager
            temp_dir = results_manager.get_path("temp")

            # Define all files to save
            data_files = {
                "test_energy_data.csv": self.generate_energy_data(),
                "test_material_data.csv": self.generate_material_flow_data(),
                "test_process_data.csv": self.generate_lca_process_data(),
                "test_production_data.csv": self.generate_production_data(),
            }

            # Save each file
            for filename, data in data_files.items():
                # Save to temp directory first
                temp_path = temp_dir / filename
                data.to_csv(temp_path, index=False)

                # Save to synthetic data directory
                results_manager.save_to_path(temp_path, "SYNTHETIC_DATA")

                # Clean up temporary file
                temp_path.unlink()

        except Exception as e:
            raise IOError(f"Error saving test data: {str(e)}")

    def generate_time_series_data(self, days=30, interval_minutes=15):
        """Generate time series manufacturing data."""
        total_points = int((days * 24 * 60) / interval_minutes)
        timestamps = [
            self.start_date + timedelta(minutes=i * interval_minutes)
            for i in range(total_points)
        ]

        # Create base patterns with daily cycles
        base_pattern = np.sin(np.linspace(0, days * 2 * np.pi, total_points))

        # Add noise and trends
        noise = np.random.normal(0, 0.1, total_points)
        trend = np.linspace(0, 0.5, total_points)

        # Create data with realistic patterns
        data = {
            "timestamp": timestamps,
            "input_amount": 100 + 10 * base_pattern + 5 * noise + trend * 10,
            "energy_used": 150 + 15 * base_pattern + 8 * noise + trend * 5,
            "cycle_time": 45
            + 5 * np.sin(np.linspace(0, days * 4 * np.pi, total_points))
            + 2 * noise,
            "temperature": 175
            + 15 * np.sin(np.linspace(0, days * 1 * np.pi, total_points))
            + 3 * noise,
        }

        # Add derived metrics with realistic relationships
        df = pd.DataFrame(data)
        df["efficiency"] = (
            18
            + 0.02 * (df["temperature"] - 175)
            - 0.003 * df["energy_used"]
            + np.random.normal(0, 0.5, total_points)
        )
        df["defect_rate"] = (
            5
            - 0.01 * df["efficiency"]
            + 0.02 * np.abs(df["temperature"] - 175)
            + np.random.normal(0, 0.3, total_points)
        )
        df["output_amount"] = df["input_amount"] * (
            1 - df["defect_rate"] / 100
        ) + np.random.normal(0, 1, total_points)

        # Add batch IDs
        df["batch_id"] = [f"BATCH_{i//4:04d}" for i in range(total_points)]

        return df

    def generate_realistic_time_series(
        self,
        duration_days: int = 30,
        interval_minutes: int = 15,
        include_anomalies: bool = True,
    ) -> pd.DataFrame:
        """
        Generate manufacturing time series data with realistic patterns.

        Args:
            duration_days: Number of days to generate data for
            interval_minutes: Sampling interval in minutes
            include_anomalies: Whether to include anomalies

        Returns:
            DataFrame with time series data
        """
        # Calculate number of samples
        samples = int((duration_days * 24 * 60) / interval_minutes)

        # Create timestamp series
        start_time = self.start_date
        timestamps = [
            start_time + timedelta(minutes=i * interval_minutes) for i in range(samples)
        ]

        # Base parameters
        temperature_base = 25.0  # degrees Celsius
        power_consumption_base = 50.0  # kW
        efficiency_base = 0.85  # efficiency
        defect_rate_base = 0.05  # defect rate

        # Create patterns
        # Daily temperature cycle
        day_cycle = np.sin(np.linspace(0, 2 * np.pi * duration_days, samples))
        # Weekly production cycle (5 day work week, 2 day reduced)
        week_hours = np.array(
            [(t.weekday() < 5 and t.hour >= 6 and t.hour < 22) for t in timestamps]
        )
        week_hours = week_hours.astype(float)

        # Create maintenance cycle (every 7 days)
        maintenance_cycle = np.ones(samples)
        for i in range(samples):
            day_num = (timestamps[i] - start_time).days
            days_since_maintenance = day_num % 7
            # Efficiency degrades after maintenance, then improves after maintenance
            maintenance_cycle[i] = 1.0 - 0.1 * (days_since_maintenance / 7)

        # Generate data
        data = []

        for i in range(samples):
            # Time-dependent factors
            hour_of_day = timestamps[i].hour
            is_working_hours = hour_of_day >= 6 and hour_of_day < 22

            # Calculate parameters with realistic correlations
            temperature = (
                temperature_base + 5 * day_cycle[i] + random.normalvariate(0, 0.5)
            )

            # Efficiency follows maintenance cycle and is affected by temperature
            efficiency = (
                efficiency_base
                * maintenance_cycle[i]
                * (1 - 0.01 * max(0, temperature - 25))
            )
            efficiency = max(0.7, min(0.95, efficiency))  # Bound to realistic range

            # Power consumption depends on production and temperature
            power_factor = week_hours[i] * (0.8 + 0.4 * (temperature - 20) / 10)
            power_consumption = (
                power_consumption_base * power_factor + random.normalvariate(0, 2.0)
            )
            power_consumption = max(5.0, power_consumption)  # Minimum standby power

            # Defect rate is inversely related to efficiency
            defect_rate = (
                defect_rate_base
                * (efficiency_base / efficiency)
                * (1 + 0.2 * (temperature - 25) / 10)
            )
            defect_rate = max(0.01, min(0.2, defect_rate))  # Bound to realistic range

            # Production amount depends on time of day and week
            input_amount = 100.0 * week_hours[i] + random.normalvariate(0, 5)
            input_amount = max(0, input_amount)

            # Output is correlated with input, efficiency and defect rate
            output_amount = input_amount * efficiency * (1 - defect_rate)

            # Add anomalies randomly
            anomaly_present = False
            anomaly_type = "none"

            if (
                include_anomalies and is_working_hours and random.random() < 0.005
            ):  # 0.5% chance during working hours
                anomaly_present = True
                anomaly_selection = random.randint(0, 3)

                if anomaly_selection == 0:
                    # Power spike
                    power_consumption *= 1.5 + random.random()
                    anomaly_type = "power_spike"
                elif anomaly_selection == 1:
                    # Overheating
                    temperature += 15 * random.random()
                    efficiency *= 0.7  # Efficiency drops with overheating
                    anomaly_type = "overheating"
                elif anomaly_selection == 2:
                    # Material quality issue
                    defect_rate *= 2 + random.random()
                    output_amount = input_amount * efficiency * (1 - defect_rate)
                    anomaly_type = "material_issue"
                elif anomaly_selection == 3:
                    # Equipment failure
                    efficiency *= 0.5
                    output_amount = input_amount * efficiency * (1 - defect_rate)
                    anomaly_type = "equipment_failure"

            # Create record
            data.append(
                {
                    "timestamp": timestamps[i],
                    "temperature": temperature,
                    "power_consumption": power_consumption,
                    "efficiency": efficiency,
                    "defect_rate": defect_rate,
                    "input_amount": input_amount,
                    "output_amount": output_amount,
                    "is_working_hours": is_working_hours,
                    "anomaly_present": anomaly_present,
                    "anomaly_type": anomaly_type,
                }
            )

        # Convert to DataFrame
        return pd.DataFrame(data)

    def generate_edge_cases(self, num_cases: int = 20) -> pd.DataFrame:
        """
        Generate edge cases for testing system robustness.

        Args:
            num_cases: Number of edge cases to generate

        Returns:
            DataFrame with edge cases
        """
        # Define edge case types
        edge_case_types = [
            "extreme_input",
            "zero_energy",
            "minimum_cycle_time",
            "maximum_defect_rate",
            "power_outage",
            "material_shortage",
            "perfect_quality",
            "rapid_degradation",
            "gradual_improvement",
            "process_interruption",
        ]

        # Generate a mix of edge cases
        data = []

        for i in range(num_cases):
            case_type = edge_case_types[i % len(edge_case_types)]
            base_case = {
                "case_id": f"EDGE_{i:03d}",
                "case_type": case_type,
                "input_amount": 100.0,
                "energy_used": 500.0,
                "cycle_time": 45.0,
                "efficiency": 0.85,
                "defect_rate": 0.05,
                "thickness_uniformity": 0.92,
                "expected_output": 0.0,  # Will be calculated below
                "description": "",
            }

            # Modify parameters based on edge case type
            if case_type == "extreme_input":
                base_case["input_amount"] = 500.0 * (1 + random.random())
                base_case["energy_used"] = 2000.0 * (1 + 0.5 * random.random())
                base_case["description"] = "Extremely high input material amount"

            elif case_type == "zero_energy":
                base_case["energy_used"] = 0.001
                base_case["efficiency"] = 0.01
                base_case["description"] = "Near-zero energy consumption (system error)"

            elif case_type == "minimum_cycle_time":
                base_case["cycle_time"] = 5.0 * random.random()
                base_case["description"] = "Unrealistically short cycle time"

            elif case_type == "maximum_defect_rate":
                base_case["defect_rate"] = 0.8 + 0.2 * random.random()
                base_case["description"] = "Extremely high defect rate"

            elif case_type == "power_outage":
                base_case["energy_used"] = 50.0
                base_case["efficiency"] = 0.2
                base_case["cycle_time"] = 120.0
                base_case[
                    "description"
                ] = "Simulation of power outage during production"

            elif case_type == "material_shortage":
                base_case["input_amount"] = 10.0 * random.random()
                base_case["description"] = "Material shortage scenario"

            elif case_type == "perfect_quality":
                base_case["efficiency"] = 0.999
                base_case["defect_rate"] = 0.001
                base_case["thickness_uniformity"] = 0.999
                base_case[
                    "description"
                ] = "Perfect quality scenario (unlikely in real production)"

            elif case_type == "rapid_degradation":
                base_case["efficiency"] = 0.4
                base_case["defect_rate"] = 0.4
                base_case["description"] = "Rapid quality degradation"

            elif case_type == "gradual_improvement":
                base_case["efficiency"] = 0.95
                base_case["defect_rate"] = 0.01
                base_case[
                    "description"
                ] = "Gradual quality improvement after maintenance"

            elif case_type == "process_interruption":
                base_case["cycle_time"] = 180.0
                base_case["efficiency"] = 0.5
                base_case["description"] = "Process interruption scenario"

            # Calculate expected output based on parameters
            base_case["expected_output"] = (
                base_case["input_amount"]
                * base_case["efficiency"]
                * (1 - base_case["defect_rate"])
            )

            data.append(base_case)

        # Convert to DataFrame
        return pd.DataFrame(data)


def generate_complete_test_dataset(
    production_batches: int = 100, time_series_days: int = 7, edge_cases: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Generate a comprehensive test dataset for AI/ML training and testing.

    Args:
        production_batches: Number of production batches
        time_series_days: Days of time series data
        edge_cases: Number of edge cases

    Returns:
        Dictionary of DataFrames with different data types
    """
    generator = ManufacturingDataGenerator()

    # Generate standard production data
    production_df = generator.generate_production_data()
    quality_df = generator.generate_quality_data()

    # Generate time series data
    time_series_df = generator.generate_realistic_time_series(
        duration_days=time_series_days, interval_minutes=30, include_anomalies=True
    )

    # Generate edge cases
    edge_cases_df = generator.generate_edge_cases(num_cases=edge_cases)

    # Save data to CSV files
    production_df.to_csv("production_data.csv", index=False)
    quality_df.to_csv("quality_data.csv", index=False)
    time_series_df.to_csv("time_series_data.csv", index=False)
    edge_cases_df.to_csv("edge_cases.csv", index=False)

    # Get results_manager path
    synthetic_dir = results_manager.get_path("SYNTHETIC_DATA")

    # Move files to results directory
    for file in [
        "production_data.csv",
        "quality_data.csv",
        "time_series_data.csv",
        "edge_cases.csv",
    ]:
        results_manager.save_to_path(Path(file), "SYNTHETIC_DATA")
        # Clean up
        Path(file).unlink()

    return {
        "production": production_df,
        "quality": quality_df,
        "time_series": time_series_df,
        "edge_cases": edge_cases_df,
    }


# For backwards compatibility
TestDataGenerator = ManufacturingDataGenerator  # Alias for existing code
