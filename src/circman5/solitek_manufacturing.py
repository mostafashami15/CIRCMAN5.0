import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List


class ValidationError(Exception):
    """Custom exception for data validation errors"""

    pass


class SoliTekManufacturingAnalysis:
    """
    A comprehensive framework for analyzing SoliTek's PV manufacturing data.
    This class implements the methodology outlined in our case study framework,
    focusing on production efficiency, sustainability metrics, and AI-driven optimization.
    """

    def __init__(self):
        # Initialize core data structures for different metric categories
        self.production_data = pd.DataFrame()
        self.energy_data = pd.DataFrame()
        self.quality_data = pd.DataFrame()
        self.material_flow = pd.DataFrame()
        self.sustainability_metrics = pd.DataFrame()

        # Define expected data schemas
        self.production_schema = {
            "timestamp": "datetime64[ns]",
            "batch_id": str,
            "product_type": str,
            "production_line": str,
            "output_quantity": float,
            "cycle_time": float,
            "yield_rate": float,
        }

        self.energy_schema = {
            "timestamp": "datetime64[ns]",
            "production_line": str,
            "energy_consumption": float,
            "energy_source": str,
            "efficiency_rate": float,
        }

        self.quality_schema = {
            "batch_id": str,
            "test_timestamp": "datetime64[ns]",
            "efficiency": float,
            "defect_rate": float,
            "thickness_uniformity": float,
            "visual_inspection": str,
        }

        self.material_schema = {
            "timestamp": "datetime64[ns]",
            "material_type": str,
            "quantity_used": float,
            "waste_generated": float,
            "recycled_amount": float,
            "batch_id": str,
        }

    def validate_production_data(self, data: pd.DataFrame) -> bool:
        """
        Validates production data against required schema and business rules.

        Args:
            data: DataFrame containing production data

        Returns:
            bool: True if data is valid, raises ValidationError otherwise
        """
        required_columns = {
            "batch_id": str,
            "timestamp": "datetime64[ns]",
            "stage": str,
            "input_amount": float,
            "output_amount": float,
            "energy_used": float,
        }

        # Check required columns exist
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")

        # Validate data types
        for col, dtype in required_columns.items():
            if not pd.api.types.is_dtype_equal(data[col].dtype, dtype):
                try:
                    data[col] = data[col].astype(dtype)
                except Exception as e:
                    raise ValidationError(f"Invalid data type for {col}: {str(e)}")

        # Business rules validation
        if (data["input_amount"] < 0).any():
            raise ValidationError("Input amounts cannot be negative")

        if (data["output_amount"] < 0).any():
            raise ValidationError("Output amounts cannot be negative")

        if (data["output_amount"] > data["input_amount"]).any():
            raise ValidationError("Output amount cannot exceed input amount")

        return True

    def load_production_data(self, file_path: str) -> None:
        """
        Load and validate production data from CSV files.
        Implements error checking and data validation.
        """
        try:
            data = pd.read_csv(file_path)
            # Validate against schema
            for column, dtype in self.production_schema.items():
                if column not in data.columns:
                    raise ValueError(f"Missing required column: {column}")
                data[column] = data[column].astype(dtype)

            self.production_data = data
            print("Production data loaded successfully")
        except Exception as e:
            print(f"Error loading production data: {str(e)}")

    def analyze_efficiency(self) -> Dict:
        """
        Analyze manufacturing efficiency metrics including:
        - Production yield rates
        - Cycle time optimization
        - Resource utilization
        """
        if self.production_data.empty:
            return {"error": "No production data available"}

        analysis = {
            "average_yield": self.production_data["yield_rate"].mean(),
            "cycle_time_stats": self.production_data["cycle_time"].describe(),
            "daily_output": self.production_data.groupby(
                pd.Grouper(key="timestamp", freq="D")
            )["output_quantity"].sum(),
        }

        return analysis

    def calculate_sustainability_metrics(self) -> Dict:
        """
        Calculate key sustainability indicators including:
        - Energy efficiency
        - Material utilization
        - Waste reduction
        - Carbon footprint
        """
        if any([self.energy_data.empty, self.material_flow.empty]):
            return {"error": "Insufficient data for sustainability calculation"}

        metrics = {
            "energy_efficiency": self._calculate_energy_efficiency(),
            "material_utilization": self._calculate_material_utilization(),
            "waste_reduction": self._calculate_waste_metrics(),
            "carbon_footprint": self._estimate_carbon_footprint(),
        }

        return metrics

    def analyze_quality_metrics(self) -> Dict:
        """
        Analyze quality control data to identify:
        - Defect patterns
        - Quality trends
        - Process optimization opportunities
        """
        if self.quality_data.empty:
            return {"error": "No quality data available"}

        analysis = {
            "average_efficiency": self.quality_data["efficiency"].mean(),
            "defect_rate_trend": self.quality_data.groupby(
                pd.Grouper(key="test_timestamp", freq="D")
            )["defect_rate"].mean(),
            "uniformity_stats": self.quality_data["thickness_uniformity"].describe(),
        }

        return analysis

    def generate_visualization(
        self, metric_type: str, save_path: Optional[str] = None
    ) -> None:
        """
        Create visualizations for different metric types:
        - Production trends
        - Energy consumption patterns
        - Quality metrics
        - Sustainability indicators
        """
        plt.figure(figsize=(12, 6))

        if metric_type == "production":
            self._visualize_production_trends()
        elif metric_type == "energy":
            self._visualize_energy_patterns()
        elif metric_type == "quality":
            self._visualize_quality_metrics()
        elif metric_type == "sustainability":
            self._visualize_sustainability_indicators()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _visualize_production_trends(self):
        """Visualize production efficiency and output trends"""
        if self.production_data.empty:
            return

        daily_output = self.production_data.groupby(
            pd.Grouper(key="timestamp", freq="D")
        )["output_quantity"].sum()

        plt.subplot(2, 1, 1)
        daily_output.plot(style=".-")
        plt.title("Daily Production Output")
        plt.ylabel("Output Quantity")

        plt.subplot(2, 1, 2)
        self.production_data["yield_rate"].plot(kind="hist", bins=20)
        plt.title("Yield Rate Distribution")
        plt.xlabel("Yield Rate")
        plt.ylabel("Frequency")

    def _visualize_energy_patterns(self):
        """Visualize energy consumption patterns"""
        if self.energy_data.empty:
            return

        energy_by_source = self.energy_data.groupby("energy_source")[
            "energy_consumption"
        ].sum()
        plt.subplot(2, 1, 1)
        energy_by_source.plot(kind="pie", autopct="%1.1f%%")
        plt.title("Energy Consumption by Source")

        hourly_consumption = self.energy_data.groupby(
            self.energy_data["timestamp"].dt.hour
        )["energy_consumption"].mean()
        plt.subplot(2, 1, 2)
        hourly_consumption.plot(style=".-")
        plt.title("Average Hourly Energy Consumption")
        plt.xlabel("Hour of Day")
        plt.ylabel("Energy Consumption")

    def _visualize_quality_metrics(self):
        """Visualize quality control metrics"""
        if self.quality_data.empty:
            return

        plt.subplot(2, 1, 1)
        self.quality_data["efficiency"].plot(kind="hist", bins=20)
        plt.title("Cell Efficiency Distribution")
        plt.xlabel("Efficiency (%)")

        plt.subplot(2, 1, 2)
        defect_rates = self.quality_data.groupby(
            pd.Grouper(key="test_timestamp", freq="D")
        )["defect_rate"].mean()
        defect_rates.plot(style=".-")
        plt.title("Daily Average Defect Rate")
        plt.ylabel("Defect Rate (%)")

    def _visualize_sustainability_indicators(self):
        """Visualize sustainability metrics"""
        if self.material_flow.empty:
            return

        waste_by_type = self.material_flow.groupby("material_type")[
            "waste_generated"
        ].sum()
        plt.subplot(2, 1, 1)
        waste_by_type.plot(kind="bar")
        plt.title("Total Waste Generation by Material Type")
        plt.xticks(rotation=45)

        recycling_rates = self.material_flow.groupby("material_type").agg(
            {"recycled_amount": "sum", "waste_generated": "sum"}
        )
        recycling_rates["rate"] = (
            recycling_rates["recycled_amount"]
            / recycling_rates["waste_generated"]
            * 100
        )

        plt.subplot(2, 1, 2)
        recycling_rates["rate"].plot(kind="bar")
        plt.title("Recycling Rate by Material Type")
        plt.ylabel("Recycling Rate (%)")
        plt.xticks(rotation=45)

    def _calculate_energy_efficiency(self) -> float:
        """Calculate energy efficiency metrics"""
        if self.energy_data.empty:
            return 0.0
        return (
            self.energy_data["efficiency_rate"] * self.energy_data["energy_consumption"]
        ).mean()

    def _calculate_material_utilization(self) -> float:
        """Calculate material utilization rate"""
        if self.material_flow.empty:
            return 0.0
        total_used = self.material_flow["quantity_used"].sum()
        total_waste = self.material_flow["waste_generated"].sum()
        return (1 - (total_waste / total_used)) * 100 if total_used > 0 else 0.0

    def _calculate_waste_metrics(self) -> Dict:
        """Calculate waste-related metrics"""
        if self.material_flow.empty:
            return {}

        return {
            "total_waste": self.material_flow["waste_generated"].sum(),
            "recycling_rate": (
                self.material_flow["recycled_amount"].sum()
                / self.material_flow["waste_generated"].sum()
                * 100
            ),
            "waste_by_type": self.material_flow.groupby("material_type")[
                "waste_generated"
            ].sum(),
        }

    def _estimate_carbon_footprint(self) -> float:
        """Estimate carbon footprint based on energy consumption"""
        if self.energy_data.empty:
            return 0.0

        # Example carbon intensity factors (kg CO2/kWh)
        carbon_factors = {"grid": 0.5, "solar": 0.0, "wind": 0.0}

        carbon_footprint = 0.0
        for source in self.energy_data["energy_source"].unique():
            source_consumption = self.energy_data[
                self.energy_data["energy_source"] == source
            ]["energy_consumption"].sum()
            carbon_footprint += source_consumption * carbon_factors.get(source, 0.5)

        return carbon_footprint

    def export_analysis_report(self, output_path: str) -> None:
        """
        Generate a comprehensive analysis report including:
        - Production efficiency analysis
        - Quality metrics
        - Sustainability indicators
        - Recommendations for optimization
        """
        report_data = {
            "production_metrics": self.analyze_efficiency(),
            "quality_analysis": self.analyze_quality_metrics(),
            "sustainability_metrics": self.calculate_sustainability_metrics(),
        }

        # Export to Excel with multiple sheets
        with pd.ExcelWriter(output_path) as writer:
            for metric, data in report_data.items():
                if isinstance(data, dict) and not any(
                    key == "error" for key in data.keys()
                ):
                    pd.DataFrame(data).to_excel(writer, sheet_name=metric)


def main():
    """
    Example usage of the SoliTek Manufacturing Analysis framework
    """
    analyzer = SoliTekManufacturingAnalysis()

    # Example data loading (once we receive actual data)
    # analyzer.load_production_data("production_data.csv")

    # Generate analysis and visualizations
    efficiency_metrics = analyzer.analyze_efficiency()
    sustainability_metrics = analyzer.calculate_sustainability_metrics()
    quality_analysis = analyzer.analyze_quality_metrics()

    # Export results
    analyzer.export_analysis_report("solitek_analysis_report.xlsx")


if __name__ == "__main__":
    main()
