import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List

from .ai.optimization import ManufacturingOptimizer, MetricsDict, PredictionDict
from .logging_config import setup_logger
from .errors import ValidationError, ProcessError, DataError, ResourceError
from .errors import ValidationError, DataError, ProcessError
from .monitoring import ManufacturingMonitor
from .visualization import ManufacturingVisualizer
from .analysis.efficiency import EfficiencyAnalyzer
from .analysis.quality import QualityAnalyzer
from .analysis.sustainability import SustainabilityAnalyzer


class SoliTekManufacturingAnalysis:
    """
    A comprehensive framework for analyzing SoliTek's PV manufacturing data.
    This class implements the methodology outlined in our case study framework,
    focusing on production efficiency, sustainability metrics, and AI-driven optimization.
    """

    def __init__(self):
        # logger initialization
        self.logger = setup_logger("solitek_manufacturing")

        # Initialize core data structures for different metric categories
        self.production_data = pd.DataFrame()
        self.energy_data = pd.DataFrame()
        self.quality_data = pd.DataFrame()
        self.material_flow = pd.DataFrame()
        self.sustainability_metrics = pd.DataFrame()

        # Analysis tools
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
        self.sustainability_analyzer = SustainabilityAnalyzer()

        # AI optimizer
        self.optimizer = ManufacturingOptimizer()
        self.is_optimizer_trained = False

        # Data monitoring
        self.monitor = ManufacturingMonitor()
        self.visualizer = ManufacturingVisualizer()

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

    def train_optimization_model(self) -> MetricsDict:
        """Train the AI optimization model with current manufacturing data."""
        if self.production_data.empty or self.quality_data.empty:
            raise ValueError("No data available for training optimization model")

        # Prepare data for training
        X_scaled, y_scaled = self.optimizer.prepare_manufacturing_data(
            self.production_data, self.quality_data
        )

        # Train the model
        metrics = self.optimizer.train_optimization_models(X_scaled, y_scaled)
        self.is_optimizer_trained = True

        self.logger.info(f"AI optimization model trained. Metrics: {metrics}")
        return metrics

    def optimize_process_parameters(
        self,
        current_params: Dict[str, float],
        constraints: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        Optimize manufacturing process parameters using AI.
        """
        if not self.is_optimizer_trained:
            self.train_optimization_model()

        optimized_params = self.optimizer.optimize_process_parameters(
            current_params, constraints
        )

        self.logger.info(
            f"Optimized parameters generated:\n"
            + "\n".join(f"{k}: {v:.2f}" for k, v in optimized_params.items())
        )

        return optimized_params

    def predict_batch_outcomes(
        self, process_params: Dict[str, float]
    ) -> PredictionDict:
        """Predict manufacturing outcomes for given parameters."""

        if not self.is_optimizer_trained:
            self.train_optimization_model()

        predictions = self.optimizer.predict_manufacturing_outcomes(process_params)

        self.logger.info(
            f"Manufacturing predictions:\n"
            + f"Predicted Output: {predictions['predicted_output']:.2f}\n"
            + f"Predicted Quality: {predictions['predicted_quality']:.2f}"
        )

        return predictions

    def analyze_optimization_potential(self) -> Dict[str, float]:
        """Analyze potential optimizations based on historical data."""
        if not self.is_optimizer_trained:
            self.train_optimization_model()

        # Get average current parameters
        current_params = {
            "input_amount": self.production_data["input_amount"].mean(),
            "energy_used": self.production_data["energy_used"].mean(),
            "cycle_time": self.production_data["cycle_time"].mean(),
            "efficiency": self.quality_data["efficiency"].mean(),
            "defect_rate": self.quality_data["defect_rate"].mean(),
            "thickness_uniformity": self.quality_data["thickness_uniformity"].mean(),
        }

        # Get optimized parameters
        optimized_params = self.optimize_process_parameters(current_params)

        # Calculate potential improvements
        improvements = {
            param: (
                (optimized_params[param] - current_params[param])
                / current_params[param]
                * 100
            )
            for param in current_params
        }

        self.logger.info(
            "Optimization potential analysis:\n"
            + "\n".join(f"{k}: {v:.1f}%" for k, v in improvements.items())
        )

        return improvements

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
            raise ValidationError(
                f"Missing required columns: {missing_cols}",
                invalid_data={"missing_columns": missing_cols},
            )

        # Validate data types
        for col, dtype in required_columns.items():
            if not pd.api.types.is_dtype_equal(data[col].dtype, dtype):
                try:
                    data[col] = data[col].astype(dtype)
                except Exception as e:
                    raise ValidationError(
                        f"Invalid data type for {col}: {str(e)}",
                        invalid_data={"column": col, "error": str(e)},
                    )

        # Business rules validation
        if (data["input_amount"] < 0).any():
            raise ValidationError(
                "Input amounts cannot be negative",
                invalid_data={"field": "input_amount"},
            )

        if (data["output_amount"] < 0).any():
            raise ValidationError(
                "Output amounts cannot be negative",
                invalid_data={"field": "output_amount"},
            )

        if (data["output_amount"] > data["input_amount"]).any():
            raise ValidationError(
                "Output amount cannot exceed input amount",
                invalid_data={"field": "output_amount"},
            )

        return True

    def load_production_data(self, file_path: str) -> None:
        """
        Load and validate production data from CSV files.
        Implements enhanced error checking and data validation with logging.

        Args:
            file_path: Path to the CSV file containing production data

        Raises:
            DataError: If file doesn't exist or is empty
            ValidationError: If data fails validation checks
            ProcessError: If file cannot be parsed or processed
        """
        try:
            self.logger.info(f"Loading production data from {file_path}")

            # Check if file exists
            if not os.path.exists(file_path):
                raise DataError(
                    f"Production data file not found: {file_path}",
                    data_source=file_path,
                )

            # Load data
            try:
                data = pd.read_csv(file_path)
            except pd.errors.ParserError as pe:
                raise ProcessError(
                    f"Error parsing CSV file: {str(pe)}", process_name="data_loading"
                )

            # Check if data is empty
            if data.empty:
                raise DataError("Production data file is empty", data_source=file_path)

            # Validate data
            if self.validate_production_data(data):
                self.production_data = data
                self.logger.info(
                    f"Successfully loaded and validated {len(data)} records from {file_path}"
                )

        except (DataError, ValidationError, ProcessError) as e:
            self.logger.error(f"{e.__class__.__name__}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading production data: {str(e)}")
            raise ProcessError(
                f"Unexpected error during data loading: {str(e)}",
                process_name="data_loading",
            )

    def analyze_efficiency(self) -> Dict:
        """
        Enhanced efficiency analysis using dedicated analyzer.
        """
        if self.production_data.empty:
            self.logger.warning("No production data available for efficiency analysis")
            return {"error": "No production data available"}

        try:
            efficiency_metrics = self.efficiency_analyzer.analyze_batch_efficiency(
                self.production_data
            )

            # Calculate additional metrics
            analysis = {
                **efficiency_metrics,
                "daily_output": self.production_data.groupby(
                    pd.Grouper(key="timestamp", freq="D")
                )["output_quantity"].sum(),
            }

            self.logger.info(f"Efficiency analysis completed: {efficiency_metrics}")
            return analysis

        except Exception as e:
            self.logger.error(f"Error in efficiency analysis: {str(e)}")
            return {"error": str(e)}

    def calculate_sustainability_metrics(self) -> Dict:
        """
        Enhanced sustainability analysis using dedicated analyzer.
        """
        if any([self.energy_data.empty, self.material_flow.empty]):
            self.logger.warning("Insufficient data for sustainability calculation")
            return {"error": "Insufficient data for sustainability calculation"}

        try:
            # Calculate carbon footprint
            carbon_footprint = self.sustainability_analyzer.calculate_carbon_footprint(
                self.energy_data
            )

            # Analyze material efficiency
            material_metrics = self.sustainability_analyzer.analyze_material_efficiency(
                self.material_flow
            )

            # Calculate overall sustainability score
            sustainability_score = (
                self.sustainability_analyzer.calculate_sustainability_score(
                    material_metrics.get("material_efficiency", 0),
                    material_metrics.get("recycling_rate", 0),
                    material_metrics.get(
                        "material_efficiency", 0
                    ),  # Using material efficiency as energy efficiency proxy
                )
            )

            metrics = {
                "carbon_footprint": carbon_footprint,
                "sustainability_score": sustainability_score,
                **material_metrics,
            }

            self.logger.info(f"Sustainability analysis completed: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error in sustainability analysis: {str(e)}")
            return {"error": str(e)}

    def analyze_quality_metrics(self) -> Dict:
        """
        Enhanced quality analysis using dedicated analyzer.
        """
        if self.quality_data.empty:
            self.logger.warning("No quality data available for analysis")
            return {"error": "No quality data available"}

        try:
            # Basic quality metrics
            quality_metrics = self.quality_analyzer.analyze_defect_rates(
                self.quality_data
            )

            # Quality trends
            trends = self.quality_analyzer.identify_quality_trends(self.quality_data)

            analysis = {**quality_metrics, "trends": trends}

            self.logger.info(f"Quality analysis completed: {quality_metrics}")
            return analysis

        except Exception as e:
            self.logger.error(f"Error in quality analysis: {str(e)}")
            return {"error": str(e)}

    def generate_comprehensive_report(self, output_path: str) -> None:
        """
        Generate a comprehensive analysis report including all metrics.
        """
        try:
            # Collect all metrics
            report_data = {
                "efficiency_metrics": self.analyze_efficiency(),
                "quality_metrics": self.analyze_quality_metrics(),
                "sustainability_metrics": self.calculate_sustainability_metrics(),
            }

            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_path) as writer:
                for metric_type, data in report_data.items():
                    if isinstance(data, dict) and not any(
                        key == "error" for key in data.keys()
                    ):
                        # Convert data to DataFrame for export
                        if metric_type == "quality_metrics" and "trends" in data:
                            # Handle nested trend data separately
                            trends_df = pd.DataFrame(data["trends"])
                            trends_df.to_excel(
                                writer, sheet_name=f"{metric_type}_trends"
                            )
                            # Remove trends from main metrics
                            data_copy = data.copy()
                            data_copy.pop("trends")
                            pd.DataFrame([data_copy]).to_excel(
                                writer, sheet_name=metric_type
                            )
                        else:
                            pd.DataFrame([data]).to_excel(
                                writer, sheet_name=metric_type
                            )

            self.logger.info(f"Comprehensive report generated at: {output_path}")

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            raise ProcessError(f"Report generation failed: {str(e)}")

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
        plt.figure(figsize=(12, 8))

        try:
            if metric_type == "production":
                self._visualize_production_trends()
            elif metric_type == "energy":
                self._visualize_energy_patterns()
            elif metric_type == "quality":
                self._visualize_quality_metrics()
            elif metric_type == "sustainability":
                self._visualize_sustainability_indicators()
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"Visualization saved to {save_path}")
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating visualization: {str(e)}")
            plt.close()
            raise ProcessError(f"Visualization generation failed: {str(e)}")

    def _visualize_production_trends(self):
        """Enhanced production efficiency and output trends visualization"""
        if self.production_data.empty:
            raise DataError("No production data available for visualization")

        # Get efficiency metrics from analyzer
        efficiency_metrics = self.efficiency_analyzer.analyze_batch_efficiency(
            self.production_data
        )

        plt.subplot(2, 2, 1)
        # Changed from output_quantity to output_amount
        daily_output = self.production_data.groupby(
            pd.Grouper(key="timestamp", freq="D")
        )["output_amount"].sum()
        daily_output.plot(style=".-", title="Daily Production Output")
        plt.ylabel("Output Amount")

        plt.subplot(2, 2, 2)
        sns.histplot(data=self.production_data, x="yield_rate", bins=20)
        plt.title("Yield Rate Distribution")
        plt.xlabel("Yield Rate (%)")

        plt.subplot(2, 2, 3)
        efficiency_trend = (
            self.production_data.set_index("timestamp")["yield_rate"]
            .rolling("7D")
            .mean()
        )
        efficiency_trend.plot(title="7-Day Rolling Average Efficiency")
        plt.ylabel("Efficiency (%)")

        plt.subplot(2, 2, 4)
        cycle_times = self.production_data.boxplot(
            column="cycle_time", by="production_line"
        )
        plt.title("Cycle Times by Production Line")
        plt.suptitle("")  # Remove automatic suptitle

    def _visualize_quality_metrics(self):
        """Enhanced quality control metrics visualization"""
        if self.quality_data.empty:
            raise DataError("No quality data available for visualization")

        # Get quality metrics from analyzer
        quality_metrics = self.quality_analyzer.analyze_defect_rates(self.quality_data)

        plt.subplot(2, 2, 1)
        sns.histplot(data=self.quality_data, x="efficiency", bins=20)
        plt.title("Cell Efficiency Distribution")
        plt.xlabel("Efficiency (%)")

        plt.subplot(2, 2, 2)
        daily_defects = self.quality_data.groupby(
            pd.Grouper(key="test_timestamp", freq="D")
        )["defect_rate"].mean()
        daily_defects.plot(style=".-", title="Daily Average Defect Rate")
        plt.ylabel("Defect Rate (%)")

        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.quality_data, y="thickness_uniformity")
        plt.title("Thickness Uniformity Distribution")

        plt.subplot(2, 2, 4)
        quality_trends = self.quality_analyzer.identify_quality_trends(
            self.quality_data
        )
        if quality_trends:
            pd.DataFrame(quality_trends)["efficiency_trend"].plot(
                title="Efficiency Trend Analysis"
            )

    def _visualize_sustainability_indicators(self):
        """Enhanced sustainability metrics visualization"""
        if self.material_flow.empty:
            raise DataError("No material flow data available for visualization")

        # Get sustainability metrics from analyzer
        material_metrics = self.sustainability_analyzer.analyze_material_efficiency(
            self.material_flow
        )

        plt.subplot(2, 2, 1)
        waste_by_type = self.material_flow.groupby("material_type")[
            "waste_generated"
        ].sum()
        waste_by_type.plot(kind="bar", title="Waste Generation by Material Type")
        plt.xticks(rotation=45)
        plt.ylabel("Waste Amount")

        plt.subplot(2, 2, 2)
        recycling_rates = self.material_flow.groupby("material_type").agg(
            {"recycled_amount": "sum", "waste_generated": "sum"}
        )
        recycling_rates["rate"] = (
            recycling_rates["recycled_amount"]
            / recycling_rates["waste_generated"]
            * 100
        )
        recycling_rates["rate"].plot(
            kind="bar", title="Recycling Rate by Material Type"
        )
        plt.ylabel("Recycling Rate (%)")
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 3)
        material_efficiency_trend = self.material_flow.groupby(
            pd.Grouper(key="timestamp", freq="W")
        ).apply(
            lambda x: (x["recycled_amount"].sum() / x["waste_generated"].sum() * 100)
        )
        material_efficiency_trend.plot(title="Weekly Material Efficiency Trend")
        plt.ylabel("Efficiency (%)")

        plt.subplot(2, 2, 4)
        if not self.energy_data.empty:
            energy_mix = self.energy_data.groupby("energy_source")[
                "energy_consumption"
            ].sum()
            energy_mix.plot(
                kind="pie", autopct="%1.1f%%", title="Energy Source Distribution"
            )

    def _visualize_energy_patterns(self):
        """Enhanced energy consumption pattern visualization"""
        if self.energy_data.empty:
            raise DataError("No energy data available for visualization")

        plt.subplot(2, 2, 1)
        energy_by_source = self.energy_data.groupby("energy_source")[
            "energy_consumption"
        ].sum()
        energy_by_source.plot(
            kind="pie", autopct="%1.1f%%", title="Energy Consumption by Source"
        )

        plt.subplot(2, 2, 2)
        hourly_consumption = self.energy_data.groupby(
            self.energy_data["timestamp"].dt.hour
        )["energy_consumption"].mean()
        hourly_consumption.plot(style=".-", title="Average Hourly Energy Consumption")
        plt.xlabel("Hour of Day")
        plt.ylabel("Energy Consumption")

        plt.subplot(2, 2, 3)
        daily_efficiency = self.energy_data.groupby(
            pd.Grouper(key="timestamp", freq="D")
        )["efficiency_rate"].mean()
        daily_efficiency.plot(title="Daily Energy Efficiency Rate")
        plt.ylabel("Efficiency Rate")

        plt.subplot(2, 2, 4)
        weekly_consumption = self.energy_data.groupby(
            pd.Grouper(key="timestamp", freq="W")
        )["energy_consumption"].sum()
        weekly_consumption.plot(kind="bar", title="Weekly Energy Consumption")
        plt.xticks(rotation=45)
        plt.ylabel("Energy Consumption")

    def _calculate_material_utilization(self) -> float:
        """
        Calculate material utilization rate using enhanced sustainability analyzer.
        Returns:
            float: Material utilization rate as a percentage.
        """
        if self.material_flow.empty:
            return 0.0

        # Use sustainability analyzer for consistent calculations
        material_metrics = self.sustainability_analyzer.analyze_material_efficiency(
            self.material_flow
        )
        return material_metrics.get("material_efficiency", 0.0)

    def _calculate_waste_metrics(self) -> Dict:
        """
        Calculate comprehensive waste-related metrics using sustainability analyzer.
        Returns:
            Dict: Contains total waste, recycling rate, and waste by material type.
        """
        if self.material_flow.empty:
            return {}

        try:
            material_metrics = self.sustainability_analyzer.analyze_material_efficiency(
                self.material_flow
            )

            waste_metrics = {
                "total_waste": self.material_flow["waste_generated"].sum(),
                "recycling_rate": material_metrics.get("recycling_rate", 0.0),
                "waste_by_type": self.material_flow.groupby("material_type")[
                    "waste_generated"
                ].sum(),
                "recovery_efficiency": material_metrics.get("material_efficiency", 0.0),
            }

            self.logger.info(f"Waste metrics calculated successfully: {waste_metrics}")
            return waste_metrics

        except Exception as e:
            self.logger.error(f"Error calculating waste metrics: {str(e)}")
            return {}

    def _calculate_energy_efficiency(self) -> float:
        """
        Calculate comprehensive energy efficiency score incorporating carbon footprint.
        Returns:
            float: Energy efficiency score considering both consumption and carbon impact.
        """
        if self.energy_data.empty:
            return 0.0

        try:
            # Calculate base energy efficiency
            base_efficiency = (
                self.energy_data["efficiency_rate"].mean() * 100
                if "efficiency_rate" in self.energy_data.columns
                else 0.0
            )

            # Calculate carbon footprint using sustainability analyzer
            carbon_footprint = self.sustainability_analyzer.calculate_carbon_footprint(
                self.energy_data
            )

            # Adjust efficiency score based on carbon footprint
            carbon_impact_factor = 1.0
            if carbon_footprint > 0:
                # Reduce efficiency score for higher carbon footprint
                carbon_impact_factor = max(
                    0.5, 1 - (carbon_footprint / 10000)
                )  # Example scaling

            adjusted_efficiency = base_efficiency * carbon_impact_factor

            self.logger.info(
                f"Energy efficiency calculated: {adjusted_efficiency:.2f}% "
                f"(Base: {base_efficiency:.2f}%, Carbon Impact: {carbon_impact_factor:.2f})"
            )

            return adjusted_efficiency

        except Exception as e:
            self.logger.error(f"Error calculating energy efficiency: {str(e)}")
            return 0.0

    def _estimate_carbon_footprint(self) -> float:
        """
        Estimate carbon footprint using sustainability analyzer.
        Returns:
            float: Estimated carbon footprint in kg CO2.
        """
        if self.energy_data.empty:
            return 0.0

        try:
            return self.sustainability_analyzer.calculate_carbon_footprint(
                self.energy_data
            )

        except Exception as e:
            self.logger.error(f"Error estimating carbon footprint: {str(e)}")
            return 0.0

    def generate_performance_report(self, save_path: str) -> None:
        """Generate visual performance report."""
        metrics = {
            "efficiency": self.calculate_efficiency(),
            "quality_score": self.calculate_quality_score(),
            "resource_efficiency": self.calculate_resource_efficiency(),
            "energy_efficiency": self.calculate_energy_efficiency(),
        }

        self.visualizer.create_kpi_dashboard(metrics, save_path)

    def export_analysis_report(self, output_path: str) -> None:
        """
        Generate a comprehensive analysis report and export to Excel.

        Args:
            output_path: Path where the Excel report should be saved
        """
        try:
            # Collect all metrics
            report_data = {
                "production_metrics": self.analyze_efficiency(),
                "quality_analysis": self.analyze_quality_metrics(),
                "sustainability_metrics": self.calculate_sustainability_metrics(),
            }

            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_path) as writer:
                for metric_type, data in report_data.items():
                    if isinstance(data, dict) and not any(
                        key == "error" for key in data.keys()
                    ):
                        pd.DataFrame([data]).to_excel(writer, sheet_name=metric_type)

            self.logger.info(f"Analysis report exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting analysis report: {str(e)}")
            raise ProcessError(f"Report export failed: {str(e)}")

    def calculate_efficiency(self) -> float:
        """Calculate overall manufacturing efficiency."""
        if self.production_data.empty:
            return 0.0
        return (
            self.production_data["output_amount"] / self.production_data["input_amount"]
        ).mean() * 100

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score."""
        if self.quality_data.empty:
            return 0.0
        return 100 - self.quality_data["defect_rate"].mean()

    def calculate_resource_efficiency(self) -> float:
        """Calculate resource utilization efficiency."""
        return self._calculate_material_utilization()

    def calculate_energy_efficiency(self) -> float:
        """Calculate energy efficiency score."""
        return self._calculate_energy_efficiency()


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
