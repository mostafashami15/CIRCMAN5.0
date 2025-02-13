"""Core manufacturing analysis module."""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Union
from pathlib import Path
from datetime import datetime

from circman5.config.project_paths import project_paths
from ..utils.logging_config import setup_logger
from ..utils.errors import ValidationError, ProcessError, DataError, ResourceError
from ..ai.optimization_prediction import ManufacturingOptimizer
from ..ai.optimization_types import MetricsDict, PredictionDict
from ..analysis.efficiency import EfficiencyAnalyzer
from ..analysis.quality import QualityAnalyzer
from ..analysis.sustainability import SustainabilityAnalyzer
from ..analysis.lca.core import LCAAnalyzer, LifeCycleImpact
from ..analysis.lca.impact_factors import (
    MATERIAL_IMPACT_FACTORS,
    ENERGY_IMPACT_FACTORS,
    RECYCLING_BENEFIT_FACTORS,
    PROCESS_IMPACT_FACTORS,
)
from ..visualization.manufacturing_visualizer import ManufacturingVisualizer
from ..visualization.lca_visualizer import LCAVisualizer
from ..monitoring import ManufacturingMonitor  # Add this line


class SoliTekManufacturingAnalysis:
    """
    A comprehensive framework for analyzing SoliTek's PV manufacturing data.
    This class implements the methodology outlined in our case study framework,
    focusing on production efficiency, sustainability metrics, and AI-driven optimization.
    """

    def __init__(self):
        # Use project paths for logging
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

        # LCA-specific initializations
        self.lca_analyzer = LCAAnalyzer()
        self.lca_data = {
            "material_flow": pd.DataFrame(),
            "energy_consumption": pd.DataFrame(),
            "process_data": pd.DataFrame(),
        }
        self.energy_sources = ["grid", "solar", "wind"]

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
            "timestamp": "datetime64[ns]",
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

        # Modify the output amount validation to allow a small margin of error
        # This accounts for potential floating-point imprecision or minor manufacturing variations
        excessive_output = data[data["output_amount"] > data["input_amount"] * 1.1]
        if not excessive_output.empty:
            raise ValidationError(
                "Output amount cannot significantly exceed input amount",
                invalid_data={
                    "field": "output_amount",
                    "problematic_rows": excessive_output.index.tolist(),
                },
            )

        return True

    def load_production_data(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Load and validate production data from CSV files.

        Args:
            file_path: Path to the CSV file. If None, uses test data path.
        """
        try:
            if file_path is None:
                # Explicitly convert to string
                file_path = str(
                    Path(project_paths.get_path("SYNTHETIC_DATA"))
                    / "test_production_data.csv"
                )
            else:
                # Ensure file_path is a string
                file_path = str(file_path)

            self.logger.info(f"Loading production data from {file_path}")

            if not Path(file_path).exists():
                raise DataError(f"Production data file not found: {file_path}")

            data = pd.read_csv(file_path)
            if data.empty:
                raise DataError("Production data file is empty")

            if self.validate_production_data(data):
                self.production_data = data
                self.logger.info(f"Successfully loaded {len(data)} records")

        except Exception as e:
            self.logger.error(f"Error loading production data: {str(e)}")
            raise

    def analyze_efficiency(self) -> Dict:
        """
        Analyze manufacturing efficiency metrics.
        Returns:
            Dict containing efficiency metrics
        """
        if self.production_data.empty:
            self.logger.warning("No production data available for efficiency analysis")
            return {"error": "No production data available"}

        try:
            # Calculate base efficiency metrics using the efficiency analyzer
            base_metrics = self.efficiency_analyzer.analyze_batch_efficiency(
                self.production_data
            )

            # Calculate additional metrics
            efficiency_metrics = {
                "yield_rate": base_metrics.get("yield_rate", 0.0),
                "cycle_time_efficiency": base_metrics.get("cycle_time_efficiency", 0.0),
                "energy_efficiency": base_metrics.get("energy_efficiency", 0.0),
                "output_amount": self.production_data["output_amount"].mean(),
                "input_amount": self.production_data["input_amount"].mean(),
            }

            self.logger.info(f"Efficiency analysis completed: {efficiency_metrics}")
            return efficiency_metrics

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

    def load_lca_data(
        self,
        material_data_path: Optional[str] = None,
        energy_data_path: Optional[str] = None,
        process_data_path: Optional[str] = None,
    ) -> None:
        """
        Load LCA-specific data from CSV files or use synthetic data for testing.

        Args:
            material_data_path: Path to material flow data CSV
            energy_data_path: Path to energy consumption data CSV
            process_data_path: Path to process data CSV
        """
        try:
            if material_data_path:
                self.lca_data["material_flow"] = pd.read_csv(material_data_path)
            if energy_data_path:
                self.lca_data["energy_consumption"] = pd.read_csv(energy_data_path)
            if process_data_path:
                self.lca_data["process_data"] = pd.read_csv(process_data_path)

            self.logger.info("Successfully loaded LCA data")

        except Exception as e:
            self.logger.error(f"Error loading LCA data: {str(e)}")
            raise

    def perform_lifecycle_assessment(
        self, batch_id: Optional[str] = None
    ) -> LifeCycleImpact:
        """
        Perform comprehensive lifecycle assessment for specified batch or overall production.

        Args:
            batch_id: Optional batch identifier for specific analysis

        Returns:
            LifeCycleImpact object containing impact assessments for all lifecycle phases
        """
        try:
            # Filter data for specific batch if provided
            material_data = self._filter_batch_data(
                self.lca_data["material_flow"], batch_id
            )
            energy_data = self._filter_batch_data(
                self.lca_data["energy_consumption"], batch_id
            )
            process_data = self._filter_batch_data(
                self.lca_data["process_data"], batch_id
            )

            # Calculate material inputs
            material_inputs = self._aggregate_material_inputs(material_data)

            # Calculate energy consumption
            total_energy = energy_data["energy_consumption"].sum()

            # Perform full LCA calculation
            impact = self.lca_analyzer.perform_full_lca(
                material_inputs=material_inputs,
                energy_consumption=total_energy,
                lifetime_years=25.0,  # Standard PV panel lifetime
                annual_energy_generation=self._calculate_energy_generation(
                    material_inputs
                ),
                grid_carbon_intensity=0.5,  # Can be adjusted based on location
                recycling_rates=self._calculate_recycling_rates(material_data),
                transport_distance=100.0,  # Average transport distance in km
            )

            self.logger.info(f"Completed lifecycle assessment for batch {batch_id}")
            return impact

        except Exception as e:
            self.logger.error(f"Error in lifecycle assessment: {str(e)}")
            raise

    def generate_lca_report(
        self, output_path: str, batch_id: Optional[str] = None
    ) -> None:
        """
        Generate comprehensive LCA report including all impact categories.

        Args:
            output_path: Path where the report should be saved
            batch_id: Optional batch identifier for specific analysis
        """
        try:
            # Perform lifecycle assessment
            impact = self.perform_lifecycle_assessment(batch_id)

            # Create detailed report
            report_data = {
                "Manufacturing Impact": {
                    "Total Impact (kg CO2-eq)": impact.manufacturing_impact,
                    "Material Production": self._calculate_material_impacts(),
                    "Energy Use": self._calculate_energy_impacts(),
                    "Process Impacts": self._calculate_process_impacts(),
                },
                "Use Phase": {
                    "Total Impact (kg CO2-eq)": impact.use_phase_impact,
                    "Energy Generation Benefit": self._calculate_generation_benefit(),
                    "Maintenance Impact": self._calculate_maintenance_impact(),
                },
                "End of Life": {
                    "Total Impact (kg CO2-eq)": impact.end_of_life_impact,
                    "Recycling Benefits": self._calculate_recycling_benefits(),
                    "Disposal Impact": self._calculate_disposal_impact(),
                    "Transport Impact": self._calculate_transport_impact(),
                },
            }

            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_path) as writer:
                for category, data in report_data.items():
                    pd.DataFrame([data]).to_excel(writer, sheet_name=category)

            self.logger.info(f"LCA report generated successfully at {output_path}")

        except Exception as e:
            self.logger.error(f"Error generating LCA report: {str(e)}")
            raise

    def _filter_batch_data(
        self, data: pd.DataFrame, batch_id: Optional[str]
    ) -> pd.DataFrame:
        """Filter data for specific batch if batch_id is provided."""
        if batch_id and not data.empty and "batch_id" in data.columns:
            return data[data["batch_id"] == batch_id]
        return data

    def _aggregate_material_inputs(
        self, material_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Aggregate material quantities by type."""
        if material_data.empty:
            return {}
        return material_data.groupby("material_type")["quantity_used"].sum().to_dict()

    def _calculate_recycling_rates(
        self, material_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate recycling rates from historical data.

        Args:
            material_data: DataFrame containing waste and recycling information
                Must include 'material_type', 'waste_generated', and 'recycled_amount' columns

        Returns:
            Dictionary mapping material types to their recycling rates (0.0 to 1.0)
        """
        if material_data.empty:
            return {}

        recycling_rates = {}

        try:
            # First, create a copy to avoid modifying the original data
            data_copy = material_data.copy()

            # Convert columns to numeric values, replacing errors with NaN
            data_copy["waste_generated"] = data_copy["waste_generated"].astype(float)
            data_copy["recycled_amount"] = data_copy["recycled_amount"].astype(float)

            # Group by material type and calculate sums
            material_totals = (
                data_copy.groupby("material_type")
                .agg({"waste_generated": "sum", "recycled_amount": "sum"})
                .astype(float)
            )  # Ensure totals are float type

            # Calculate recycling rate for each material
            for material in material_totals.index:
                # Get values as native Python floats
                waste = float(material_totals.at[material, "waste_generated"])
                recycled = float(material_totals.at[material, "recycled_amount"])

                # Calculate rate with safe division
                if waste > 0:
                    rate = recycled / waste
                    # Ensure rate is between 0 and 1
                    rate = max(0.0, min(1.0, rate))
                else:
                    rate = 0.0

                recycling_rates[material] = rate

            self.logger.info(
                f"Calculated recycling rates for {len(recycling_rates)} materials"
            )
            return recycling_rates

        except Exception as e:
            self.logger.error(f"Error calculating recycling rates: {str(e)}")
            return {}

    def _calculate_energy_generation(self, material_inputs: Dict[str, float]) -> float:
        """
        Calculate expected annual energy generation based on panel specifications.

        Args:
            material_inputs: Dictionary of material quantities in kg

        Returns:
            float: Annual energy generation in kWh
        """
        try:
            # Convert the glass weight to a float and calculate area
            glass_weight = float(material_inputs.get("solar_glass", 0))
            total_panel_area = glass_weight / 10.0  # Approximate area from glass weight

            average_efficiency = 0.20  # 20% efficiency
            solar_irradiance = 1000.0  # kWh/m²/year (typical value)

            return total_panel_area * average_efficiency * solar_irradiance
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Error calculating energy generation: {str(e)}")
            return 0.0

    def _calculate_material_impacts(self) -> float:
        """Calculate detailed material production impacts."""
        if self.lca_data["material_flow"].empty:
            return 0.0

        total_impact = 0.0
        try:
            material_quantities = self._aggregate_material_inputs(
                self.lca_data["material_flow"]
            )
            for material, quantity in material_quantities.items():
                quantity = float(quantity)  # Ensure numeric type
                impact_factor = float(MATERIAL_IMPACT_FACTORS.get(material, 0.0))
                if quantity > 0:  # Now comparing numeric values
                    total_impact += quantity * impact_factor
            return total_impact
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Error calculating material impacts: {str(e)}")
            return 0.0

    def _calculate_energy_impacts(self) -> float:
        """
        Calculate environmental impacts from energy consumption.
        Returns the total impact in kg CO2-eq.
        """
        if self.lca_data["energy_consumption"].empty:
            return 0.0

        total_impact = 0.0
        energy_data = self.lca_data["energy_consumption"]

        # Calculate impact for each energy source
        for source in self.energy_sources:
            # Sum up consumption for this energy source
            source_consumption = energy_data[energy_data["energy_source"] == source][
                "energy_consumption"
            ].sum()

            # Get impact factor for this energy source
            impact_factor = ENERGY_IMPACT_FACTORS.get(source, 0.0)

            # Calculate impact and add to total
            total_impact += source_consumption * impact_factor

        return total_impact

    def _calculate_process_impacts(self) -> float:
        """
        Calculate environmental impacts from manufacturing processes.
        Each process step has its own impact factor based on industry standards.
        Returns the total impact in kg CO2-eq.
        """
        if self.lca_data["process_data"].empty:
            return 0.0

        total_impact = 0.0
        process_data = self.lca_data["process_data"]

        # Calculate impact for each distinct manufacturing process step
        for process_step in process_data["process_step"].unique():
            # Calculate total time spent on this process step
            step_time = process_data[process_data["process_step"] == process_step][
                "process_time"
            ].sum()

            # Get the environmental impact factor for this process
            impact_factor = PROCESS_IMPACT_FACTORS.get(process_step, 0.0)

            # Calculate impact of this process step and add to total
            total_impact += step_time * impact_factor

        return total_impact

    def _calculate_generation_benefit(self) -> float:
        """
        Calculate environmental benefits from clean energy generation over the panel lifetime.
        This represents the emissions avoided by generating solar power instead of using grid electricity.
        Returns the total benefit in kg CO2-eq (negative value represents environmental benefit).
        """
        if self.material_flow.empty:
            return 0.0

        # Calculate total panel area from material inputs
        material_inputs = self._aggregate_material_inputs(self.material_flow)
        # Convert glass weight to approximate panel area (10kg glass per m²)
        total_panel_area = material_inputs.get("solar_glass", 0) / 10.0  # m²

        # Calculate annual energy generation
        average_efficiency = 0.20  # 20% efficiency
        solar_irradiance = 1000.0  # kWh/m²/year (typical value)
        annual_generation = total_panel_area * average_efficiency * solar_irradiance

        # Calculate emissions avoided over 25-year lifetime
        grid_carbon_intensity = 0.5  # kg CO2/kWh (can be adjusted by region)
        lifetime_benefit = annual_generation * 25 * grid_carbon_intensity

        # Return negative value since this is an environmental benefit
        return -lifetime_benefit

    def _calculate_maintenance_impact(self) -> float:
        """
        Calculate environmental impacts from maintenance activities over panel lifetime.
        Includes regular cleaning, inspections, and minor repairs.
        Returns the total impact in kg CO2-eq.
        """
        if self.material_flow.empty:
            return 0.0

        # Calculate panel area for maintenance calculations
        material_inputs = self._aggregate_material_inputs(self.material_flow)
        total_panel_area = material_inputs.get("solar_glass", 0) / 10.0  # m²

        # Define maintenance impact factors
        cleaning_impact = 0.1  # kg CO2-eq/m²/year for cleaning
        inspection_impact = 0.05  # kg CO2-eq/m²/year for inspections

        # Calculate annual maintenance impact
        annual_impact = total_panel_area * (cleaning_impact + inspection_impact)

        # Calculate lifetime impact (25-year standard lifetime)
        lifetime_impact = annual_impact * 25

        return lifetime_impact

    def _calculate_recycling_benefits(self) -> float:
        """
        Calculate environmental benefits from material recycling.
        Each material has a specific benefit factor representing emissions avoided through recycling.
        Returns the total benefit in kg CO2-eq (negative value represents benefit).
        """
        if self.lca_data["material_flow"].empty:
            return 0.0

        total_benefit = 0.0
        material_data = self.lca_data["material_flow"]

        # Calculate benefits for each material type
        for material in material_data["material_type"].unique():
            # Get total recycled amount for this material
            recycled_amount = material_data[material_data["material_type"] == material][
                "recycled_amount"
            ].sum()

            # Get recycling benefit factor for this material
            benefit_factor = RECYCLING_BENEFIT_FACTORS.get(material, 0.0)

            # Calculate and add recycling benefit
            total_benefit += recycled_amount * benefit_factor

        return total_benefit  # Negative value represents environmental benefit

    def _calculate_disposal_impact(self) -> float:
        """
        Calculate environmental impacts from waste disposal.
        Considers the impact of landfilling non-recycled materials.
        Returns the total impact in kg CO2-eq.
        """
        if self.lca_data["material_flow"].empty:
            return 0.0

        total_impact = 0.0
        material_data = self.lca_data["material_flow"]

        # Calculate disposal impact for each material type
        for material in material_data["material_type"].unique():
            # Get total waste and recycled amounts
            material_subset = material_data[material_data["material_type"] == material]
            waste_generated = material_subset["waste_generated"].sum()
            recycled_amount = material_subset["recycled_amount"].sum()

            # Calculate amount going to disposal
            disposed_amount = waste_generated - recycled_amount

            # Apply disposal impact factor (can vary by material type)
            disposal_impact_factor = 0.1  # kg CO2-eq per kg disposed
            total_impact += disposed_amount * disposal_impact_factor

        return total_impact

    def _calculate_transport_impact(self) -> float:
        """
        Calculate environmental impacts from transportation.
        Considers the impact of transporting materials and finished products.
        Returns the total impact in kg CO2-eq.
        """
        if self.lca_data["material_flow"].empty:
            return 0.0

        # Calculate total mass being transported
        total_mass = self.lca_data["material_flow"]["quantity_used"].sum()

        # Define transport parameters
        transport_distance = 100.0  # Average transport distance in km
        transport_emission_factor = 0.062  # kg CO2-eq per tonne-km

        # Convert mass to tonnes and calculate impact
        impact = (total_mass / 1000) * transport_distance * transport_emission_factor

        return impact

    def _visualize_production_trends(self):
        """Enhanced production efficiency and output trends visualization"""
        if self.production_data.empty:
            raise DataError("No production data available for visualization")

        plt.subplot(2, 2, 1)
        daily_output = self.production_data.groupby(
            pd.Grouper(key="timestamp", freq="D")
        )["output_amount"].sum()
        daily_output.plot(style=".-", title="Daily Production Output")
        plt.ylabel("Output Amount")

        plt.subplot(2, 2, 2)
        # Updated seaborn plot without categorical warning
        sns.histplot(
            data=self.production_data,
            x="yield_rate",
            bins=20,
            stat="count",
            common_norm=False,
        )
        plt.title("Yield Rate Distribution")
        plt.xlabel("Yield Rate (%)")

        plt.subplot(2, 2, 3)
        efficiency_trend = (
            self.production_data.set_index("timestamp")["yield_rate"]
            .rolling("7D", min_periods=1)
            .mean()
        )
        efficiency_trend.plot(title="7-Day Rolling Average Efficiency")
        plt.ylabel("Efficiency (%)")

        plt.subplot(2, 2, 4)
        # Updated boxplot without deprecation warning
        sns.boxplot(
            data=self.production_data, y="cycle_time", x="production_line", orient="v"
        )
        plt.title("Cycle Times by Production Line")

    def _visualize_quality_metrics(self):
        """Enhanced quality control metrics visualization"""
        if self.quality_data.empty:
            raise DataError("No quality data available for visualization")

        # Get quality metrics from analyzer
        quality_metrics = self.quality_analyzer.analyze_defect_rates(self.quality_data)

        plt.subplot(2, 2, 1)
        sns.histplot(data=self.quality_data, x="efficiency", bins=20, stat="count")
        plt.title("Cell Efficiency Distribution")
        plt.xlabel("Efficiency (%)")

        plt.subplot(2, 2, 2)
        daily_defects = self.quality_data.groupby(
            pd.Grouper(key="timestamp", freq="D"), observed=True
        )["defect_rate"].mean()
        daily_defects.plot(style=".-", title="Daily Average Defect Rate")
        plt.ylabel("Defect Rate (%)")

        plt.subplot(2, 2, 3)
        sns.boxplot(
            data=self.quality_data, y="thickness_uniformity", orientation="vertical"
        )
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

        material_metrics = self.sustainability_analyzer.analyze_material_efficiency(
            self.material_flow
        )

        plt.subplot(2, 2, 1)
        waste_by_type = self.material_flow.groupby("material_type", observed=True)[
            "waste_generated"
        ].sum()
        waste_by_type.plot(kind="bar", title="Waste Generation by Material Type")
        plt.xticks(rotation=45)
        plt.ylabel("Waste Amount")

        plt.subplot(2, 2, 2)
        # Updated groupby operation to avoid warning
        recycling_rates = self.material_flow.groupby(
            "material_type", observed=True
        ).agg({"recycled_amount": "sum", "waste_generated": "sum"})
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
        # Updated groupby and apply operation
        material_efficiency_trend = (
            self.material_flow.groupby(
                pd.Grouper(key="timestamp", freq="W"), observed=True
            )
            .agg({"recycled_amount": "sum", "waste_generated": "sum"})
            .assign(
                efficiency=lambda x: (x["recycled_amount"] / x["waste_generated"] * 100)
            )["efficiency"]
        )
        material_efficiency_trend.plot(title="Weekly Material Efficiency Trend")
        plt.ylabel("Efficiency (%)")

        plt.subplot(2, 2, 4)
        if not self.energy_data.empty:
            energy_mix = self.energy_data.groupby("energy_source", observed=True)[
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

    def export_analysis_report(self, output_path: Optional[str] = None) -> None:
        """
        Generate a comprehensive analysis report and export to Excel.

        Args:
            output_path: Optional path where the Excel report should be saved.
                        If None, uses default project path.
        """
        try:
            if output_path is None:
                run_dir = project_paths.get_run_directory()
                output_path = str(run_dir / "reports" / "analysis_report.xlsx")

            # Collect all metrics
            report_data = {
                "production_metrics": self.analyze_efficiency(),
                "quality_analysis": self.analyze_quality_metrics(),
                "sustainability_metrics": self.calculate_sustainability_metrics(),
            }

            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_path) as writer:
                has_data = False
                for metric_type, data in report_data.items():
                    if isinstance(data, dict) and not any(
                        key == "error" for key in data.keys()
                    ):
                        pd.DataFrame([data]).to_excel(writer, sheet_name=metric_type)
                        has_data = True

                if not has_data:
                    pd.DataFrame(["No data available"]).to_excel(
                        writer, sheet_name="Empty_Report"
                    )

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
