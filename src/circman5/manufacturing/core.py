# src/circman5/manufacturing/core.py

from typing import Dict, Optional, List, Union, Mapping
from pathlib import Path
import pandas as pd

from .optimization.types import PredictionDict
from .optimization.types import MetricsDict
from ..utils.logging_config import setup_logger
from ..utils.errors import DataError, ProcessError
from .optimization import ManufacturingModel, ProcessOptimizer
from .analyzers import EfficiencyAnalyzer, QualityAnalyzer, SustainabilityAnalyzer
from .data_loader import ManufacturingDataLoader
from ..monitoring import ManufacturingMonitor
from .reporting.visualizations import ManufacturingVisualizer
from .reporting.reports import ReportGenerator
from .lifecycle.lca_analyzer import LCAAnalyzer, LifeCycleImpact
from .lifecycle.visualizer import LCAVisualizer
from .lifecycle.impact_factors import (
    MATERIAL_IMPACT_FACTORS,
    ENERGY_IMPACT_FACTORS,
    RECYCLING_BENEFIT_FACTORS,
)


class SoliTekManufacturingAnalysis:
    """
    Core class for SoliTek manufacturing analysis, integrating all analysis modules.
    """

    def __init__(self):
        self.logger = setup_logger("solitek_manufacturing")

        # Initialize data loader
        self.data_loader = ManufacturingDataLoader()

        # Initialize analyzers
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
        self.sustainability_analyzer = SustainabilityAnalyzer()
        self.lca_analyzer = LCAAnalyzer()

        # Initialize visualizers
        self.visualizer = ManufacturingVisualizer()
        self.lca_visualizer = LCAVisualizer()

        # Initialize report generator
        self.report_generator = ReportGenerator()

        # Initialize optimizer and monitor
        self.model = ManufacturingModel()
        self.optimizer = ProcessOptimizer(self.model)
        self.is_optimizer_trained = False
        self.monitor = ManufacturingMonitor()

        # Initialize data structures
        self.production_data = pd.DataFrame()
        self.energy_data = pd.DataFrame()
        self.quality_data = pd.DataFrame()
        self.material_flow = pd.DataFrame()
        self.lca_data = {
            "material_flow": pd.DataFrame(),
            "energy_consumption": pd.DataFrame(),
            "process_data": pd.DataFrame(),
        }

    def load_data(
        self,
        production_path: Optional[str] = None,
        quality_path: Optional[str] = None,
        material_path: Optional[str] = None,
        energy_path: Optional[str] = None,
        process_path: Optional[str] = None,
    ) -> None:
        """
        Load all required data files for analysis.

        Args:
            production_path: Path to production data file
            quality_path: Path to quality data file
            material_path: Path to material flow data file
            energy_path: Path to energy consumption data file
            process_path: Path to process data file
        """
        try:
            if production_path:
                self.production_data = self.data_loader.load_production_data(
                    production_path
                )

            if quality_path:
                self.quality_data = self.data_loader.load_quality_data(quality_path)

            if material_path:
                self.material_flow = self.data_loader.load_material_data(material_path)
                self.lca_data["material_flow"] = self.material_flow

            if energy_path:
                self.energy_data = self.data_loader.load_energy_data(energy_path)
                self.lca_data["energy_consumption"] = self.energy_data

            if process_path:
                self.lca_data["process_data"] = self.data_loader.load_process_data(
                    process_path
                )

            self.logger.info("Successfully loaded all data files")

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise DataError(f"Data loading failed: {str(e)}")

    def perform_lifecycle_assessment(
        self, batch_id: Optional[str] = None, output_dir: Optional[Path] = None
    ) -> LifeCycleImpact:
        """
        Perform comprehensive lifecycle assessment and visualization.

        Args:
            batch_id: Optional batch identifier for specific analysis

        Returns:
            LifeCycleImpact: Object containing impact assessments
        """
        try:
            # Validate input data
            if self.lca_data["material_flow"].empty:
                raise ValueError("No material flow data available")
            if self.lca_data["energy_consumption"].empty:
                raise ValueError("No energy consumption data available")

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

            # Calculate material inputs and energy consumption
            material_inputs = self.lca_analyzer._aggregate_material_inputs(
                material_data
            )
            total_energy = energy_data["energy_consumption"].sum()

            # Calculate recycling rates
            recycling_rates = self.lca_analyzer.calculate_recycling_rates(material_data)

            # Calculate annual energy generation
            annual_energy_generation = self.lca_analyzer.calculate_energy_generation(
                material_inputs
            )

            # Perform full LCA calculation
            impact = self.lca_analyzer.perform_full_lca(
                material_inputs=material_inputs,
                energy_consumption=total_energy,
                lifetime_years=25.0,  # Standard PV panel lifetime
                annual_energy_generation=annual_energy_generation,
                grid_carbon_intensity=0.5,  # Can be adjusted based on location
                recycling_rates=recycling_rates,
                transport_distance=100.0,  # Average transport distance in km
            )

            # Generate visualizations
            self._generate_lca_visualizations(
                impact, material_data, energy_data, batch_id, output_dir=output_dir
            )

            # Save LCA results
            self.lca_analyzer.save_results(impact, batch_id)

            self.logger.info(f"Completed lifecycle assessment for batch {batch_id}")
            return impact

        except Exception as e:
            self.logger.error(f"Error in lifecycle assessment: {str(e)}")
            raise ProcessError(f"Lifecycle assessment failed: {str(e)}")

    def analyze_manufacturing_performance(self) -> Dict[str, Dict[str, float]]:
        """Perform comprehensive manufacturing performance analysis.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing all performance metrics
        """
        try:
            # Calculate efficiency metrics and ensure float values
            efficiency_metrics = self.efficiency_analyzer.analyze_batch_efficiency(
                self.production_data
            )
            efficiency_metrics = {k: float(v) for k, v in efficiency_metrics.items()}

            # Calculate quality metrics and ensure float values
            quality_metrics = self.quality_analyzer.analyze_defect_rates(
                self.quality_data
            )
            quality_metrics = {k: float(v) for k, v in quality_metrics.items()}

            # Get sustainability metrics (already flattened and converted to float)
            sustainability_metrics = self.calculate_sustainability_metrics()

            # Generate visualizations
            self._generate_performance_visualizations()

            # Return structured metrics dictionary
            performance_metrics: Dict[str, Dict[str, float]] = {
                "efficiency": efficiency_metrics,
                "quality": quality_metrics,
                "sustainability": sustainability_metrics,
            }

            return performance_metrics

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            raise ProcessError(f"Performance analysis failed: {str(e)}")

    def generate_reports(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Generate comprehensive reports including all analyses.

        Args:
            output_dir: Optional directory for report output (str or Path)
        """
        try:
            # Convert str to Path if needed; use current directory as default if None
            output_path = Path(output_dir) if output_dir else Path.cwd()

            # Collect all metrics
            performance_metrics = self.analyze_manufacturing_performance()

            # Generate reports using report generator
            self.report_generator.generate_comprehensive_report(
                performance_metrics, output_dir=output_path
            )

            # Rename the generated report file to "analysis_report.xlsx"
            generated_file = output_path / "comprehensive_analysis.xlsx"
            expected_file = output_path / "analysis_report.xlsx"
            if generated_file.exists():
                generated_file.rename(expected_file)

            self.logger.info("Successfully generated all reports")

        except Exception as e:
            self.logger.error(f"Error generating reports: {str(e)}")
            raise ProcessError(f"Report generation failed: {str(e)}")

    def _generate_lca_visualizations(
        self,
        impact: LifeCycleImpact,
        material_data: pd.DataFrame,
        energy_data: pd.DataFrame,
        batch_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Generate all LCA-related visualizations."""
        try:
            # Use provided output directory or fall back to default
            viz_dir = Path(output_dir) if output_dir else self.lca_visualizer.viz_dir
            viz_dir.mkdir(parents=True, exist_ok=True)

            # Create visualization filename with batch_id if provided
            filename_suffix = f"_{batch_id}" if batch_id else ""

            # Generate lifecycle comparison visualization
            self.lca_visualizer.plot_lifecycle_comparison(
                impact.manufacturing_impact,
                impact.use_phase_impact,
                impact.end_of_life_impact,
                save_path=str(viz_dir / f"lifecycle_comparison{filename_suffix}.png"),
            )

            # Generate material flow visualization
            self.lca_visualizer.plot_material_flow(
                material_data,
                save_path=str(viz_dir / f"material_flow{filename_suffix}.png"),
            )

            # Generate energy consumption visualization
            self.lca_visualizer.plot_energy_consumption_trends(
                energy_data,
                save_path=str(viz_dir / f"energy_trends{filename_suffix}.png"),
            )

            self.logger.info(f"Generated visualizations in {viz_dir}")

        except Exception as e:
            self.logger.error(f"Error generating LCA visualizations: {str(e)}")
            raise ProcessError(f"Visualization generation failed: {str(e)}")

    def _generate_performance_visualizations(self) -> None:
        """Generate all performance-related visualizations."""
        try:
            # Production trends
            self.visualizer.visualize_production_trends(
                self.production_data, save_path="production_trends.png"
            )

            # Quality metrics
            self.visualizer.visualize_quality_metrics(
                self.quality_data,
                analyzer=self.quality_analyzer,
                save_path="quality_metrics.png",
            )

            # Sustainability indicators
            self.visualizer.visualize_sustainability_indicators(
                self.material_flow,
                self.energy_data,
                save_path="sustainability_indicators.png",
            )

            # Energy patterns
            self.visualizer.visualize_energy_patterns(
                self.energy_data, save_path="energy_patterns.png"
            )

        except Exception as e:
            self.logger.error(f"Error generating performance visualizations: {str(e)}")
            raise ProcessError(f"Visualization generation failed: {str(e)}")

    def _filter_batch_data(
        self, data: pd.DataFrame, batch_id: Optional[str]
    ) -> pd.DataFrame:
        """
        Filter data for specific batch if batch_id is provided.

        Args:
            data: DataFrame to filter
            batch_id: Batch identifier

        Returns:
            Filtered DataFrame
        """
        if batch_id and not data.empty and "batch_id" in data.columns:
            return data[data["batch_id"] == batch_id]
        return data

    def train_optimization_model(self) -> MetricsDict:
        """Train the AI optimization model with current manufacturing data."""
        if self.production_data.empty or self.quality_data.empty:
            raise ValueError("No data available for training optimization model")

        try:
            metrics = self.model.train_optimization_model(
                self.production_data, self.quality_data
            )
            self.is_optimizer_trained = True
            self.logger.info(f"AI optimization model trained. Metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error training optimization model: {str(e)}")
            raise

    def optimize_process_parameters(
        self,
        current_params: Dict[str, float],
        constraints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Optimize manufacturing process parameters."""
        try:
            # Make sure model is trained
            if not self.is_optimizer_trained:
                metrics = self.train_optimization_model()
                if not metrics or metrics.get("r2", 0) <= 0:
                    raise ValueError("Model training failed or produced poor results")

            optimized_params = self.optimizer.optimize_process_parameters(
                current_params, constraints  # type: ignore
            )
            self.logger.info(
                f"Optimized parameters generated:\n"
                + "\n".join(f"{k}: {v:.2f}" for k, v in optimized_params.items())
            )
            return optimized_params
        except Exception as e:
            self.logger.error(f"Error optimizing process parameters: {str(e)}")
            raise

    def predict_batch_outcomes(
        self, process_params: Dict[str, float]
    ) -> PredictionDict:
        """Predict manufacturing outcomes for given parameters."""
        try:
            predictions = self.model.predict_batch_outcomes(process_params)
            self.logger.info(
                f"Manufacturing predictions:\n"
                + f"Predicted Output: {predictions['predicted_output']:.2f}\n"
                + f"Predicted Quality: {predictions['predicted_quality']:.2f}"
            )
            return predictions
        except Exception as e:
            self.logger.error(f"Error predicting batch outcomes: {str(e)}")
            raise

    def analyze_optimization_potential(self) -> Dict[str, float]:
        """Analyze potential optimizations based on historical data."""
        try:
            improvements = self.optimizer.analyze_optimization_potential(
                self.production_data, self.quality_data
            )
            self.logger.info(
                "Optimization potential analysis:\n"
                + "\n".join(f"{k}: {v:.1f}%" for k, v in improvements.items())
            )
            return improvements
        except Exception as e:
            self.logger.error(f"Error analyzing optimization potential: {str(e)}")
            raise

    def analyze_quality_metrics(self) -> Dict[str, float]:
        """Analyze quality control metrics."""
        try:
            if self.quality_data.empty:
                raise DataError("No quality data available for analysis")

            # Get base quality metrics
            metrics = self.quality_analyzer.analyze_defect_rates(self.quality_data)

            # Add quality score
            quality_score = self.quality_analyzer.calculate_quality_score(
                self.quality_data
            )
            metrics["overall_quality_score"] = quality_score

            return metrics
        except Exception as e:
            self.logger.error(f"Error analyzing quality metrics: {str(e)}")
            raise ProcessError(f"Quality analysis failed: {str(e)}")

    def analyze_efficiency(self) -> Dict[str, float]:
        """Analyze manufacturing efficiency metrics."""
        try:
            if self.production_data.empty:
                raise DataError("No production data available for analysis")

            return self.efficiency_analyzer.analyze_batch_efficiency(
                self.production_data
            )
        except Exception as e:
            self.logger.error(f"Error analyzing efficiency: {str(e)}")
            raise ProcessError(f"Efficiency analysis failed: {str(e)}")

    def calculate_sustainability_metrics(self) -> Dict[str, float]:
        """Calculate sustainability metrics.

        Returns:
            Dict[str, float]: Flattened dictionary of sustainability metrics
        """
        try:
            # Get nested metrics from sustainability analyzer
            nested_metrics = (
                self.sustainability_analyzer.calculate_sustainability_metrics(
                    self.energy_data, self.material_flow
                )
            )

            # Flatten the nested dictionary into a single level
            flattened_metrics: Dict[str, float] = {}
            for category, metrics in nested_metrics.items():
                for metric, value in metrics.items():
                    # Ensure all values are float
                    try:
                        float_value = float(value)
                    except (TypeError, ValueError) as e:
                        self.logger.warning(
                            f"Could not convert {metric} value to float: {str(e)}"
                        )
                        float_value = 0.0

                    flattened_metrics[f"{category}_{metric}"] = float_value

            return flattened_metrics

        except Exception as e:
            self.logger.error(f"Error calculating sustainability metrics: {str(e)}")
            raise ProcessError(f"Sustainability calculation failed: {str(e)}")

    def load_lca_data(
        self, material_data_path: str, energy_data_path: str, process_data_path: str
    ) -> None:
        """Load all LCA-related data files."""
        try:
            self.lca_data["material_flow"] = self.data_loader.load_material_data(
                material_data_path
            )
            self.lca_data["energy_consumption"] = self.data_loader.load_energy_data(
                energy_data_path
            )
            self.lca_data["process_data"] = self.data_loader.load_process_data(
                process_data_path
            )
            self.logger.info("Successfully loaded all LCA data files")
        except Exception as e:
            self.logger.error(f"Error loading LCA data: {str(e)}")
            raise DataError(f"LCA data loading failed: {str(e)}")

    def load_production_data(self, file_path: Optional[str] = None) -> None:
        """Load production data from file."""
        try:
            if file_path is None:
                file_path = "test_production_data.csv"  # Default test file
            self.production_data = self.data_loader.load_production_data(file_path)
            self.logger.info("Successfully loaded production data")
        except Exception as e:
            self.logger.error(f"Error loading production data: {str(e)}")
            raise DataError(f"Production data loading failed: {str(e)}")

    def export_analysis_report(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Export comprehensive analysis report.

        Args:
            output_path: Optional path for report output (str or Path)
        """
        try:
            # Convert str to Path if needed
            path = Path(output_path) if output_path else None

            metrics = self.analyze_manufacturing_performance()
            self.report_generator.generate_comprehensive_report(
                metrics, output_dir=path
            )
            self.logger.info(f"Successfully exported analysis report to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting analysis report: {str(e)}")
            raise ProcessError(f"Report export failed: {str(e)}")

    def generate_visualization(self, metric_type: str, save_path: str) -> None:
        """Generate visualization for specified metric type."""
        try:
            if metric_type == "production":
                self.visualizer.visualize_production_trends(
                    self.production_data, save_path
                )
            elif metric_type == "energy":
                self.visualizer.visualize_energy_patterns(self.energy_data, save_path)
            elif metric_type == "quality":
                self.visualizer.visualize_quality_metrics(
                    self.quality_data,
                    analyzer=self.quality_analyzer,
                    save_path=save_path,
                )
            elif metric_type == "sustainability":
                self.visualizer.visualize_sustainability_indicators(
                    self.material_flow, self.energy_data, save_path
                )
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
        except Exception as e:
            self.logger.error(f"Error generating visualization: {str(e)}")
            raise ProcessError(f"Visualization generation failed: {str(e)}")

    def generate_comprehensive_report(self, output_path: str) -> None:
        """Generate a comprehensive analysis report including all metrics."""
        try:
            # Collect all metrics
            metrics = {
                "efficiency_metrics": self.analyze_efficiency(),
                "quality_metrics": self.analyze_quality_metrics(),
                "sustainability_metrics": self.calculate_sustainability_metrics(),
            }

            # Use pandas to write to Excel
            with pd.ExcelWriter(output_path) as writer:
                for metric_type, data in metrics.items():
                    if isinstance(data, dict) and not any(
                        key == "error" for key in data.keys()
                    ):
                        pd.DataFrame([data]).to_excel(writer, sheet_name=metric_type)

            self.logger.info(f"Comprehensive report generated at: {output_path}")

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            raise ProcessError(f"Report generation failed: {str(e)}")
