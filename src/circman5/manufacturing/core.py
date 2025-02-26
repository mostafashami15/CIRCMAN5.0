# src/circman5/manufacturing/core.py

from typing import Any, Dict, Optional, List, Tuple, Union, Mapping
from pathlib import Path
import numpy as np
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
from ..utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from .digital_twin.core.twin_core import DigitalTwin, DigitalTwinConfig
from .digital_twin.core.synchronization import SynchronizationManager, SyncMode
import json
import random
import datetime


class SoliTekManufacturingAnalysis:
    """
    Core class for SoliTek manufacturing analysis, integrating all analysis modules.
    """

    def __init__(self):
        self.logger = setup_logger("solitek_manufacturing")

        # Initialize constants service
        self.constants = ConstantsService()

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

        # Initialize digital twin system
        self.digital_twin = None
        self.sync_manager = None
        self._initialize_digital_twin()

    def _initialize_digital_twin(self):
        """Initialize the Digital Twin system."""
        try:
            # Get digital twin configuration from constants service
            self.digital_twin_config = DigitalTwinConfig.from_constants()

            # Initialize the Digital Twin
            self.digital_twin = DigitalTwin(self.digital_twin_config)
            self.digital_twin.initialize()

            # Initialize synchronization manager
            self.sync_manager = SynchronizationManager(self.digital_twin.state_manager)

            # Register data sources for synchronization
            self.sync_manager.register_data_source(
                "production", self._get_production_data
            )
            self.sync_manager.register_data_source("quality", self._get_quality_data)
            self.sync_manager.register_data_source("material", self._get_material_data)
            self.sync_manager.register_data_source("energy", self._get_energy_data)

            # Start synchronization if configured to do so
            if self.digital_twin_config.synchronization_mode == "real_time":
                self.sync_manager.start_synchronization()

            self.logger.info("Digital Twin system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Digital Twin: {str(e)}")
            # Continue operating without Digital Twin

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

            # Update digital twin with loaded data
            self._update_digital_twin_with_loaded_data()

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise DataError(f"Data loading failed: {str(e)}")

    def _update_digital_twin_with_loaded_data(self) -> None:
        """Update the Digital Twin with the loaded data."""
        if self.digital_twin is None or self.sync_manager is None:
            return  # Digital Twin not initialized

        try:
            # Perform immediate synchronization
            self.sync_manager.synchronize_now(save_results=True)
            self.logger.info("Digital Twin updated with loaded data")
        except Exception as e:
            self.logger.error(f"Failed to update Digital Twin: {str(e)}")

    def _get_production_data(self) -> Dict[str, Any]:
        """Get production data for Digital Twin synchronization."""
        if self.production_data.empty:
            return {}

        # Return the most recent production data
        latest_data = self.production_data.iloc[-1].to_dict()
        return {"production": latest_data}

    def _get_quality_data(self) -> Dict[str, Any]:
        """Get quality data for Digital Twin synchronization."""
        if self.quality_data.empty:
            return {}

        latest_data = self.quality_data.iloc[-1].to_dict()
        return {"quality": latest_data}

    def _get_material_data(self) -> Dict[str, Any]:
        """Get material flow data for Digital Twin synchronization."""
        if self.material_flow.empty:
            return {}

        latest_data = self.material_flow.iloc[-1].to_dict()
        return {"material": latest_data}

    def _get_energy_data(self) -> Dict[str, Any]:
        """Get energy consumption data for Digital Twin synchronization."""
        if self.energy_data.empty:
            return {}

        latest_data = self.energy_data.iloc[-1].to_dict()
        return {"energy": latest_data}

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

            # Calculate annual energy generation with validation
            annual_energy_generation = self.lca_analyzer.calculate_energy_generation(
                material_inputs
            )

            # Ensure annual energy generation is positive
            if annual_energy_generation <= 0:
                self.logger.warning(
                    "Calculated annual energy generation was invalid, using default value"
                )
                annual_energy_generation = 1000.0  # Use reasonable default

            # Get default values from constants service
            grid_carbon_intensities = self.constants.get_constant(
                "impact_factors", "GRID_CARBON_INTENSITIES"
            )
            default_grid_intensity = grid_carbon_intensities.get("eu_average", 0.5)

            # Perform full LCA calculation
            impact = self.lca_analyzer.perform_full_lca(
                material_inputs=material_inputs,
                energy_consumption=total_energy,
                lifetime_years=25.0,  # Standard PV panel lifetime
                annual_energy_generation=annual_energy_generation,
                grid_carbon_intensity=default_grid_intensity,
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
            # Use results_manager if no output_dir specified
            if output_dir is None:
                output_dir = results_manager.get_path("reports")
            else:
                output_dir = Path(output_dir)

            # Collect all metrics
            performance_metrics = self.analyze_manufacturing_performance()

            # Generate reports using report generator
            self.report_generator.generate_comprehensive_report(
                performance_metrics, output_dir=output_dir
            )

            # Rename the generated report file to "analysis_report.xlsx"
            generated_file = output_dir / "comprehensive_analysis.xlsx"
            expected_file = output_dir / "analysis_report.xlsx"
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
            viz_dir = (
                Path(output_dir)
                if output_dir
                else results_manager.get_path("visualizations")
            )
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

    def generate_enhanced_dashboard(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Generate an enhanced dashboard with advanced visualizations.

        Args:
            save_path: Optional path to save the dashboard

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the TwinDashboard
            from .digital_twin.visualization.dashboard import TwinDashboard

            # Create dashboard with the state manager
            dashboard = TwinDashboard(self.digital_twin.state_manager)

            # Generate enhanced dashboard
            dashboard.generate_enhanced_dashboard(save_path)

            self.logger.info("Generated enhanced dashboard")
            return True

        except Exception as e:
            self.logger.error(f"Error generating enhanced dashboard: {str(e)}")
            return False

    def generate_material_flow_sankey(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Generate a Sankey diagram visualization of material flow.

        Args:
            save_path: Optional path to save the visualization

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the TwinVisualizer
            from .digital_twin.visualization.twin_visualizer import TwinVisualizer

            # Create visualizer with the state manager
            visualizer = TwinVisualizer(self.digital_twin.state_manager)

            # Generate Sankey diagram
            visualizer.visualize_material_flow_sankey(save_path)

            self.logger.info("Generated material flow Sankey diagram")
            return True

        except Exception as e:
            self.logger.error(
                f"Error generating material flow Sankey diagram: {str(e)}"
            )
            return False

    def generate_efficiency_heatmap(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Generate a heatmap visualization of efficiency metrics.

        Args:
            save_path: Optional path to save the visualization

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the TwinVisualizer
            from .digital_twin.visualization.twin_visualizer import TwinVisualizer

            # Create visualizer with the state manager
            visualizer = TwinVisualizer(self.digital_twin.state_manager)

            # Generate efficiency heatmap
            visualizer.visualize_efficiency_heatmap(save_path)

            self.logger.info("Generated efficiency metrics heatmap")
            return True

        except Exception as e:
            self.logger.error(f"Error generating efficiency heatmap: {str(e)}")
            return False

    def generate_process_visualization(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Generate process-specific visualizations.

        Args:
            save_path: Optional path to save the visualization

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the ProcessVisualizer
            from .digital_twin.visualization.process_visualizer import ProcessVisualizer

            # Create process visualizer with the state manager
            visualizer = ProcessVisualizer(self.digital_twin.state_manager)

            # Generate visualization
            visualizer.visualize_manufacturing_stages(save_path)

            self.logger.info("Generated process visualization")
            return True

        except Exception as e:
            self.logger.error(f"Error generating process visualization: {str(e)}")
            return False

    def compare_digital_twin_states(
        self,
        state1: Optional[Dict[str, Any]] = None,
        state2: Optional[Dict[str, Any]] = None,
        labels: Tuple[str, str] = ("Before", "After"),
        save_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Compare two digital twin states with visualization.

        Args:
            state1: First state to compare (uses earliest history state if None)
            state2: Second state to compare (uses current state if None)
            labels: Labels for the states in the visualization
            save_path: Optional path to save the visualization

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the TwinVisualizer
            from .digital_twin.visualization.twin_visualizer import TwinVisualizer

            # Get states if not provided
            if state1 is None:
                # Get earliest state from history
                history = self.digital_twin.state_manager.get_history(limit=20)
                if history:
                    state1 = history[0]
                else:
                    self.logger.warning("No historical states available for comparison")
                    return False

            if state2 is None:
                # Use current state
                state2 = self.digital_twin.state_manager.get_current_state()

            # Create visualizer with the state manager
            visualizer = TwinVisualizer(self.digital_twin.state_manager)

            # Generate comparison visualization
            visualizer.visualize_state_comparison(state1, state2, labels, save_path)

            self.logger.info("Generated state comparison visualization")
            return True

        except Exception as e:
            self.logger.error(f"Error comparing digital twin states: {str(e)}")
            return False

    def analyze_parameter_sensitivity(
        self,
        parameter: str,
        min_value: float,
        max_value: float,
        steps: int = 5,
        target_metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Analyze and visualize how changes to a parameter affect different metrics.
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the TwinVisualizer
            from .digital_twin.visualization.twin_visualizer import TwinVisualizer

            # Create value range as a list of evenly spaced values
            value_range = []
            for i in range(steps):
                # Calculate each value in the range
                val = min_value + (max_value - min_value) * i / (
                    steps - 1 if steps > 1 else 1
                )
                value_range.append(val)

            # Use default metrics if none provided
            if target_metrics is None:
                target_metrics = [
                    "production_line.production_rate",
                    "production_line.energy_consumption",
                    "production_line.temperature",
                ]

            # Create visualizer with the state manager
            visualizer = TwinVisualizer(self.digital_twin.state_manager)

            # Generate parameter sensitivity visualization
            visualizer.visualize_parameter_sensitivity(
                parameter, value_range, target_metrics, save_path
            )

            self.logger.info(
                f"Generated parameter sensitivity analysis for {parameter}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error in parameter sensitivity analysis: {str(e)}")
            return False

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

    def simulate_manufacturing_scenario(
        self, parameters: Optional[Dict[str, Any]] = None, steps: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Simulate a manufacturing scenario using the Digital Twin."""
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Run simulation with Digital Twin
            simulated_states = self.digital_twin.simulate(
                steps=steps, parameters=parameters
            )

            # Convert states to JSON-serializable format
            serializable_states = []
            for state in simulated_states:
                serializable_state = self._make_json_serializable(state)
                serializable_states.append(serializable_state)

            # Save simulation results using results_manager
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp_str}.json"

            # Use results_manager to get the temp directory
            temp_dir = results_manager.get_path("temp")
            temp_path = temp_dir / filename

            with open(temp_path, "w") as f:
                json.dump(serializable_states, f, indent=2)

            # Save to digital_twin directory
            results_manager.save_file(temp_path, "digital_twin")

            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

            self.logger.info(
                f"Simulation completed with {len(simulated_states)} states"
            )
            return simulated_states

        except Exception as e:
            self.logger.error(f"Error in manufacturing simulation: {str(e)}")
            raise ProcessError(f"Simulation failed: {str(e)}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert an object to a JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, "isoformat") and callable(obj.isoformat):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def optimize_using_digital_twin(
        self,
        current_params: Dict[str, float],
        constraints: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
        simulation_steps: int = 10,
    ) -> Dict[str, float]:
        """
        Optimize manufacturing parameters using the Digital Twin.

        Args:
            current_params: Current manufacturing parameters
            constraints: Optional parameter constraints
            simulation_steps: Number of simulation steps to run

        Returns:
            Optimized parameters
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Get current state from Digital Twin
            current_state = self.digital_twin.get_current_state()

            # Run multiple simulations with parameter variations
            best_params = current_params.copy()
            best_performance = self._evaluate_simulation_performance(
                [self.digital_twin.simulate(steps=1, parameters=current_params)[-1]]
            )

            # Simple parameter space exploration (could be enhanced with more advanced algorithms)
            for _ in range(5):  # Try 5 different parameter variations
                # Create parameter variation within constraints
                variation = {}
                for param, value in current_params.items():
                    if constraints and param in constraints:
                        constraint = constraints[param]
                        if isinstance(constraint, tuple) and len(constraint) == 2:
                            min_val, max_val = constraint
                            variation[param] = min_val + random.random() * (
                                max_val - min_val
                            )
                        else:
                            # Use constraint as target value with ±10% variation
                            variation[param] = float(constraint) * (
                                0.9 + 0.2 * random.random()
                            )
                    else:
                        # Default ±15% variation
                        variation[param] = value * (0.85 + 0.3 * random.random())

                # Simulate with these parameters
                simulated_states = self.digital_twin.simulate(
                    steps=simulation_steps, parameters=variation
                )

                # Evaluate performance
                performance = self._evaluate_simulation_performance(simulated_states)

                if performance > best_performance:
                    best_performance = performance
                    best_params = variation

            self.logger.info(
                f"Digital Twin optimization completed with performance score: {best_performance:.2f}"
            )
            return best_params

        except Exception as e:
            self.logger.error(f"Error in Digital Twin optimization: {str(e)}")
            raise ProcessError(f"Digital Twin optimization failed: {str(e)}")

    def _evaluate_simulation_performance(
        self, simulated_states: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate the performance of a simulation.

        Args:
            simulated_states: List of simulated states

        Returns:
            Performance score (higher is better)
        """
        if not simulated_states:
            return 0.0

        # Get the final state
        final_state = simulated_states[-1]

        # Initialize score
        score = 0.0

        # Production line performance
        if "production_line" in final_state:
            prod_line = final_state["production_line"]

            # Production rate
            if "production_rate" in prod_line:
                score += prod_line["production_rate"] * 0.5

            # Energy efficiency
            if (
                "energy_consumption" in prod_line
                and prod_line["energy_consumption"] > 0
            ):
                if "production_rate" in prod_line:
                    energy_efficiency = (
                        prod_line["production_rate"] / prod_line["energy_consumption"]
                    )
                    score += energy_efficiency * 50.0

            # Temperature optimization (penalize if outside optimal range)
            if "temperature" in prod_line:
                temp = prod_line["temperature"]
                if temp < 20 or temp > 25:
                    # Penalize temperatures outside optimal range
                    score -= abs(temp - 22.5) * 0.2

        # Material efficiency
        if "materials" in final_state:
            materials = final_state["materials"]
            for material_name, material_data in materials.items():
                if "quality" in material_data:
                    score += material_data["quality"] * 10.0

        return score

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

    def save_digital_twin_state(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Save the current Digital Twin state.

        Args:
            file_path: Optional path to save the state

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            return self.digital_twin.save_state(file_path)
        except Exception as e:
            self.logger.error(f"Failed to save Digital Twin state: {str(e)}")
            return False

    def load_digital_twin_state(self, file_path: Union[str, Path]) -> bool:
        """
        Load a Digital Twin state from a file.

        Args:
            file_path: Path to the state file

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            return self.digital_twin.load_state(file_path)
        except Exception as e:
            self.logger.error(f"Failed to load Digital Twin state: {str(e)}")
            return False

    def verify_digital_twin_integration(self) -> Dict[str, Any]:
        """
        Verify Digital Twin integration and return status.

        Returns:
            Dict with digital twin status information
        """
        status = {
            "initialized": self.digital_twin is not None,
            "synchronization_active": False,
            "current_state": None,
            "history_length": 0,
        }

        if self.digital_twin:
            # Get synchronization status
            if self.sync_manager:
                status["synchronization_active"] = self.sync_manager.is_running

            # Get current state summary
            current_state = self.digital_twin.get_current_state()
            if current_state:
                status["current_state"] = {
                    "timestamp": current_state.get("timestamp", "N/A"),
                    "system_status": current_state.get("system_status", "N/A"),
                    "components": list(current_state.keys()),
                }

            # Get history length
            status["history_length"] = len(
                self.digital_twin.state_manager.state_history
            )

        return status

    def export_analysis_report(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Export comprehensive analysis report.

        Args:
            output_path: Optional path for report output (str or Path)
        """
        try:
            # Use results_manager if no output_path specified
            if output_path is None:
                output_path = (
                    results_manager.get_path("reports") / "analysis_report.xlsx"
                )
            else:
                output_path = Path(output_path)

            metrics = self.analyze_manufacturing_performance()
            self.report_generator.generate_comprehensive_report(
                metrics, output_dir=output_path.parent
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

    def generate_comprehensive_report(self, output_path: Optional[str] = None) -> None:
        """Generate a comprehensive analysis report including all metrics."""
        try:
            # Use results_manager if no output_path specified
            if output_path is None:
                output_path = str(
                    results_manager.get_path("reports") / "comprehensive_report.xlsx"
                )

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

    def generate_digital_twin_visualization(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Generate a visualization of the current digital twin state.

        Args:
            save_path: Optional path to save the visualization

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the TwinVisualizer
            from .digital_twin.visualization.twin_visualizer import TwinVisualizer

            # Create visualizer with the state manager
            visualizer = TwinVisualizer(self.digital_twin.state_manager)

            # Generate visualization
            visualizer.visualize_current_state(save_path)

            self.logger.info("Generated Digital Twin visualization")
            return True

        except Exception as e:
            self.logger.error(f"Error generating Digital Twin visualization: {str(e)}")
            return False

    def generate_digital_twin_dashboard(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Generate a comprehensive dashboard for the digital twin.

        Args:
            save_path: Optional path to save the dashboard

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the TwinDashboard
            from .digital_twin.visualization.dashboard import TwinDashboard

            # Create dashboard with the state manager
            dashboard = TwinDashboard(self.digital_twin.state_manager)

            # Generate dashboard
            dashboard.generate_dashboard(save_path)

            self.logger.info("Generated Digital Twin dashboard")
            return True

        except Exception as e:
            self.logger.error(f"Error generating Digital Twin dashboard: {str(e)}")
            return False

    def visualize_digital_twin_history(
        self,
        metrics: List[str] = [
            "production_line.production_rate",
            "production_line.energy_consumption",
        ],
        limit: int = 20,
        save_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Visualize historical data from the digital twin.

        Args:
            metrics: List of metric paths to visualize
            limit: Maximum number of historical states to include
            save_path: Optional path to save the visualization

        Returns:
            bool: True if successful, False otherwise
        """
        if self.digital_twin is None:
            raise ValueError("Digital Twin not initialized")

        try:
            # Import the TwinVisualizer
            from .digital_twin.visualization.twin_visualizer import TwinVisualizer

            # Create visualizer with the state manager
            visualizer = TwinVisualizer(self.digital_twin.state_manager)

            # Generate visualization
            visualizer.visualize_historical_states(metrics, limit, save_path)

            self.logger.info("Generated Digital Twin historical visualization")
            return True

        except Exception as e:
            self.logger.error(
                f"Error generating Digital Twin historical visualization: {str(e)}"
            )
            return False
