# src/circman5/manufacturing/digital_twin/integration/lca_integration.py

"""
LCA Integration module for CIRCMAN5.0 Digital Twin.

This module handles the integration between the digital twin and lifecycle assessment (LCA) components,
enabling environmental impact calculations based on digital twin data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Mapping, TYPE_CHECKING

# Only import type hints during type checking to avoid circular imports
if TYPE_CHECKING:
    from ..core.twin_core import DigitalTwin

import pandas as pd
import datetime
import json
from pathlib import Path

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService
from ...lifecycle.lca_analyzer import LCAAnalyzer, LifeCycleImpact
from ...lifecycle.visualizer import LCAVisualizer


class LCAIntegration:
    """
    Integrates the digital twin with lifecycle assessment components.

    This class handles data extraction from the digital twin for LCA processing,
    sends data to LCA analysis, and incorporates results back into the digital twin.

    Attributes:
        digital_twin: Reference to the DigitalTwin instance
        lca_analyzer: LCAAnalyzer instance for impact calculations
        lca_visualizer: LCAVisualizer instance for visualization
        constants: ConstantsService for accessing configurations
        logger: Logger instance for this class
    """

    def __init__(
        self,
        digital_twin: "DigitalTwin",  # Use string literal for forward reference
        lca_analyzer: Optional[LCAAnalyzer] = None,
        lca_visualizer: Optional[LCAVisualizer] = None,
    ):
        """
        Initialize the LCA integration.

        Args:
            digital_twin: Digital Twin instance to integrate with
            lca_analyzer: Optional LCAAnalyzer instance (created if not provided)
            lca_visualizer: Optional LCAVisualizer instance (created if not provided)
        """
        self.digital_twin = digital_twin
        self.constants = ConstantsService()
        self.logger = setup_logger("lca_integration")

        # Initialize LCA components
        self.lca_analyzer = lca_analyzer or LCAAnalyzer()
        self.lca_visualizer = lca_visualizer or LCAVisualizer()

        # Initialize storage for LCA results
        self.lca_results_history: List[Dict[str, Any]] = []

        self.logger.info("LCA Integration initialized")

    def extract_material_data_from_state(
        self, state: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Extract material data from digital twin state for LCA calculations.

        Args:
            state: Optional state dictionary (uses current state if None)

        Returns:
            pd.DataFrame: Material flow data frame for LCA calculations
        """
        # Get current state if not provided
        if state is None:
            state = self.digital_twin.get_current_state()

        # Check if state is None after retrieval
        if state is None:
            self.logger.warning(
                "Received None state in extract_material_data_from_state"
            )
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=[
                    "batch_id",
                    "timestamp",
                    "material_type",
                    "quantity_used",
                    "waste_generated",
                    "recycled_amount",
                ]
            )

        # Extract timestamp
        timestamp = state.get("timestamp", datetime.datetime.now().isoformat())
        if isinstance(timestamp, str):
            try:
                timestamp_obj = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp_obj = datetime.datetime.now()
        else:
            timestamp_obj = timestamp

        # Extract batch ID (use timestamp if not available)
        batch_id = state.get(
            "batch_id", f"batch_{timestamp_obj.strftime('%Y%m%d%H%M%S')}"
        )

        # Initialize empty dataframe
        material_data = []

        # Extract materials data
        if "materials" in state:
            materials = state["materials"]
            for material_name, material_info in materials.items():
                if isinstance(material_info, dict):
                    # Calculate waste and recycling based on inventory
                    inventory = float(material_info.get("inventory", 0))
                    quality = float(material_info.get("quality", 0.9))

                    # Estimate waste as a function of quality (lower quality = more waste)
                    waste_rate = max(0.05, 1.0 - quality)
                    waste_generated = inventory * waste_rate

                    # Estimate recycling based on material type
                    # This is a simplified model using values from constants service
                    recycling_benefit_factors = self.constants.get_constant(
                        "impact_factors", "RECYCLING_BENEFIT_FACTORS"
                    )

                    # Map material name to recycling material type
                    material_type_mapping = {
                        "silicon_wafer": "silicon",
                        "solar_glass": "glass",
                        "tempered_glass": "glass",
                        "aluminum_frame": "aluminum",
                        "copper_wiring": "copper",
                        "backsheet": "plastic",
                        "eva_sheet": "plastic",
                        "junction_box": "plastic",
                    }

                    # Get recycling rate based on material type
                    recycling_material = material_type_mapping.get(
                        material_name, "plastic"
                    )
                    # Use a default recycling rate of 0.5 if not found
                    recycling_rate = 0.5

                    # Calculate recycled amount
                    recycled_amount = waste_generated * recycling_rate

                    material_data.append(
                        {
                            "batch_id": batch_id,
                            "timestamp": timestamp_obj,
                            "material_type": material_name,
                            "quantity_used": inventory,
                            "waste_generated": waste_generated,
                            "recycled_amount": recycled_amount,
                        }
                    )

        # Create dataframe
        if not material_data:
            # Create empty dataframe with required columns
            return pd.DataFrame(
                columns=[
                    "batch_id",
                    "timestamp",
                    "material_type",
                    "quantity_used",
                    "waste_generated",
                    "recycled_amount",
                ]
            )

        df = pd.DataFrame(material_data)
        self.logger.debug(f"Extracted material data from state: {len(df)} rows")
        return df

    def extract_energy_data_from_state(
        self, state: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Extract energy consumption data from digital twin state for LCA calculations.

        Args:
            state: Optional state dictionary (uses current state if None)

        Returns:
            pd.DataFrame: Energy consumption data frame for LCA calculations
        """
        # Get current state if not provided
        if state is None:
            state = self.digital_twin.get_current_state()

        # Check if state is None after retrieval
        if state is None:
            self.logger.warning("Received None state in extract_energy_data_from_state")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=[
                    "batch_id",
                    "timestamp",
                    "energy_source",
                    "energy_consumption",
                    "process_stage",
                ]
            )

        # Extract timestamp
        timestamp = state.get("timestamp", datetime.datetime.now().isoformat())
        if isinstance(timestamp, str):
            try:
                timestamp_obj = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp_obj = datetime.datetime.now()
        else:
            timestamp_obj = timestamp

        # Extract batch ID (use timestamp if not available)
        batch_id = state.get(
            "batch_id", f"batch_{timestamp_obj.strftime('%Y%m%d%H%M%S')}"
        )

        # Initialize energy data
        energy_data = []

        # Extract energy consumption from production line
        if "production_line" in state:
            prod_line = state["production_line"]
            if "energy_consumption" in prod_line:
                energy_consumption = float(prod_line["energy_consumption"])

                # Default to grid electricity if source not specified
                energy_source = prod_line.get("energy_source", "grid_electricity")

                energy_data.append(
                    {
                        "batch_id": batch_id,
                        "timestamp": timestamp_obj,
                        "energy_source": energy_source,
                        "energy_consumption": energy_consumption,
                        "process_stage": "production",
                    }
                )

        # Create dataframe
        if not energy_data:
            # Create empty dataframe with required columns
            return pd.DataFrame(
                columns=[
                    "batch_id",
                    "timestamp",
                    "energy_source",
                    "energy_consumption",
                    "process_stage",
                ]
            )

        df = pd.DataFrame(energy_data)
        self.logger.debug(f"Extracted energy data from state: {len(df)} rows")
        return df

    def perform_lca_analysis(
        self,
        state: Optional[Dict[str, Any]] = None,
        batch_id: Optional[str] = None,
        save_results: bool = True,
        output_dir: Optional[Path] = None,
    ) -> LifeCycleImpact:
        """
        Perform lifecycle assessment analysis based on digital twin state.

        Args:
            state: Optional state dictionary (uses current state if None)
            batch_id: Optional batch identifier for the analysis
            save_results: Whether to save results to file
            output_dir: Optional directory to save results (uses results_manager if None)

        Returns:
            LifeCycleImpact: Results of the lifecycle assessment
        """
        try:
            # Get current state if not provided
            if state is None:
                state = self.digital_twin.get_current_state()

            # Check if state is None after retrieval
            if state is None:
                self.logger.warning("Cannot perform LCA analysis: state is None")
                # Return empty impact object
                return LifeCycleImpact(
                    manufacturing_impact=0.0,
                    use_phase_impact=0.0,
                    end_of_life_impact=0.0,
                )

            # Extract timestamp for batch_id if not provided
            if batch_id is None:
                timestamp = state.get("timestamp", datetime.datetime.now().isoformat())
                if isinstance(timestamp, str):
                    try:
                        timestamp_obj = datetime.datetime.fromisoformat(timestamp)
                    except ValueError:
                        timestamp_obj = datetime.datetime.now()
                else:
                    timestamp_obj = timestamp

                batch_id = f"batch_{timestamp_obj.strftime('%Y%m%d%H%M%S')}"

            # Extract material and energy data
            material_data = self.extract_material_data_from_state(state)
            energy_data = self.extract_energy_data_from_state(state)

            # Get aggregated material inputs
            material_inputs = self._aggregate_material_quantities(material_data)

            # Get total energy consumption
            total_energy = (
                energy_data["energy_consumption"].sum()
                if not energy_data.empty
                else 0.0
            )

            # Calculate recycling rates
            recycling_rates = self.lca_analyzer.calculate_recycling_rates(material_data)

            # Set default parameters for LCA calculation
            lifetime_years = 25.0  # Default PV panel lifetime
            transport_distance = 100.0  # Default transport distance in km

            # Calculate annual energy generation based on material inputs
            annual_energy_generation = self.lca_analyzer.calculate_energy_generation(
                material_inputs
            )

            # Get grid carbon intensity from constants
            grid_intensities = self.constants.get_constant(
                "impact_factors", "GRID_CARBON_INTENSITIES"
            )
            grid_carbon_intensity = grid_intensities.get("eu_average", 0.275)

            # Perform lifecycle assessment
            impact = self.lca_analyzer.perform_full_lca(
                material_inputs=material_inputs,
                energy_consumption=total_energy,
                lifetime_years=lifetime_years,
                annual_energy_generation=annual_energy_generation,
                grid_carbon_intensity=grid_carbon_intensity,
                recycling_rates=recycling_rates,
                transport_distance=transport_distance,
            )

            # Save results if requested
            if save_results:
                if output_dir is None:
                    # Use results_manager
                    lca_results_dir = results_manager.get_path("lca_results")
                else:
                    lca_results_dir = output_dir

                self.lca_analyzer.save_results(
                    impact=impact, batch_id=batch_id, output_dir=lca_results_dir
                )

                # Create visualizations
                self.lca_visualizer.create_comprehensive_report(
                    impact_data=impact.to_dict(),
                    material_data=material_data,
                    energy_data=energy_data,
                    output_dir=lca_results_dir,
                    batch_id=batch_id,
                )

            # Store results in history
            self.lca_results_history.append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "batch_id": batch_id,
                    "impact": impact.to_dict(),
                    "material_inputs": material_inputs,
                    "energy_consumption": total_energy,
                }
            )

            # Save history to file
            self._save_lca_history()

            self.logger.info(
                f"LCA analysis completed for {batch_id}. "
                f"Total impact: {impact.total_carbon_footprint:.2f} kg CO2-eq"
            )

            return impact

        except Exception as e:
            self.logger.error(f"Error performing LCA analysis: {str(e)}")
            # Return empty impact object
            return LifeCycleImpact(
                manufacturing_impact=0.0, use_phase_impact=0.0, end_of_life_impact=0.0
            )

    def _aggregate_material_quantities(
        self, material_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Aggregate material quantities by type from material flow data.

        Args:
            material_data: DataFrame with material flow data

        Returns:
            Dict[str, float]: Dictionary mapping material types to total quantities
        """
        if material_data.empty:
            return {}

        try:
            material_totals = material_data.groupby("material_type")[
                "quantity_used"
            ].sum()
            return material_totals.to_dict()
        except Exception as e:
            self.logger.error(f"Error aggregating material quantities: {str(e)}")
            return {}

    def _save_lca_history(self) -> None:
        """Save LCA analysis history to file using results_manager."""
        try:
            # Create filename with timestamp
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lca_history_{timestamp_str}.json"

            # Save to file
            with open(filename, "w") as f:
                json.dump(self.lca_results_history, f, indent=2)

            # Save using results_manager
            results_manager.save_file(Path(filename), "lca_results")

            # Clean up temporary file
            Path(filename).unlink()

            self.logger.debug(f"Saved LCA history to results manager")

        except Exception as e:
            self.logger.error(f"Error saving LCA history: {str(e)}")

    def generate_lca_report(self, num_results: int = 5) -> Dict[str, Any]:
        """
        Generate a comprehensive LCA report.

        Args:
            num_results: Number of most recent results to include in the report

        Returns:
            Dict[str, Any]: Report data
        """
        if not self.lca_results_history:
            return {"error": "No LCA analysis history available"}

        try:
            # Get most recent results
            recent_results = self.lca_results_history[-num_results:]

            # Extract impact trends
            impact_trends = {
                "manufacturing_impact": [
                    r["impact"]["Manufacturing Impact"] for r in recent_results
                ],
                "use_phase_impact": [
                    r["impact"]["Use Phase Impact"] for r in recent_results
                ],
                "end_of_life_impact": [
                    r["impact"]["End of Life Impact"] for r in recent_results
                ],
                "total_carbon_footprint": [
                    r["impact"]["Total Carbon Footprint"] for r in recent_results
                ],
            }

            # Calculate averages
            avg_impacts = {
                key: sum(values) / len(values) if values else 0
                for key, values in impact_trends.items()
            }

            # Generate report
            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_analyses": len(self.lca_results_history),
                "average_impacts": avg_impacts,
                "latest_analysis": self.lca_results_history[-1]
                if self.lca_results_history
                else None,
                "impact_trends": impact_trends,
                "current_state": self.digital_twin.get_current_state(),
            }

            # Save report
            report_file = "lca_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            results_manager.save_file(Path(report_file), "reports")
            Path(report_file).unlink()

            return report

        except Exception as e:
            self.logger.error(f"Error generating LCA report: {str(e)}")
            return {"error": str(e)}

    def compare_scenarios(
        self,
        baseline_state: Dict[str, Any],
        alternative_state: Dict[str, Any],
        scenario_name: str = "scenario_comparison",
    ) -> Dict[str, Any]:
        """
        Compare LCA impacts between two different digital twin states.

        Args:
            baseline_state: Baseline state for comparison
            alternative_state: Alternative state for comparison
            scenario_name: Name of the comparison scenario

        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            # Perform LCA for both scenarios
            baseline_impact = self.perform_lca_analysis(
                state=baseline_state,
                batch_id=f"{scenario_name}_baseline",
                save_results=True,
            )

            alternative_impact = self.perform_lca_analysis(
                state=alternative_state,
                batch_id=f"{scenario_name}_alternative",
                save_results=True,
            )

            # Calculate differences
            impact_differences = {
                "manufacturing_impact": alternative_impact.manufacturing_impact
                - baseline_impact.manufacturing_impact,
                "use_phase_impact": alternative_impact.use_phase_impact
                - baseline_impact.use_phase_impact,
                "end_of_life_impact": alternative_impact.end_of_life_impact
                - baseline_impact.end_of_life_impact,
                "total_carbon_footprint": alternative_impact.total_carbon_footprint
                - baseline_impact.total_carbon_footprint,
            }

            # Calculate percentage differences
            percent_differences = {}
            for key, value in impact_differences.items():
                baseline_value = getattr(baseline_impact, key, 0)
                if baseline_value != 0:
                    percent_differences[key] = (value / baseline_value) * 100
                else:
                    percent_differences[key] = 0

            # Create comparison report
            comparison = {
                "timestamp": datetime.datetime.now().isoformat(),
                "scenario_name": scenario_name,
                "baseline_impact": baseline_impact.to_dict(),
                "alternative_impact": alternative_impact.to_dict(),
                "absolute_differences": impact_differences,
                "percent_differences": percent_differences,
            }

            # Save comparison report
            report_file = f"lca_comparison_{scenario_name}.json"
            with open(report_file, "w") as f:
                json.dump(comparison, f, indent=2)

            results_manager.save_file(Path(report_file), "reports")
            Path(report_file).unlink()

            self.logger.info(
                f"LCA comparison completed for {scenario_name}. "
                f"Total impact difference: {impact_differences['total_carbon_footprint']:.2f} kg CO2-eq "
                f"({percent_differences['total_carbon_footprint']:.2f}%)"
            )

            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing scenarios: {str(e)}")
            return {"error": str(e)}

    def simulate_lca_improvements(
        self, improvement_scenarios: Dict[str, Dict[str, float]]
    ) -> Mapping[str, Union[LifeCycleImpact, str]]:
        """
        Simulate LCA improvements based on various improvement scenarios.

        Args:
            improvement_scenarios: Dictionary mapping scenario names to parameter adjustments

        Returns:
            Mapping[str, Union[LifeCycleImpact, str]]: Results for each scenario
        """
        try:
            # Get current state as baseline
            baseline_state = self.digital_twin.get_current_state()

            # Perform baseline LCA
            baseline_impact = self.perform_lca_analysis(
                state=baseline_state, batch_id="baseline", save_results=True
            )

            # Initialize results dictionary
            results = {"baseline": baseline_impact}

            # Process each improvement scenario
            for scenario_name, adjustments in improvement_scenarios.items():
                # Create a modified state
                modified_state = self._apply_scenario_adjustments(
                    baseline_state, adjustments
                )

                # Perform LCA for the scenario
                scenario_impact = self.perform_lca_analysis(
                    state=modified_state,
                    batch_id=f"scenario_{scenario_name}",
                    save_results=True,
                )

                # Store results
                results[scenario_name] = scenario_impact

            # Create comparison report
            self._generate_scenarios_comparison(results)

            return results

        except Exception as e:
            self.logger.error(f"Error simulating LCA improvements: {str(e)}")
            return {"error": str(e)}

    def _apply_scenario_adjustments(
        self, base_state: Dict[str, Any], adjustments: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Apply scenario adjustments to create a modified state.

        Args:
            base_state: Base state to modify
            adjustments: Dictionary of adjustments to apply

        Returns:
            Dict[str, Any]: Modified state
        """
        import copy

        # Create a deep copy of the state
        modified_state = copy.deepcopy(base_state)

        # Apply material adjustments
        if "materials" in modified_state:
            for material_name, material_info in modified_state["materials"].items():
                if isinstance(material_info, dict):
                    # Adjust inventory if specified
                    if f"{material_name}_quantity" in adjustments:
                        adjustment_factor = adjustments[f"{material_name}_quantity"]
                        material_info["inventory"] = (
                            material_info.get("inventory", 0) * adjustment_factor
                        )

                    # Adjust quality if specified
                    if f"{material_name}_quality" in adjustments:
                        adjustment_factor = adjustments[f"{material_name}_quality"]
                        material_info["quality"] = min(
                            1.0, material_info.get("quality", 0.9) * adjustment_factor
                        )

        # Apply energy adjustments
        if "production_line" in modified_state:
            if (
                "energy_efficiency" in adjustments
                and "energy_consumption" in modified_state["production_line"]
            ):
                adjustment_factor = adjustments["energy_efficiency"]
                modified_state["production_line"]["energy_consumption"] = (
                    modified_state["production_line"]["energy_consumption"]
                    / adjustment_factor
                )

            # Apply other production line adjustments
            for key, adjustment_factor in adjustments.items():
                if key in modified_state["production_line"]:
                    modified_state["production_line"][key] = (
                        modified_state["production_line"][key] * adjustment_factor
                    )

        return modified_state

    def _generate_scenarios_comparison(
        self, results: Dict[str, LifeCycleImpact]
    ) -> None:
        """
        Generate a comparison report for multiple scenarios.

        Args:
            results: Dictionary mapping scenario names to LCA results
        """
        try:
            # Extract baseline for comparison
            baseline = results.get("baseline")
            if not baseline:
                self.logger.warning("No baseline found in results")
                return

            # Create comparison data
            comparison = {
                "timestamp": datetime.datetime.now().isoformat(),
                "scenarios": list(results.keys()),
                "baseline": baseline.to_dict(),
                "scenario_results": {
                    name: impact.to_dict() for name, impact in results.items()
                },
                "comparisons": {},
            }

            # Calculate comparisons to baseline
            for scenario_name, impact in results.items():
                if scenario_name == "baseline":
                    continue

                # Calculate differences
                differences = {
                    "manufacturing_impact": impact.manufacturing_impact
                    - baseline.manufacturing_impact,
                    "use_phase_impact": impact.use_phase_impact
                    - baseline.use_phase_impact,
                    "end_of_life_impact": impact.end_of_life_impact
                    - baseline.end_of_life_impact,
                    "total_carbon_footprint": impact.total_carbon_footprint
                    - baseline.total_carbon_footprint,
                }

                # Calculate percentage differences
                percent_differences = {}
                for key, value in differences.items():
                    baseline_value = getattr(baseline, key, 0)
                    if baseline_value != 0:
                        percent_differences[key] = (value / baseline_value) * 100
                    else:
                        percent_differences[key] = 0

                comparison["comparisons"][scenario_name] = {
                    "absolute_differences": differences,
                    "percent_differences": percent_differences,
                }

            # Save comparison report
            report_file = f"lca_scenarios_comparison.json"
            with open(report_file, "w") as f:
                json.dump(comparison, f, indent=2)

            results_manager.save_file(Path(report_file), "reports")
            Path(report_file).unlink()

            self.logger.info(f"Generated scenarios comparison report")

        except Exception as e:
            self.logger.error(f"Error generating scenarios comparison: {str(e)}")
