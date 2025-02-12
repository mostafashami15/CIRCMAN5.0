# src/circman5/manufacturing/analyzers/efficiency.py

"""
Efficiency analyzer for manufacturing processes.
"""
from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict, Optional
from ...utils.logging_config import setup_logger
from circman5.utils.errors import ValidationError


class EfficiencyAnalyzer:
    """Analyzes manufacturing efficiency metrics."""

    def __init__(self):
        """Initialize efficiency analyzer."""
        self.metrics = {}
        self.logger = setup_logger("efficiency_analyzer")

    def analyze_batch_efficiency(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze batch efficiency metrics."""
        try:
            # Validate input data
            if data.empty:
                return {}

            # Validate required columns
            required_columns = ["input_amount", "output_amount"]
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")

            # Validate data values
            if (data["input_amount"] <= 0).any():
                raise ValidationError("Input amounts must be positive")
            if (data["output_amount"] < 0).any():
                raise ValidationError("Output amounts cannot be negative")

            # Calculate metrics
            yield_rate = (
                (data["output_amount"].mean() / data["input_amount"].mean()) * 100
                if data["input_amount"].mean() > 0
                else 0
            )

            metrics = {
                "yield_rate": float(yield_rate),
                "output_amount": float(data["output_amount"].mean()),
                "input_amount": float(data["input_amount"].mean()),
            }

            # Add optional metrics if columns exist
            if "cycle_time" in data.columns:
                metrics["cycle_time_efficiency"] = float(
                    data["output_amount"].sum() / data["cycle_time"].sum()
                )
            if "energy_used" in data.columns:
                metrics["energy_efficiency"] = float(
                    data["output_amount"].sum() / data["energy_used"].sum()
                )

            self.logger.info(f"Efficiency analysis completed: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error in efficiency analysis: {str(e)}")
            raise

    def calculate_basic_efficiency(
        self, production_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate basic efficiency metrics."""
        valid_output = max(0, production_data["output_amount"].mean())
        valid_input = max(0, production_data["input_amount"].mean())

        # Calculate yield rate
        yield_rate = (valid_output / valid_input * 100) if valid_input > 0 else 0

        return {
            "yield_rate": yield_rate,
            "output_amount": valid_output,
            "input_amount": valid_input,
        }

    def calculate_cycle_time_efficiency(self, production_data: pd.DataFrame) -> float:
        """Calculate cycle time efficiency."""
        valid_output = max(0, production_data["output_amount"].mean())
        valid_cycle_time = max(0.1, production_data["cycle_time"].mean())
        return max(0, valid_output / valid_cycle_time)

    def calculate_energy_efficiency(self, production_data: pd.DataFrame) -> float:
        """Calculate energy efficiency."""
        valid_output = max(0, production_data["output_amount"].mean())
        valid_energy = max(0.1, production_data["energy_used"].mean())
        return max(0, valid_output / valid_energy)

    def calculate_overall_efficiency(self, production_data: pd.DataFrame) -> float:
        """
        Calculate overall manufacturing efficiency.

        Returns:
            float: Efficiency percentage
        """
        if production_data.empty:
            return 0.0

        return (
            production_data["output_amount"] / production_data["input_amount"]
        ).mean() * 100

    def plot_efficiency_trends(
        self, efficiency_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Plot efficiency metrics trends visualization.

        Args:
            efficiency_data: DataFrame containing efficiency metrics
            save_path: Optional path to save visualization
        """
        try:
            if efficiency_data.empty:
                print("No efficiency data to plot")  # Replace logger with print for now
                return

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Plot yield rate trend
            if "yield_rate" in efficiency_data.columns:
                daily_yield = efficiency_data.groupby(
                    pd.Grouper(key="timestamp", freq="D")
                )["yield_rate"].mean()
                daily_yield.plot(ax=axes[0, 0], title="Daily Yield Rate")
                axes[0, 0].set_ylabel("Yield Rate (%)")
                axes[0, 0].grid(True)

                # Add padding to prevent identical ymin/ymax
                ymin, ymax = axes[0, 0].get_ylim()
                if ymin == ymax:
                    ymin -= 1
                    ymax += 1
                axes[0, 0].set_ylim(ymin, ymax)

            # Repeat similar adjustments for other plots
            # Output vs Input plot
            if (
                "input_amount" in efficiency_data.columns
                and "output_amount" in efficiency_data.columns
            ):
                axes[0, 1].scatter(
                    efficiency_data["input_amount"],
                    efficiency_data["output_amount"],
                    alpha=0.5,
                )
                axes[0, 1].set_title("Output vs Input Amount")
                axes[0, 1].set_xlabel("Input Amount")
                axes[0, 1].set_ylabel("Output Amount")
                axes[0, 1].grid(True)

                # Add padding to X and Y axis limits
                xmin, xmax = axes[0, 1].get_xlim()
                ymin, ymax = axes[0, 1].get_ylim()
                if xmin == xmax:
                    xmin -= 1
                    xmax += 1
                if ymin == ymax:
                    ymin -= 1
                    ymax += 1
                axes[0, 1].set_xlim(xmin, xmax)
                axes[0, 1].set_ylim(ymin, ymax)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Error plotting efficiency trends: {str(e)}")

    def test_efficiency_visualization(analyzer, sample_production_data, visualizations_dir):  # type: ignore
        """Test efficiency visualization functionality."""
        # Generate visualization path
        viz_path = visualizations_dir / "efficiency_trends.png"

        # Test visualization generation
        analyzer.plot_efficiency_trends(sample_production_data, str(viz_path))

        # Verify file was created
        assert viz_path.exists()

        # Test handling of empty data
        empty_df = pd.DataFrame()
        analyzer.plot_efficiency_trends(empty_df, str(viz_path))
