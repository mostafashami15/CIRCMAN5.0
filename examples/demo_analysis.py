# examples/demo_analysis.py

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.result_paths import get_run_directory
from circman5.manufacturing.reporting.results_collector import ResultsCollector


def run_comprehensive_demo():
    print("Starting CIRCMAN5.0 Demo Analysis...")

    # Initialize the manufacturing analysis system
    analyzer = SoliTekManufacturingAnalysis()

    # Generate synthetic test data
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=30)

    # Generate complete dataset including LCA data
    dataset = generator.generate_complete_lca_dataset()

    # Load data into analyzer
    analyzer.production_data = dataset["production_data"]
    analyzer.energy_data = dataset["energy_consumption"]
    analyzer.quality_data = generator.generate_quality_data()
    analyzer.material_flow = dataset["material_flow"]
    analyzer.lca_data = dataset

    # Get run directory for saving results
    run_dir = get_run_directory()

    # Run comprehensive analysis
    print("\nPerforming manufacturing analysis...")
    performance_metrics = analyzer.analyze_manufacturing_performance()

    # Generate all visualizations
    print("\nGenerating visualizations...")
    analyzer.generate_visualization(
        "production", str(run_dir / "visualizations" / "production_trends.png")
    )
    analyzer.generate_visualization(
        "quality", str(run_dir / "visualizations" / "quality_metrics.png")
    )
    analyzer.generate_visualization(
        "energy", str(run_dir / "visualizations" / "energy_patterns.png")
    )
    analyzer.generate_visualization(
        "sustainability",
        str(run_dir / "visualizations" / "sustainability_indicators.png"),
    )

    # Perform lifecycle assessment
    print("\nPerforming lifecycle assessment...")
    lca_results = analyzer.perform_lifecycle_assessment(output_dir=run_dir)

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report_path = run_dir / "reports" / "comprehensive_analysis.xlsx"
    analyzer.generate_comprehensive_report(str(report_path))

    # Generate a summary of all results
    summary_path = run_dir / "analysis_summary.txt"
    generate_analysis_summary(run_dir, performance_metrics, lca_results, summary_path)

    print(f"\nAnalysis complete! Results saved in: {run_dir}")
    return performance_metrics, lca_results


def generate_analysis_summary(
    run_dir: Path, performance_metrics: dict, lca_results: object, output_path: Path
) -> None:
    """Generate a comprehensive summary of all analysis results."""
    summary = [
        "CIRCMAN5.0 Analysis Results Summary",
        "=================================",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "1. Manufacturing Performance",
        "-------------------------",
        "Efficiency Metrics:",
        f"- Yield Rate: {performance_metrics['efficiency']['yield_rate']:.2f}%",
        f"- Energy Efficiency: {performance_metrics['efficiency']['energy_efficiency']:.2f}",
        f"- Cycle Time Efficiency: {performance_metrics['efficiency']['cycle_time_efficiency']:.2f}",
        "\nQuality Metrics:",
        f"- Defect Rate: {performance_metrics['quality']['avg_defect_rate']:.2f}%",
        f"- Efficiency Score: {performance_metrics['quality']['efficiency_score']:.2f}",
        f"- Uniformity Score: {performance_metrics['quality']['uniformity_score']:.2f}",
        "\nSustainability Metrics:",
        f"- Material Efficiency: {performance_metrics['sustainability'].get('material_efficiency_material_efficiency', 0):.2f}%",
        f"- Recycling Rate: {performance_metrics['sustainability'].get('material_efficiency_recycling_rate', 0):.2f}%",
        f"- Sustainability Score: {performance_metrics['sustainability'].get('sustainability_score_overall', 0):.2f}",
        "\n2. Generated Files",
        "----------------",
        "Visualizations:",
    ]

    # Add visualization files
    viz_dir = run_dir / "visualizations"
    if viz_dir.exists():
        for file in viz_dir.glob("*.png"):
            summary.append(f"- {file.name}")

    summary.extend(["\nReports:", "- comprehensive_analysis.xlsx", "- lca_impact.xlsx"])

    with open(output_path, "w") as f:
        f.write("\n".join(summary))


if __name__ == "__main__":
    metrics, lca_results = run_comprehensive_demo()
