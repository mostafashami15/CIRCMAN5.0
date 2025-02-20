# examples/demo_analysis.py

import sys
from pathlib import Path
import pandas as pd

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.results_manager import results_manager


def run_comprehensive_demo():
    """Run comprehensive manufacturing analysis demo."""
    print("Starting CIRCMAN5.0 Demo Analysis...")

    # Initialize the manufacturing analysis system
    analyzer = SoliTekManufacturingAnalysis()

    # Generate and load test data
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=30)
    dataset = generator.generate_complete_lca_dataset()

    analyzer.production_data = dataset["production_data"]
    analyzer.energy_data = dataset["energy_consumption"]
    analyzer.quality_data = generator.generate_quality_data()
    analyzer.material_flow = dataset["material_flow"]
    analyzer.lca_data = dataset

    # Run comprehensive analysis
    print("\nPerforming manufacturing analysis...")
    performance_metrics = analyzer.analyze_manufacturing_performance()

    # Generate all visualizations
    print("\nGenerating visualizations...")
    viz_dir = results_manager.get_path("visualizations")
    analyzer.generate_visualization(
        "production", str(viz_dir / "production_trends.png")
    )
    analyzer.generate_visualization("quality", str(viz_dir / "quality_metrics.png"))
    analyzer.generate_visualization("energy", str(viz_dir / "energy_patterns.png"))
    analyzer.generate_visualization(
        "sustainability", str(viz_dir / "sustainability_indicators.png")
    )

    # Perform lifecycle assessment
    print("\nPerforming lifecycle assessment...")
    lca_results = analyzer.perform_lifecycle_assessment(
        output_dir=results_manager.get_run_directory()
    )

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report_path = results_manager.get_path("reports") / "comprehensive_analysis.xlsx"
    analyzer.generate_comprehensive_report(str(report_path))

    # Generate results summary
    results_manager.generate_summary(performance_metrics)

    print(f"\nAnalysis complete!")
    print(f"Results saved in: {results_manager.get_run_directory()}")

    return performance_metrics, lca_results


if __name__ == "__main__":
    metrics, lca_results = run_comprehensive_demo()
