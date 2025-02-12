from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime
from pathlib import Path
from circman5.config import project_paths


def ensure_directories_exist():
    """Create necessary directories for test outputs if they don't exist."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    results_base = os.path.join(project_root, "tests", "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_base, "runs", f"run_{timestamp}")

    # Create base structure
    os.makedirs(os.path.join(results_base, "archive"), exist_ok=True)
    os.makedirs(os.path.join(results_base, "runs"), exist_ok=True)

    # Create run directory structure
    dirs = {
        "base": project_root,
        "results": run_dir,
        "data": os.path.join(run_dir, "input_data"),
        "viz": os.path.join(run_dir, "visualizations"),
        "reports": os.path.join(run_dir, "reports"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    latest_link = os.path.join(results_base, "latest")
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            shutil.rmtree(latest_link)
    os.symlink(run_dir, latest_link)

    return dirs


def test_framework():
    """Comprehensive test of the SoliTek manufacturing analysis framework."""
    directories = ensure_directories_exist()

    print("Starting SoliTek Manufacturing Analysis Framework Test")
    print("-" * 50)

    # Create log file
    log_path = os.path.join(directories["results"], "test_log.txt")
    with open(log_path, "w") as log_file:

        def log_print(message):
            print(message)
            log_file.write(message + "\n")

        log_print("Test started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        try:
            # Generate test data
            log_print("\nGenerating test data...")
            generator = ManufacturingDataGenerator(start_date="2024-01-01", days=30)

            # Generate data
            production_data = generator.generate_production_data()
            energy_data = generator.generate_energy_data()
            quality_data = generator.generate_quality_data()
            material_data = generator.generate_material_flow_data()

            # Save test data files
            data_dir = directories["data"]
            production_path = os.path.join(data_dir, "test_production_data.csv")
            production_data.to_csv(production_path, index=False)

            energy_path = os.path.join(data_dir, "test_energy_data.csv")
            energy_data.to_csv(energy_path, index=False)

            quality_path = os.path.join(data_dir, "test_quality_data.csv")
            quality_data.to_csv(quality_path, index=False)

            material_path = os.path.join(data_dir, "test_material_data.csv")
            material_data.to_csv(material_path, index=False)

            # Initialize analysis framework
            log_print("\nInitializing analysis framework...")
            analyzer = SoliTekManufacturingAnalysis()

            # Test data loading
            log_print("\nTesting data loading capabilities...")
            analyzer.load_data(production_path=production_path)
            analyzer.energy_data = energy_data
            analyzer.quality_data = quality_data
            analyzer.material_flow = material_data
            log_print("All test data loaded successfully")

            # Test efficiency analysis
            log_print("\nTesting efficiency analysis...")
            efficiency_metrics = analyzer.analyze_manufacturing_performance()[
                "efficiency"
            ]
            log_print("Efficiency Metrics:")
            if "yield_rate" in efficiency_metrics:
                log_print(f"Average Yield: {efficiency_metrics['yield_rate']:.2f}%")
            if "cycle_time_stats" in efficiency_metrics:
                log_print("Cycle Time Statistics:")
                log_print(str(efficiency_metrics["cycle_time_stats"]))

            # Test sustainability metrics
            log_print("\nTesting sustainability calculations...")
            sustainability_metrics = analyzer.analyze_manufacturing_performance()[
                "sustainability"
            ]
            log_print("Sustainability Metrics:")
            for metric, value in sustainability_metrics.items():
                log_print(f"{metric}: {value}")

            # Test quality analysis
            log_print("\nTesting quality metrics analysis...")
            quality_metrics = analyzer.analyze_manufacturing_performance()["quality"]
            log_print("Quality Metrics:")
            if "average_efficiency" in quality_metrics:
                log_print(
                    f"Average Efficiency: {quality_metrics['average_efficiency']:.2f}%"
                )

            # Test visualization generation
            log_print("\nTesting visualization capabilities...")
            for metric_type in ["production", "energy", "quality", "sustainability"]:
                viz_path = os.path.join(
                    directories["viz"], f"{metric_type}_analysis.png"
                )
                # This depends on the metric_type, but you can use the internal methods:
                if metric_type == "production":
                    analyzer.visualizer.visualize_production_trends(
                        analyzer.production_data, save_path=viz_path
                    )
                elif metric_type == "quality":
                    analyzer.visualizer.visualize_quality_metrics(
                        analyzer.quality_data,
                        analyzer=analyzer.quality_analyzer,
                        save_path=viz_path,
                    )
                elif metric_type == "sustainability":
                    analyzer.visualizer.visualize_sustainability_indicators(
                        analyzer.material_flow, analyzer.energy_data, save_path=viz_path
                    )
                elif metric_type == "energy":
                    analyzer.visualizer.visualize_energy_patterns(
                        analyzer.energy_data, save_path=viz_path
                    )
                log_print(f"Generated visualization for {metric_type}")

            # Test report generation
            log_print("\nTesting report generation...")
            report_path = os.path.join(directories["reports"], "analysis_report.xlsx")
            analyzer.generate_reports(output_dir=Path(report_path).parent)
            log_print("Analysis report generated successfully")

        except Exception as e:
            log_print(f"\nERROR: {str(e)}")
            raise

        finally:
            log_print(
                "\nTest completed at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            log_print(f"\nTest results saved in: {directories['results']}")


if __name__ == "__main__":
    test_framework()
