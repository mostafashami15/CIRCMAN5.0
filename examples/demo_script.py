#!/usr/bin/env python
"""
CIRCMAN5.0 System Demonstration
This script showcases the AI-driven manufacturing optimization system's capabilities.
"""

import sys
from pathlib import Path
from datetime import datetime

from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.logging_config import setup_logger
from circman5.config.project_paths import project_paths


def prepare_data(data_generator):
    """Prepare synthetic manufacturing data with consistent column names."""
    production_data = data_generator.generate_production_data()
    quality_data = data_generator.generate_quality_data()
    energy_data = data_generator.generate_energy_data()
    material_flow = data_generator.generate_material_flow_data()
    return production_data, quality_data, energy_data, material_flow


def main():
    """Main function to run the CIRCMAN5.0 demo."""
    # Get run directory and subdirectories
    run_dir = project_paths.get_run_directory()
    input_data_dir = run_dir / "input_data"
    reports_dir = run_dir / "reports"
    visualizations_dir = run_dir / "visualizations"

    # Setup logger
    logger = setup_logger("demo_script")
    logger.info("Initializing CIRCMAN5.0 Manufacturing Analysis System...")

    try:
        # Initialize system
        analyzer = SoliTekManufacturingAnalysis()

        # Generate synthetic manufacturing data
        logger.info("Generating synthetic manufacturing data...")
        data_generator = ManufacturingDataGenerator(start_date="2024-01-01", days=30)
        production_data, quality_data, energy_data, material_flow = prepare_data(
            data_generator
        )

        # Save input data for current run
        production_data.to_csv(input_data_dir / "production_data.csv", index=False)
        quality_data.to_csv(input_data_dir / "quality_data.csv", index=False)
        energy_data.to_csv(input_data_dir / "energy_data.csv", index=False)
        material_flow.to_csv(input_data_dir / "material_flow.csv", index=False)

        # Update analyzer with data
        analyzer.production_data = production_data
        analyzer.quality_data = quality_data
        analyzer.energy_data = energy_data
        analyzer.material_flow = material_flow

        logger.info(f"Generated {len(analyzer.production_data)} production records.")

        # Train AI optimization model
        logger.info("Training AI optimization model...")
        metrics = analyzer.train_optimization_model()
        logger.info(f"Model training metrics: {metrics}")

        # Optimize process parameters
        logger.info("Optimizing manufacturing parameters...")
        current_params = {
            "input_amount": 100.0,
            "energy_used": 150.0,
            "cycle_time": 50.0,
            "efficiency": 21.0,
            "defect_rate": 2.0,
            "thickness_uniformity": 95.0,
        }
        optimized_params = analyzer.optimize_process_parameters(current_params)

        logger.info("Optimization Results:")
        for param, value in optimized_params.items():
            logger.info(f"  {param}: {value:.2f}")

        # Generate visualizations
        logger.info("Generating analysis visualizations...")
        for metric in ["production", "energy", "quality", "sustainability"]:
            viz_path = visualizations_dir / f"{metric}_analysis.png"
            analyzer.generate_visualization(metric, str(viz_path))
            logger.info(f"  Saved {metric} visualization")

        # Generate comprehensive analysis report
        report_path = reports_dir / "analysis_report.xlsx"
        analyzer.export_analysis_report(str(report_path))
        logger.info("Analysis report saved")

    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
