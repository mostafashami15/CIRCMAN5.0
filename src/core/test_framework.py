from solitek_manufacturing import SoliTekManufacturingAnalysis
from test_data_generator import TestDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

def test_framework():
    """
    Comprehensive test of the SoliTek manufacturing analysis framework
    using generated test data.
    """
    print("Starting SoliTek Manufacturing Analysis Framework Test")
    print("-" * 50)

    # Generate test data
    print("Generating test data...")
    generator = TestDataGenerator(start_date='2024-01-01', days=30)
    
    production_data = generator.generate_production_data()
    energy_data = generator.generate_energy_data()
    quality_data = generator.generate_quality_data()
    material_data = generator.generate_material_flow_data()

    # Initialize analysis framework
    print("\nInitializing analysis framework...")
    analyzer = SoliTekManufacturingAnalysis()

    # Test data loading
    print("\nTesting data loading capabilities...")
    
    # Save and load production data
    production_data.to_csv('test_production_data.csv', index=False)
    analyzer.load_production_data('test_production_data.csv')
    print("Production data loaded successfully")

    # Load other data directly
    analyzer.energy_data = energy_data
    analyzer.quality_data = quality_data
    analyzer.material_flow = material_data
    print("All test data loaded successfully")

    # Test efficiency analysis
    print("\nTesting efficiency analysis...")
    efficiency_metrics = analyzer.analyze_efficiency()
    print("Efficiency Metrics:")
    print(f"Average Yield: {efficiency_metrics['average_yield']:.2f}%")
    print("Cycle Time Statistics:")
    print(efficiency_metrics['cycle_time_stats'])

    # Test sustainability metrics
    print("\nTesting sustainability calculations...")
    sustainability_metrics = analyzer.calculate_sustainability_metrics()
    print("Sustainability Metrics:")
    for metric, value in sustainability_metrics.items():
        print(f"{metric}: {value}")

    # Test quality analysis
    print("\nTesting quality metrics analysis...")
    quality_metrics = analyzer.analyze_quality_metrics()
    print("Quality Metrics:")
    print(f"Average Efficiency: {quality_metrics['average_efficiency']:.2f}%")

    # Test visualization generation
    print("\nTesting visualization capabilities...")
    for metric_type in ['production', 'energy', 'quality', 'sustainability']:
        analyzer.generate_visualization(
            metric_type, 
            save_path=f'test_{metric_type}_viz.png'
        )
        print(f"Generated visualization for {metric_type}")

    # Test report generation
    print("\nTesting report generation...")
    analyzer.export_analysis_report('test_analysis_report.xlsx')
    print("Analysis report generated successfully")

    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_framework()