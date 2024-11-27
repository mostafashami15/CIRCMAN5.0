from solitek_manufacturing import SoliTekManufacturingAnalysis
from test_data_generator import TestDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def ensure_directories_exist():
    """Create necessary directories for test outputs if they don't exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = [
        os.path.join(base_dir, 'test_results'),
        os.path.join(base_dir, 'test_results', 'visualizations'),
        os.path.join(base_dir, 'test_results', 'reports'),
        os.path.join(base_dir, 'test_results', 'data')
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    
    return {
        'base': base_dir,
        'results': dirs[0],
        'viz': dirs[1],
        'reports': dirs[2],
        'data': dirs[3]
    }

def test_framework():
    """
    Comprehensive test of the SoliTek manufacturing analysis framework
    using generated test data.
    """
    # Setup directories and timestamped run folder
    directories = ensure_directories_exist()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(directories['results'], f'run_{timestamp}')
    os.makedirs(run_dir)
    
    print("Starting SoliTek Manufacturing Analysis Framework Test")
    print("-" * 50)
    
    # Create log file
    log_path = os.path.join(run_dir, 'test_log.txt')
    with open(log_path, 'w') as log_file:
        def log_print(message):
            print(message)
            log_file.write(message + '\n')
            
        log_print("Test started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Generate test data
        log_print("\nGenerating test data...")
        generator = TestDataGenerator(start_date='2024-01-01', days=30)
        
        production_data = generator.generate_production_data()
        energy_data = generator.generate_energy_data()
        quality_data = generator.generate_quality_data()
        material_data = generator.generate_material_flow_data()
        
        # Save test data
        data_dir = os.path.join(run_dir, 'input_data')
        os.makedirs(data_dir)
        production_data.to_csv(os.path.join(data_dir, 'test_production_data.csv'), index=False)
        energy_data.to_csv(os.path.join(data_dir, 'test_energy_data.csv'), index=False)
        quality_data.to_csv(os.path.join(data_dir, 'test_quality_data.csv'), index=False)
        material_data.to_csv(os.path.join(data_dir, 'test_material_data.csv'), index=False)
        
        # Initialize analysis framework
        log_print("\nInitializing analysis framework...")
        analyzer = SoliTekManufacturingAnalysis()
        
        # Test data loading
        log_print("\nTesting data loading capabilities...")
        analyzer.load_production_data(os.path.join(data_dir, 'test_production_data.csv'))
        analyzer.energy_data = energy_data
        analyzer.quality_data = quality_data
        analyzer.material_flow = material_data
        log_print("All test data loaded successfully")
        
        # Test efficiency analysis
        log_print("\nTesting efficiency analysis...")
        efficiency_metrics = analyzer.analyze_efficiency()
        log_print("Efficiency Metrics:")
        log_print(f"Average Yield: {efficiency_metrics['average_yield']:.2f}%")
        log_print("Cycle Time Statistics:")
        log_print(str(efficiency_metrics['cycle_time_stats']))
        
        # Test sustainability metrics
        log_print("\nTesting sustainability calculations...")
        sustainability_metrics = analyzer.calculate_sustainability_metrics()
        log_print("Sustainability Metrics:")
        for metric, value in sustainability_metrics.items():
            log_print(f"{metric}: {value}")
        
        # Test quality analysis
        log_print("\nTesting quality metrics analysis...")
        quality_metrics = analyzer.analyze_quality_metrics()
        log_print("Quality Metrics:")
        log_print(f"Average Efficiency: {quality_metrics['average_efficiency']:.2f}%")
        
        # Test visualization generation
        log_print("\nTesting visualization capabilities...")
        viz_dir = os.path.join(run_dir, 'visualizations')
        os.makedirs(viz_dir)
        
        for metric_type in ['production', 'energy', 'quality', 'sustainability']:
            viz_path = os.path.join(viz_dir, f'{metric_type}_analysis.png')
            analyzer.generate_visualization(metric_type, save_path=viz_path)
            log_print(f"Generated visualization for {metric_type}")
        
        # Test report generation
        log_print("\nTesting report generation...")
        report_path = os.path.join(run_dir, 'analysis_report.xlsx')
        analyzer.export_analysis_report(report_path)
        log_print("Analysis report generated successfully")
        
        log_print("\nTest completed at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log_print(f"\nTest results saved in: {run_dir}")

if __name__ == "__main__":
    test_framework()