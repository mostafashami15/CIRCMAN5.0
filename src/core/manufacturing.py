# File: pv_tracking.py

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List

class AdvancedPVManufacturing:
    def __init__(self):
        # Define datatypes for each DataFrame
        self.batch_dtypes = {
            'batch_id': str,
            'start_time': 'datetime64[ns]',
            'stage': str,
            'status': str,
            'input_material': str,
            'input_amount': float,
            'output_amount': float,
            'yield_rate': float,
            'energy_used': float,
            'completion_time': 'datetime64[ns]'
        }
        
        self.quality_dtypes = {
            'batch_id': str,
            'test_time': 'datetime64[ns]',
            'efficiency': float,
            'defect_rate': float,
            'thickness_uniformity': float,
            'contamination_level': float
        }
        
        self.circular_dtypes = {
            'batch_id': str,
            'recycled_content': float,
            'recyclable_output': float,
            'water_reused': float,
            'material_efficiency': float,
            'waste_recyclability': float
        }

        # Initialize DataFrames
        self.batches = pd.DataFrame(columns=self.batch_dtypes.keys()).astype(self.batch_dtypes)
        self.quality_data = pd.DataFrame(columns=self.quality_dtypes.keys()).astype(self.quality_dtypes)
        self.circular_metrics = pd.DataFrame(columns=self.circular_dtypes.keys()).astype(self.circular_dtypes)

        # Define manufacturing stages
        self.stages = {
            'silicon_purification': {
                'input': 'raw_silicon',
                'output': 'purified_silicon',
                'expected_yield': 0.90
            },
            'wafer_production': {
                'input': 'purified_silicon',
                'output': 'silicon_wafer',
                'expected_yield': 0.95
            },
            'cell_production': {
                'input': 'silicon_wafer',
                'output': 'solar_cell',
                'expected_yield': 0.98
            }
        }

    def start_batch(self, batch_id: str, stage: str, input_amount: float) -> None:
        """
        Start a new production batch.

        Args:
            batch_id: Unique identifier for the batch
            stage: Manufacturing stage (must be one of the defined stages)
            input_amount: Amount of input material in kg

        Raises:
            ValueError: If batch_id already exists or stage is invalid
            ValueError: If input_amount is negative or zero
        """

        """Start a new production batch"""
        
        # Validate batch_id
        if batch_id in self.batches['batch_id'].values:
            raise ValueError(f"Batch {batch_id} already exists")

        # Validate stage
        if stage not in self.stages:
            valid_stages = list(self.stages.keys())
            raise ValueError(f"Invalid stage. Valid stages are: {valid_stages}")

        # Validate input amount
        if input_amount <= 0:
            raise ValueError("Input amount must be positive")

        # Create new batch (keep your existing dictionary structure for now)
        new_batch = {
            'batch_id': batch_id,
            'start_time': datetime.now(),
            'stage': stage,
            'status': 'in_progress',
            'input_material': self.stages[stage]['input'],
            'input_amount': float(input_amount),
            'output_amount': 0.0,
            'yield_rate': 0.0,
            'energy_used': 0.0,
            'completion_time': pd.NaT
        }

        # Add to batches DataFrame
        self.batches = pd.concat([self.batches, pd.DataFrame([new_batch])], 
                                ignore_index=True)
        
        print(f"\nStarted Batch: {batch_id}")
        print(f"Stage: {stage}")
        print(f"Input Material: {self.stages[stage]['input']}")
        print(f"Amount: {input_amount} kg")

    def complete_batch(self, batch_id, output_amount, energy_used):
        """Complete a production batch"""
        if batch_id not in self.batches['batch_id'].values:
            print(f"Error: Batch {batch_id} not found!")
            return

        batch_idx = self.batches[self.batches['batch_id'] == batch_id].index[0]
        
        self.batches.loc[batch_idx, 'output_amount'] = float(output_amount)
        self.batches.loc[batch_idx, 'energy_used'] = float(energy_used)
        self.batches.loc[batch_idx, 'status'] = 'completed'
        self.batches.loc[batch_idx, 'completion_time'] = datetime.now()

        input_amount = self.batches.loc[batch_idx, 'input_amount']
        yield_rate = (output_amount / input_amount * 100) if input_amount > 0 else 0
        self.batches.loc[batch_idx, 'yield_rate'] = round(yield_rate, 2)

        print(f"\nCompleted Batch: {batch_id}")
        print(f"Output Amount: {output_amount} kg")
        print(f"Yield Rate: {yield_rate:.1f}%")
        print(f"Energy Used: {energy_used} kWh")

    def record_quality_check(self, batch_id, efficiency, defect_rate,
                           thickness_uniformity, contamination_level):
        """Record quality control measurements"""
        new_quality_record = {
            'batch_id': batch_id,
            'test_time': datetime.now(),
            'efficiency': float(efficiency),
            'defect_rate': float(defect_rate),
            'thickness_uniformity': float(thickness_uniformity),
            'contamination_level': float(contamination_level)
        }
        
        self.quality_data = pd.concat([self.quality_data, pd.DataFrame([new_quality_record])], 
                                    ignore_index=True)
        
        print(f"\nQuality Check Recorded for Batch {batch_id}")
        print(f"Cell Efficiency: {efficiency}%")
        print(f"Defect Rate: {defect_rate}%")

    def record_circular_metrics(self, batch_id, recycled_content,
                              recyclable_output, water_reused):
        """Record circular economy metrics"""
        new_circular_record = {
            'batch_id': batch_id,
            'recycled_content': float(recycled_content),
            'recyclable_output': float(recyclable_output),
            'water_reused': float(water_reused),
            'material_efficiency': self._calculate_material_efficiency(batch_id),
            'waste_recyclability': 95.0
        }
        
        self.circular_metrics = pd.concat([self.circular_metrics, 
                                         pd.DataFrame([new_circular_record])], 
                                         ignore_index=True)
        
        print(f"\nCircular Metrics Recorded for Batch {batch_id}")
        print(f"Recycled Content: {recycled_content}%")
        print(f"Water Reused: {water_reused}%")

    def _calculate_material_efficiency(self, batch_id):
        """Calculate material efficiency for a batch"""
        try:
            batch_data = self.batches[self.batches['batch_id'] == batch_id].iloc[0]
            input_amount = batch_data['input_amount']
            output_amount = batch_data['output_amount']
            
            if input_amount > 0:
                efficiency = (output_amount / input_amount) * 100
                return float(round(efficiency, 2))
            return 0.0
        except Exception as e:
            print(f"Error calculating material efficiency: {e}")
            return 0.0

    def get_batch_summary(self, batch_id):
        """Get detailed summary for a specific batch"""
        if batch_id not in self.batches['batch_id'].values:
            print(f"Batch {batch_id} not found!")
            return

        # Get data from all tables
        batch = self.batches[self.batches['batch_id'] == batch_id].iloc[0]
        quality = self.quality_data[self.quality_data['batch_id'] == batch_id].iloc[0]
        circular = self.circular_metrics[self.circular_metrics['batch_id'] == batch_id].iloc[0]

        print(f"\n=== Detailed Batch Summary: {batch_id} ===")
        
        # Production Details
        print("\nProduction Details:")
        print(f"Stage: {batch['stage']}")
        print(f"Status: {batch['status']}")
        print(f"Start Time: {batch['start_time']}")
        if batch['completion_time'] is not pd.NaT:
            print(f"Completion Time: {batch['completion_time']}")

        # Material Flow
        print("\nMaterial Flow:")
        print(f"Input Material: {batch['input_material']}")
        print(f"Input Amount: {batch['input_amount']} kg")
        print(f"Output Amount: {batch['output_amount']} kg")
        print(f"Material Efficiency: {batch['yield_rate']}%")
        print(f"Waste Generated: {batch['input_amount'] - batch['output_amount']:.2f} kg")

        # Quality Metrics
        print("\nQuality Metrics:")
        print(f"Cell Efficiency: {quality['efficiency']}%")
        print(f"Defect Rate: {quality['defect_rate']}%")
        print(f"Thickness Uniformity: {quality['thickness_uniformity']}%")
        print(f"Contamination Level: {quality['contamination_level']}%")

        # Circular Economy Metrics
        print("\nCircular Economy Metrics:")
        print(f"Recycled Content: {circular['recycled_content']}%")
        print(f"Recyclable Output: {circular['recyclable_output']}%")
        print(f"Water Reused: {circular['water_reused']}%")

        # Resource Usage
        print("\nResource Usage:")
        if batch['energy_used'] > 0:
            print(f"Energy Consumed: {batch['energy_used']} kWh")
            print(f"Energy Efficiency: {batch['output_amount']/batch['energy_used']:.2f} kg/kWh")
            # Add these methods to your AdvancedPVManufacturing class:


    def visualize_batch_performance(self, batch_id, save_path=None):
        """Create visualizations for batch performance"""
        plt.figure(figsize=(15, 10))
        
        # 1. Material Flow
        plt.subplot(221)
        batch = self.batches[self.batches['batch_id'] == batch_id].iloc[0]
        waste = batch['input_amount'] - batch['output_amount']
        values = [batch['input_amount'], batch['output_amount'], waste]
        bars = plt.bar(['Input', 'Output', 'Waste'], values, 
                       color=['blue', 'green', 'red'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}kg\n({height/values[0]*100:.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('Material Flow Analysis')
        plt.ylabel('Amount (kg)')

        if save_path:
            plt.savefig(f"{save_path}/batch_{batch_id}_performance.png")
            plt.close()
        else:
            plt.show()
            plt.close()

    def visualize_trends(self, save_path=None):
        """Visualize trends across multiple batches"""
        if len(self.batches) < 2:
            print("Need at least 2 batches for trend analysis")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 1. Yield Rate Trend
        plt.subplot(221)
        plt.plot(range(len(self.batches)), self.batches['yield_rate'], 
                 marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xticks(range(len(self.batches)), self.batches['batch_id'])
        plt.title('Yield Rate Trend')
        plt.xlabel('Batch ID')
        plt.ylabel('Yield Rate (%)')
        
        # 2. Energy Efficiency Trend
        plt.subplot(222)
        energy_efficiency = self.batches['output_amount'] / self.batches['energy_used']
        plt.plot(range(len(self.batches)), energy_efficiency, 
                 marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xticks(range(len(self.batches)), self.batches['batch_id'])
        plt.title('Energy Efficiency Trend')
        plt.xlabel('Batch ID')
        plt.ylabel('kg/kWh')

        if save_path:
            plt.savefig(f"{save_path}/production_trends.png")
            plt.close()
        else:
            plt.show()
            plt.close()

    def save_data_to_excel(self, filename="pv_manufacturing_data.xlsx"):
        """Save all manufacturing data to Excel file"""
        try:
            # Create Excel writer object
            with pd.ExcelWriter(filename) as writer:
                # Save each dataset to different sheets
                self.batches.to_excel(writer, sheet_name='Production_Data', index=False)
                self.quality_data.to_excel(writer, sheet_name='Quality_Data', index=False)
                self.circular_metrics.to_excel(writer, sheet_name='Circular_Metrics', index=False)
                
                # Create and save summary
                summary = self.create_summary_report()
                summary.to_excel(writer, sheet_name='Summary_Report')
            
            print(f"\nData successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def save_analysis_report(self, filename="analysis_report.txt"):
        """Save detailed analysis report"""
        try:
            with open(filename, 'w') as f:
                f.write("=== PV Manufacturing Analysis Report ===\n\n")
                
                # Production Summary
                f.write("Production Summary:\n")
                f.write("-" * 50 + "\n")
                summary_df = self.create_summary_report()
                for _, row in summary_df.iterrows():
                    f.write(f"{row['Metric']}: {row['Value']} {row['Unit']}\n")
                
                # Quality Metrics
                f.write("\nQuality Metrics Summary:\n")
                f.write("-" * 50 + "\n")
                quality_stats = self.quality_data.describe()
                f.write(str(quality_stats))
                
                # Process Efficiency
                f.write("\n\nProcess Efficiency:\n")
                f.write("-" * 50 + "\n")
                efficiency_stats = self.batches['yield_rate'].describe()
                f.write(str(efficiency_stats))
                
                # Circular Economy Metrics
                f.write("\n\nCircular Economy Metrics:\n")
                f.write("-" * 50 + "\n")
                circular_stats = self.circular_metrics.describe()
                f.write(str(circular_stats))
                
                # Recommendations
                f.write("\n\nRecommendations:\n")
                f.write("-" * 50 + "\n")
                recommendations = self.generate_optimization_recommendations()
                for rec in recommendations:
                    f.write(f"\nArea: {rec['area']}\n")
                    f.write(f"Current: {rec['current']}\n")
                    f.write(f"Target: {rec['target']}\n")
                    f.write(f"Recommendation: {rec['recommendation']}\n")
                
            print(f"\nAnalysis report saved to {filename}")
        except Exception as e:
            print(f"Error saving analysis report: {e}")

    def create_summary_report(self):
        """Create a comprehensive summary report"""
        summary_data = {
            'Metric': [],
            'Value': [],
            'Unit': []
        }
        
        total_input = self.batches['input_amount'].sum()
        total_output = self.batches['output_amount'].sum()
        total_waste = total_input - total_output
        
        summary_data['Metric'].extend([
            'Total Production',
            'Total Waste',
            'Average Yield Rate',
            'Total Energy Used'
        ])
        
        summary_data['Value'].extend([
            total_output,
            total_waste,
            self.batches['yield_rate'].mean(),
            self.batches['energy_used'].sum()
        ])
        
        summary_data['Unit'].extend([
            'kg',
            'kg',
            '%',
            'kWh'
        ])
        
        return pd.DataFrame(summary_data)

    def perform_advanced_analysis(self):
        """Perform advanced statistical analysis"""
        analysis_results = {
            'Process Efficiency': {},
            'Quality Trends': {},
            'Circular Economy Impact': {},
            'Resource Optimization': {}
        }
        
        # Process Efficiency Analysis
        analysis_results['Process Efficiency'] = {
            'yield_trend': self.batches['yield_rate'].describe(),
            'energy_efficiency': (self.batches['output_amount'] / 
                                self.batches['energy_used']).describe(),
            'waste_rate': ((self.batches['input_amount'] - 
                           self.batches['output_amount']) / 
                          self.batches['input_amount']).describe()
        }
        
        # Quality Analysis
        analysis_results['Quality Trends'] = {
            'efficiency_trend': self.quality_data['efficiency'].describe(),
            'defect_rate_trend': self.quality_data['defect_rate'].describe(),
            'uniformity_trend': self.quality_data['thickness_uniformity'].describe()
        }
        
        return analysis_results

    def generate_optimization_recommendations(self):
        """Generate optimization recommendations based on data analysis"""
        recommendations = []
        
        avg_yield = self.batches['yield_rate'].mean()
        if avg_yield < 92:
            recommendations.append({
                'area': 'Yield Optimization',
                'current': f'{avg_yield:.1f}%',
                'target': '92%',
                'recommendation': 'Consider process parameter optimization'
            })
        
        energy_efficiency = (self.batches['output_amount'] / 
                            self.batches['energy_used']).mean()
        if energy_efficiency < 0.7:
            recommendations.append({
                'area': 'Energy Efficiency',
                'current': f'{energy_efficiency:.2f} kg/kWh',
                'target': '0.7 kg/kWh',
                'recommendation': 'Review energy consumption patterns'
            })
        
        return recommendations

def main():
    # Create output directory in your PV_Project folder
    import os
    base_dir = "/Users/mostafashami/Desktop/PV_Project"
    output_dir = os.path.join(base_dir, "pv_manufacturing_results")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create manufacturing system
    factory = AdvancedPVManufacturing()

    # Process batches
    for i in range(1, 4):
        batch_id = f"BATCH_{i:03d}"
        
        # Start batch
        factory.start_batch(batch_id, "silicon_purification", 100)
        
        # Record quality check
        factory.record_quality_check(
            batch_id=batch_id,
            efficiency=21.5 + i*0.5,
            defect_rate=2.3 - i*0.1,
            thickness_uniformity=95.5 + i*0.2,
            contamination_level=0.5 - i*0.05
        )
        
        # Record circular metrics
        factory.record_circular_metrics(
            batch_id=batch_id,
            recycled_content=30 + i*2,
            recyclable_output=95,
            water_reused=80 + i
        )
        
        # Complete batch
        factory.complete_batch(batch_id, 90 + i, 150 - i*5)
        
        # Show batch summary
        factory.get_batch_summary(batch_id)
    
    # Save visualizations
    factory.visualize_batch_performance("BATCH_001", save_path=output_dir)
    factory.visualize_trends(save_path=output_dir)
    
    # Save data to Excel
    excel_file = os.path.join(output_dir, "pv_production_data.xlsx")
    factory.save_data_to_excel(excel_file)
    
    # Perform analysis
    analysis = factory.perform_advanced_analysis()
    
    # Save analysis results
    analysis_file = os.path.join(output_dir, "analysis_report.txt")
    with open(analysis_file, "w") as f:
        f.write("=== Advanced Analysis Results ===\n")
        for category, results in analysis.items():
            f.write(f"\n{category}:\n")
            for metric, stats in results.items():
                f.write(f"\n{metric}:\n")
                f.write(str(stats))
                f.write("\n")
    
    # Get and save recommendations
    recommendations = factory.generate_optimization_recommendations()
    recommendations_file = os.path.join(output_dir, "recommendations.txt")
    with open(recommendations_file, "w") as f:
        f.write("=== Optimization Recommendations ===\n")
        for rec in recommendations:
            f.write(f"\nArea: {rec['area']}\n")
            f.write(f"Current: {rec['current']}\n")
            f.write(f"Target: {rec['target']}\n")
            f.write(f"Recommendation: {rec['recommendation']}\n")

    print(f"\nAll results have been saved to the '{output_dir}' directory")
    
    # Print file locations for easy access
    print("\nFile Locations:")
    print(f"Excel Data: {excel_file}")
    print(f"Analysis Report: {analysis_file}")
    print(f"Recommendations: {recommendations_file}")
    print(f"Visualizations: {output_dir}/batch_BATCH_001_performance.png")
    print(f"                {output_dir}/production_trends.png")

if __name__ == "__main__":
    main()