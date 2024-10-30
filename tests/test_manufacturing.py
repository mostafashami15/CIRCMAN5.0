import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.core.manufacturing import AdvancedPVManufacturing

class TestPVManufacturing(unittest.TestCase):
    """Test cases for PV Manufacturing System"""

    def setUp(self):
        """Setup test cases"""
        self.pv_system = AdvancedPVManufacturing()
        self.test_batch_id = "TEST_BATCH_001"
        
    def test_batch_initialization(self):
        """Test batch creation and initialization"""
        # Test batch creation
        self.pv_system.start_batch(
            batch_id=self.test_batch_id,
            stage="silicon_purification",
            input_amount=100
        )
        
        # Assert batch exists
        self.assertIn(self.test_batch_id, 
                     self.pv_system.batches['batch_id'].values)
        
        # Check initial values
        batch = self.pv_system.batches[
            self.pv_system.batches['batch_id'] == self.test_batch_id
        ].iloc[0]
        
        self.assertEqual(batch['status'], 'in_progress')
        self.assertEqual(batch['input_amount'], 100.0)
        self.assertEqual(batch['output_amount'], 0.0)

    def test_quality_control(self):
        """Test quality control measurements"""
        # Setup
        self.pv_system.start_batch(self.test_batch_id, "silicon_purification", 100)
        
        # Record quality check
        self.pv_system.record_quality_check(
            batch_id=self.test_batch_id,
            efficiency=21.5,
            defect_rate=2.3,
            thickness_uniformity=95.5,
            contamination_level=0.5
        )
        
        # Get quality data
        quality = self.pv_system.quality_data[
            self.pv_system.quality_data['batch_id'] == self.test_batch_id
        ].iloc[0]
        
        # Assert quality measurements
        self.assertEqual(quality['efficiency'], 21.5)
        self.assertEqual(quality['defect_rate'], 2.3)
        self.assertEqual(quality['thickness_uniformity'], 95.5)
        self.assertEqual(quality['contamination_level'], 0.5)

    def test_circular_metrics(self):
        """Test circularity metrics recording and calculation"""
        # Setup
        self.pv_system.start_batch(self.test_batch_id, "silicon_purification", 100)
        
        # Record circular metrics
        self.pv_system.record_circular_metrics(
            batch_id=self.test_batch_id,
            recycled_content=30,
            recyclable_output=95,
            water_reused=80
        )
        
        # Get circular metrics
        metrics = self.pv_system.circular_metrics[
            self.pv_system.circular_metrics['batch_id'] == self.test_batch_id
        ].iloc[0]
        
        # Assert metrics
        self.assertEqual(metrics['recycled_content'], 30.0)
        self.assertEqual(metrics['recyclable_output'], 95.0)
        self.assertEqual(metrics['water_reused'], 80.0)

    def test_batch_completion(self):
        """Test batch completion process"""
        # Setup
        self.pv_system.start_batch(self.test_batch_id, "silicon_purification", 100)
        
        # Complete batch
        self.pv_system.complete_batch(
            batch_id=self.test_batch_id,
            output_amount=90,
            energy_used=150
        )
        
        # Get batch data
        batch = self.pv_system.batches[
            self.pv_system.batches['batch_id'] == self.test_batch_id
        ].iloc[0]
        
        # Assert completion status and metrics
        self.assertEqual(batch['status'], 'completed')
        self.assertEqual(batch['output_amount'], 90.0)
        self.assertEqual(batch['energy_used'], 150.0)
        self.assertEqual(batch['yield_rate'], 90.0)

    def test_material_efficiency(self):
        """Test material efficiency calculations"""
        # Setup and complete a batch
        self.pv_system.start_batch(self.test_batch_id, "silicon_purification", 100)
        self.pv_system.complete_batch(self.test_batch_id, 90, 150)
        
        # Calculate efficiency
        efficiency = self.pv_system._calculate_material_efficiency(self.test_batch_id)
        
        # Assert efficiency calculation
        self.assertEqual(efficiency, 90.0)

    def test_error_handling(self):
        """Test error handling in the system"""
        # Test duplicate batch
        self.pv_system.start_batch("TEST_001", "silicon_purification", 100)
        self.pv_system.start_batch("TEST_001", "silicon_purification", 100)
        
        # Test invalid stage
        self.pv_system.start_batch("TEST_002", "invalid_stage", 100)
        
        # Test invalid batch completion
        self.pv_system.complete_batch("INVALID_BATCH", 90, 150)

if __name__ == '__main__':
    unittest.main()