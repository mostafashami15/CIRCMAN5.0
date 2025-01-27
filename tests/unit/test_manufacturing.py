import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from circman5.manufacturing import AdvancedPVManufacturing


class TestPVManufacturing:
    """Test cases for PV Manufacturing System"""

    def setup_method(self):
        """Setup test cases"""
        self.pv_system = AdvancedPVManufacturing()
        self.test_batch_id = "TEST_BATCH_001"

    def test_batch_initialization(self):
        """Test batch creation and initialization"""
        # Test batch creation
        self.pv_system.start_batch(
            batch_id=self.test_batch_id, stage="silicon_purification", input_amount=100
        )

        # Assert batch exists
        assert self.test_batch_id in self.pv_system.batches["batch_id"].values

        # Check initial values
        batch = self.pv_system.batches[
            self.pv_system.batches["batch_id"] == self.test_batch_id
        ].iloc[0]

        assert batch["status"] == "in_progress"
        assert batch["input_amount"] == 100.0
        assert batch["output_amount"] == 0.0

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
            contamination_level=0.5,
        )

        # Get quality data
        quality = self.pv_system.quality_data[
            self.pv_system.quality_data["batch_id"] == self.test_batch_id
        ].iloc[0]

        # Assert quality measurements
        assert quality["efficiency"] == 21.5
        assert quality["defect_rate"] == 2.3
        assert quality["thickness_uniformity"] == 95.5
        assert quality["contamination_level"] == 0.5

    def test_circular_metrics(self):
        """Test circularity metrics recording and calculation"""
        # Setup
        self.pv_system.start_batch(self.test_batch_id, "silicon_purification", 100)

        # Record circular metrics
        self.pv_system.record_circular_metrics(
            batch_id=self.test_batch_id,
            recycled_content=30,
            recyclable_output=95,
            water_reused=80,
        )

        # Get circular metrics
        metrics = self.pv_system.circular_metrics[
            self.pv_system.circular_metrics["batch_id"] == self.test_batch_id
        ].iloc[0]

        # Assert metrics
        assert metrics["recycled_content"] == 30.0
        assert metrics["recyclable_output"] == 95.0
        assert metrics["water_reused"] == 80.0

    def test_batch_completion(self):
        """Test batch completion process"""
        # Setup
        self.pv_system.start_batch(self.test_batch_id, "silicon_purification", 100)

        # Complete batch
        self.pv_system.complete_batch(
            batch_id=self.test_batch_id, output_amount=90, energy_used=150
        )

        # Get batch data
        batch = self.pv_system.batches[
            self.pv_system.batches["batch_id"] == self.test_batch_id
        ].iloc[0]

        # Assert completion status and metrics
        assert batch["status"] == "completed"
        assert batch["output_amount"] == 90.0
        assert batch["energy_used"] == 150.0
        assert batch["yield_rate"] == 90.0

    def test_material_efficiency(self):
        """Test material efficiency calculations"""
        # Setup and complete a batch
        self.pv_system.start_batch(self.test_batch_id, "silicon_purification", 100)
        self.pv_system.complete_batch(self.test_batch_id, 90, 150)

        # Calculate efficiency
        efficiency = self.pv_system._calculate_material_efficiency(self.test_batch_id)

        # Assert efficiency calculation
        assert efficiency == 90.0

    def test_error_handling(self):
        """Test error handling in the system"""

        # Start the first batch successfully
        self.pv_system.start_batch("TEST_001", "silicon_purification", 100)

        # ✅ Expect a ValueError when trying to start the same batch again
        with pytest.raises(ValueError, match="Batch TEST_001 already exists"):
            self.pv_system.start_batch("TEST_001", "silicon_purification", 100)

        # ✅ Test invalid stage (expecting ValueError)
        with pytest.raises(ValueError, match="Invalid stage. Valid stages are"):
            self.pv_system.start_batch("TEST_002", "invalid_stage", 100)

        # ✅ Test invalid batch completion (expecting ValueError)
        with pytest.raises(ValueError, match="Batch INVALID_BATCH not found"):
            self.pv_system.complete_batch("INVALID_BATCH", 90, 150)
