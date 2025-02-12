import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def test_data():
    """Generate test data for model validation."""
    np.random.seed(42)
    production_data = pd.DataFrame(
        {
            "input_amount": np.random.uniform(90, 110, 100),
            "energy_used": np.random.uniform(140, 160, 100),
            "cycle_time": np.random.uniform(45, 55, 100),
            "output_amount": np.random.uniform(85, 105, 100),
            "batch_id": [f"BATCH_{i:03d}" for i in range(100)],
        }
    )

    quality_data = pd.DataFrame(
        {
            "efficiency": np.random.uniform(20, 22, 100),
            "defect_rate": np.random.uniform(1, 3, 100),
            "thickness_uniformity": np.random.uniform(94, 96, 100),
            "batch_id": [f"BATCH_{i:03d}" for i in range(100)],
        }
    )

    return {"production": production_data, "quality": quality_data}


@pytest.fixture
def model():
    """Create a test model instance."""
    from circman5.manufacturing.optimization.model import ManufacturingModel

    return ManufacturingModel()


@pytest.fixture
def optimizer():
    """Create a test optimizer instance."""
    from circman5.manufacturing.optimization.optimizer import ProcessOptimizer

    return ProcessOptimizer()
