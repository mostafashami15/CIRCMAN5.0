"""Test helper utilities for manufacturing tests."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional


def generate_test_data(days: int = 30) -> Dict[str, pd.DataFrame]:
    """Generate synthetic test data for manufacturing tests."""
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(days * 24)]

    # Production data
    production_data = pd.DataFrame(
        {
            "timestamp": dates,
            "batch_id": [f"BATCH_{i:04d}" for i in range(len(dates))],
            "input_amount": np.random.uniform(90, 110, len(dates)),
            "output_amount": np.random.uniform(80, 100, len(dates)),
            "energy_used": np.random.uniform(140, 160, len(dates)),
            "cycle_time": np.random.uniform(45, 55, len(dates)),
        }
    )

    # Quality data
    quality_data = pd.DataFrame(
        {
            "test_timestamp": dates,
            "batch_id": production_data["batch_id"],
            "efficiency": np.random.uniform(20, 22, len(dates)),
            "defect_rate": np.random.uniform(1, 3, len(dates)),
            "thickness_uniformity": np.random.uniform(94, 96, len(dates)),
        }
    )

    # Energy data
    energy_data = pd.DataFrame(
        {
            "timestamp": dates,
            "energy_consumption": np.random.uniform(40, 60, len(dates)),
            "energy_source": np.random.choice(["grid", "solar", "wind"], len(dates)),
        }
    )

    # Material data
    material_data = pd.DataFrame(
        {
            "timestamp": dates,
            "material_type": np.random.choice(["Silicon", "Glass", "EVA"], len(dates)),
            "quantity_used": np.random.uniform(900, 1100, len(dates)),
            "waste_generated": np.random.uniform(45, 55, len(dates)),
            "recycled_amount": np.random.uniform(20, 30, len(dates)),
        }
    )

    return {
        "production": production_data,
        "quality": quality_data,
        "energy": energy_data,
        "material": material_data,
    }
