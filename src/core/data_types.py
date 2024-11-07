# src/core/data_types.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class BatchData:
    """Manufacturing batch information."""
    batch_id: str
    start_time: datetime
    stage: str
    status: str
    input_material: str
    input_amount: float
    output_amount: float = 0.0
    yield_rate: float = 0.0
    energy_used: float = 0.0
    completion_time: Optional[datetime] = None

@dataclass
class QualityData:
    """Quality control measurements."""
    batch_id: str
    test_time: datetime
    efficiency: float
    defect_rate: float
    thickness_uniformity: float
    contamination_level: float

@dataclass
class CircularMetrics:
    """Circular economy metrics."""
    batch_id: str
    recycled_content: float
    recyclable_output: float
    water_reused: float
    material_efficiency: float
    waste_recyclability: float = 95.0