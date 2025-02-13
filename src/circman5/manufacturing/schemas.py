# src/circman5/manufacturing/schemas.py
"""Data schemas for manufacturing analysis."""

from typing import Dict, Union, Type, Literal
from pandas.core.dtypes.base import ExtensionDtype

# Define valid pandas dtype strings
PandasDtype = Union[
    ExtensionDtype,
    Type[str],
    Type[float],
    Literal["datetime64[ns]", "string", "float64"],
]

SchemaType = Dict[str, PandasDtype]

# Production Data Schema
PRODUCTION_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "batch_id": str,
    "product_type": str,
    "production_line": str,
    "output_amount": float,
    "cycle_time": float,
    "yield_rate": float,
    "input_amount": float,
    "output_amount": float,
    "energy_used": float,
}

# Energy Data Schema
ENERGY_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "production_line": str,
    "energy_consumption": float,
    "energy_source": str,
    "efficiency_rate": float,
}

# Quality Data Schema
QUALITY_SCHEMA: SchemaType = {
    "batch_id": str,
    "test_timestamp": "datetime64[ns]",
    "efficiency": float,
    "defect_rate": float,
    "thickness_uniformity": float,
    "visual_inspection": str,
}

# Material Data Schema
MATERIAL_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "material_type": str,
    "quantity_used": float,
    "waste_generated": float,
    "recycled_amount": float,
    "batch_id": str,
}

# LCA-specific Schemas
LCA_MATERIAL_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "material_type": str,
    "quantity": float,
    "impact_factor": float,
    "batch_id": str,
}

LCA_ENERGY_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "source": str,
    "consumption": float,
    "carbon_intensity": float,
    "batch_id": str,
}

LCA_PROCESS_SCHEMA: SchemaType = {
    "timestamp": "datetime64[ns]",
    "process_step": str,
    "duration": float,
    "impact_factor": float,
    "batch_id": str,
}
