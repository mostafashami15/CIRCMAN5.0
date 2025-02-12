# src/circman5/manufacturing/lifecycle/__init__.py
from .lca_analyzer import LCAAnalyzer, LifeCycleImpact
from .impact_factors import (
    MATERIAL_IMPACT_FACTORS,
    ENERGY_IMPACT_FACTORS,
    TRANSPORT_IMPACT_FACTORS,
    RECYCLING_BENEFIT_FACTORS,
    PROCESS_IMPACT_FACTORS,
)
from .visualizer import LCAVisualizer

__all__ = [
    "LCAAnalyzer",
    "LifeCycleImpact",
    "LCAVisualizer",
    "MATERIAL_IMPACT_FACTORS",
    "ENERGY_IMPACT_FACTORS",
    "TRANSPORT_IMPACT_FACTORS",
    "RECYCLING_BENEFIT_FACTORS",
    "PROCESS_IMPACT_FACTORS",
]
