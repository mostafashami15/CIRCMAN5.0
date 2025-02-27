# src/circman5/manufacturing/digital_twin/integration/__init__.py

"""Integration modules for the Digital Twin system."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ai_integration import AIIntegration
    from .lca_integration import LCAIntegration

# These imports will only be used when explicitly imported by other modules
# to avoid circular imports during module initialization
__all__ = ["AIIntegration", "LCAIntegration"]
