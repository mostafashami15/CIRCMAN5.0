# src/circman5/adapters/services/constants_service.py

from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ..base.config_manager import ConfigurationManager
from ..config.manufacturing import ManufacturingAdapter
from ..config.impact_factors import ImpactFactorsAdapter
from ..config.optimization import OptimizationAdapter


class ConstantsService:
    """Service for centralized access to all system constants."""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConstantsService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the constants service."""
        if self._initialized:
            return

        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = ConfigurationManager()

        # Register all adapters
        self._register_adapters()

        # Load initial configurations
        self._load_all_configs()

        self._initialized = True

    def _register_adapters(self) -> None:
        """Register all configuration adapters."""
        try:
            self.config_manager.register_adapter(
                "manufacturing", ManufacturingAdapter()
            )
            self.config_manager.register_adapter(
                "impact_factors", ImpactFactorsAdapter()
            )
            self.config_manager.register_adapter("optimization", OptimizationAdapter())
        except Exception as e:
            self.logger.error(f"Error registering adapters: {str(e)}")
            raise

    def _load_all_configs(self) -> None:
        """Load all configurations."""
        try:
            for adapter_name in self.config_manager.adapters.keys():
                self.config_manager.load_config(adapter_name)
        except Exception as e:
            self.logger.error(f"Error loading configurations: {str(e)}")
            raise

    def get_manufacturing_constants(self) -> Dict[str, Any]:
        """Get manufacturing constants."""
        return self.config_manager.get_config("manufacturing")

    def get_impact_factors(self) -> Dict[str, Any]:
        """Get impact factors constants."""
        return self.config_manager.get_config("impact_factors")

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self.config_manager.get_config("optimization")

    def get_constant(self, adapter: str, key: str) -> Any:
        """
        Get specific constant by adapter and key.

        Args:
            adapter: Adapter name
            key: Configuration key

        Returns:
            Any: Configuration value

        Raises:
            KeyError: If key not found
        """
        config = self.config_manager.get_config(adapter)
        if key not in config:
            raise KeyError(f"Key not found in {adapter} config: {key}")
        return config[key]

    def reload_configs(self) -> None:
        """Reload all configurations."""
        self.config_manager.reload_all()
