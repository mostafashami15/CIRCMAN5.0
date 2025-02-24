# src/circman5/adapters/base/config_manager.py

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from .adapter_base import ConfigAdapterBase


class ConfigurationManager:
    """Manages configuration loading and validation across different adapters."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional base path for configuration files
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path or Path("config")
        self.adapters: Dict[str, ConfigAdapterBase] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}

    def register_adapter(self, name: str, adapter: ConfigAdapterBase) -> None:
        """
        Register a new configuration adapter.

        Args:
            name: Unique identifier for the adapter
            adapter: Adapter instance to register

        Raises:
            ValueError: If adapter name already exists
        """
        if name in self.adapters:
            raise ValueError(f"Adapter already registered: {name}")

        self.adapters[name] = adapter
        self.logger.info(f"Registered adapter: {name}")

    def load_config(self, adapter_name: str) -> Dict[str, Any]:
        """Load configuration using specified adapter."""
        if adapter_name not in self.adapters:
            self.logger.error(f"Unknown adapter: {adapter_name}")
            raise ValueError(f"Unknown adapter: {adapter_name}")

        adapter = self.adapters[adapter_name]
        self.logger.info(
            f"Loading config for {adapter_name} using adapter {adapter.__class__.__name__}"
        )

        try:
            config = adapter.load_config()
            self.logger.info(
                f"Loaded config for {adapter_name}: keys: {list(config.keys())}"
            )

            if not adapter.validate_config(config):
                self.logger.warning(
                    f"Invalid configuration for {adapter_name}, using defaults"
                )
                config = adapter.get_defaults()

            self.configs[adapter_name] = config
            return config

        except Exception as e:
            self.logger.error(f"Error loading config for {adapter_name}: {str(e)}")
            raise

    def get_config(self, adapter_name: str) -> Dict[str, Any]:
        """
        Get loaded configuration for an adapter.

        Args:
            adapter_name: Name of adapter

        Returns:
            Dict[str, Any]: Current configuration

        Raises:
            ValueError: If adapter not found or config not loaded
        """
        if adapter_name not in self.configs:
            self.load_config(adapter_name)

        return self.configs[adapter_name]

    def reload_all(self) -> None:
        """Reload all registered configurations."""
        for name in self.adapters.keys():
            self.load_config(name)

    def get_adapter(self, name: str) -> ConfigAdapterBase:
        """
        Get registered adapter by name.

        Args:
            name: Adapter name

        Returns:
            ConfigAdapterBase: Registered adapter

        Raises:
            ValueError: If adapter not found
        """
        if name not in self.adapters:
            raise ValueError(f"Unknown adapter: {name}")

        return self.adapters[name]
