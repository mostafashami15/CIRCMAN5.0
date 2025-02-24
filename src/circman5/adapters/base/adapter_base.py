# src/circman5/adapters/base/adapter_base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging


class ConfigAdapterBase(ABC):
    """Base interface for configuration adapters."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize adapter with optional config path."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path

    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from source.

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary to validate

        Returns:
            bool: True if configuration is valid
        """
        pass

    @abstractmethod
    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default configuration values.

        Returns:
            Dict[str, Any]: Default configuration
        """
        pass

    def _load_json_config(self, path: Path) -> Dict[str, Any]:
        """
        Helper method to load JSON configuration.

        Args:
            path: Path to JSON config file

        Returns:
            Dict[str, Any]: Loaded configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If invalid JSON
        """
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {path}")
            raise ValueError(f"Invalid JSON configuration: {str(e)}")
