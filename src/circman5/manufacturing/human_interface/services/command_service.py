# src/circman5/manufacturing/human_interface/services/command_service.py

"""
Command service for CIRCMAN5.0 Human-Machine Interface.

This module provides a centralized command handling and routing system
for the human interface, processing user actions and dispatching them
to the appropriate components.
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable
import datetime
import threading
import json

from ....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from ..core.interface_manager import interface_manager


class CommandService:
    """
    Command service for the Human-Machine Interface.

    This service provides a centralized command handling and routing system
    for the human interface, processing user actions and dispatching them
    to the appropriate components.

    Attributes:
        command_handlers: Dictionary of command handlers
        logger: Logger instance for this class
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(CommandService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the command service."""
        if self._initialized:
            return

        self.logger = setup_logger("command_service")
        self.constants = ConstantsService()

        # Initialize command handlers
        self.command_handlers: Dict[str, Callable] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Command history for debugging
        self.command_history: List[Dict[str, Any]] = []
        self.max_history = 100  # Maximum number of commands to keep in history

        # Register with interface manager
        interface_manager.register_component("command_service", self)

        self._initialized = True
        self.logger.info("Command Service initialized")

    def register_handler(self, command: str, handler: Callable) -> None:
        """
        Register a command handler.

        Args:
            command: Command name
            handler: Handler function

        Raises:
            ValueError: If handler already registered
        """
        with self._lock:
            if command in self.command_handlers:
                raise ValueError(f"Handler already registered for command: {command}")

            self.command_handlers[command] = handler
            self.logger.debug(f"Handler registered for command: {command}")

    def unregister_handler(self, command: str) -> bool:
        """
        Unregister a command handler.

        Args:
            command: Command name

        Returns:
            bool: True if handler was unregistered
        """
        with self._lock:
            if command in self.command_handlers:
                del self.command_handlers[command]
                self.logger.debug(f"Handler unregistered for command: {command}")
                return True
            return False

    def execute_command(
        self, command: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a command.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Dict[str, Any]: Command result
        """
        # Ensure params is not None
        params = params or {}
        start_time = datetime.datetime.now()

        try:
            # Check if we have a specific handler
            with self._lock:
                handler = self.command_handlers.get(command)

            # If we have a specific handler, use it
            if handler:
                self.logger.debug(
                    f"Executing command {command} with registered handler"
                )
                result = handler(params)
            else:
                # Otherwise, use interface manager's handle_command
                self.logger.debug(f"Executing command {command} with interface manager")
                result = interface_manager.handle_command(command, params)

            # Calculate execution time
            end_time = datetime.datetime.now()
            execution_time = (
                end_time - start_time
            ).total_seconds() * 1000  # in milliseconds

            # Record in history
            self._add_to_history(command, params, result, execution_time)

            # Log result
            if isinstance(result, dict) and result.get("success", True):
                self.logger.info(
                    f"Command {command} executed successfully in {execution_time:.2f}ms"
                )
            else:
                error = (
                    result.get("error", "Unknown error")
                    if isinstance(result, dict)
                    else "Unknown error"
                )
                self.logger.warning(f"Command {command} failed: {error}")

            return result

        except Exception as e:
            # Calculate execution time
            end_time = datetime.datetime.now()
            execution_time = (
                end_time - start_time
            ).total_seconds() * 1000  # in milliseconds

            # Create error result
            result = {"success": False, "error": str(e)}

            # Record in history
            self._add_to_history(command, params, result, execution_time)

            self.logger.error(f"Error executing command {command}: {str(e)}")
            return result

    def get_command_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get command execution history.

        Args:
            limit: Optional limit on the number of history items to retrieve

        Returns:
            List[Dict[str, Any]]: Command history
        """
        with self._lock:
            if limit is None or limit >= len(self.command_history):
                return self.command_history.copy()
            else:
                return self.command_history[-limit:].copy()

    def _add_to_history(
        self,
        command: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        execution_time: float,
    ) -> None:
        """
        Add a command execution to history.

        Args:
            command: Command name
            params: Command parameters
            result: Command result
            execution_time: Execution time in milliseconds
        """
        with self._lock:
            # Create history entry
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "command": command,
                "params": params,
                "result": {
                    "success": result.get("success", False)
                    if isinstance(result, dict)
                    else False,
                    "error": result.get("error")
                    if isinstance(result, dict) and "error" in result
                    else None,
                },
                "execution_time": execution_time,
            }

            # Add to history
            self.command_history.append(entry)

            # Trim history if needed
            if len(self.command_history) > self.max_history:
                self.command_history = self.command_history[-self.max_history :]


# Create global instance
command_service = CommandService()
