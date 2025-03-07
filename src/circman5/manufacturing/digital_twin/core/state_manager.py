# src/circman5/manufacturing/digital_twin/core/state_manager.py

"""
State Manager module for CIRCMAN5.0 Digital Twin.

This module implements state tracking, validation, and history management for the digital twin system.
It maintains the current state and historical states of the manufacturing system.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import copy
import datetime
import threading
from collections import deque
import json

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from circman5.adapters.services.constants_service import ConstantsService


class StateManager:
    """
    Manages the state of the digital twin system.

    The StateManager maintains the current state of the system and historical state information,
    providing functionality for state validation, update, and history tracking.

    Attributes:
        current_state: Current state of the digital twin
        state_history: Historical states of the digital twin
        history_length: Maximum number of historical states to keep
        logger: Logger instance for this class
    """

    # Singleton pattern implementation
    _instance = None
    _initialized = False
    _init_lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance is created."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, history_length: Optional[int] = None):
        """
        Initialize the StateManager.

        Args:
            history_length: Maximum number of historical states to keep
        """
        # Skip initialization if already done
        with self._init_lock:
            if StateManager._initialized:
                # If a new history_length is provided and differs from current, update it
                if history_length is not None and history_length != self.history_length:
                    self.logger.info(
                        f"Updating history length from {self.history_length} to {history_length}"
                    )
                    self.history_length = history_length
                    self.state_history = deque(
                        list(self.state_history), maxlen=history_length
                    )
                return

            # Load configuration from constants service
            self.constants = ConstantsService()
            self.dt_config = self.constants.get_digital_twin_config()
            self.state_config = self.dt_config.get("STATE_MANAGEMENT", {})

            # Use provided history length or get from config
            self.history_length = history_length or self.state_config.get(
                "default_history_length", 1000
            )

            # Initialize state containers
            self.current_state: Dict[str, Any] = {}
            self.state_history: deque = deque(maxlen=self.history_length)

            # Setup logging
            self.logger = setup_logger("state_manager")
            self.logger.info(
                f"StateManager initialized with history length {self.history_length}"
            )

            # Mark as initialized
            StateManager._initialized = True

    # The rest of your methods remain unchanged
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the current state and add to history.

        Args:
            state: New state to set
        """
        # Validate state
        is_valid, message = self.validate_state(state)
        if not is_valid:
            self.logger.warning(f"Setting state with validation warning: {message}")

        # Make deep copy to avoid reference issues
        state_copy = copy.deepcopy(state)

        # Add timestamp if not present
        if "timestamp" not in state_copy:
            state_copy["timestamp"] = datetime.datetime.now().isoformat()

        # Store previous state in history
        if self.current_state:
            self.state_history.append(copy.deepcopy(self.current_state))

        # Set new current state
        self.current_state = state_copy
        self.logger.debug(f"State updated, history length: {len(self.state_history)}")

    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Update parts of the current state.

        Args:
            updates: Dictionary with updates to apply to the current state
        """
        if not self.current_state:
            self.logger.warning(
                "Attempting to update empty state, setting as new state instead"
            )
            self.set_state(updates)
            return

        # Make deep copy of current state
        new_state = copy.deepcopy(self.current_state)

        # Apply updates with deep merge
        self._deep_update(new_state, updates)

        # Set the updated state
        self.set_state(new_state)

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state.

        Returns:
            Dict[str, Any]: Copy of the current state
        """
        return copy.deepcopy(self.current_state)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical states.

        Args:
            limit: Optional limit on the number of historical states to retrieve

        Returns:
            List[Dict[str, Any]]: List of historical states
        """
        if limit is None or limit >= len(self.state_history):
            # Return a copy of all history
            return list(copy.deepcopy(self.state_history))
        else:
            # Return limited history (most recent states)
            history_list = list(self.state_history)
            return copy.deepcopy(history_list[-limit:])

    def get_state_at_time(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Get the state at a specific time.

        Args:
            timestamp: ISO format timestamp to look for

        Returns:
            Optional[Dict[str, Any]]: State at the specified time, or None if not found
        """
        # Check current state first
        if self.current_state.get("timestamp") == timestamp:
            return copy.deepcopy(self.current_state)

        # Look through history
        for state in reversed(self.state_history):
            if state.get("timestamp") == timestamp:
                return copy.deepcopy(state)

        return None

    def clear_history(self) -> None:
        """Clear the state history."""
        self.state_history.clear()
        self.logger.info("State history cleared")

    def validate_state(self, state: Any) -> Tuple[bool, str]:
        """
        Validate a state dictionary.

        Args:
            state: State dictionary to validate

        Returns:
            Tuple[bool, str]: (is_valid, message) where message explains any validation issues
        """
        # Check if state is a dictionary
        if not isinstance(state, dict):
            return False, "State must be a dictionary"

        # Check for required fields
        # This is a simple example; in a real implementation, this would be more comprehensive
        if "timestamp" in state:
            try:
                # Validate ISO format timestamp
                datetime.datetime.fromisoformat(state["timestamp"])
            except ValueError:
                return False, "Invalid timestamp format, expected ISO format"

        # Check for production_line if present
        if "production_line" in state and not isinstance(
            state["production_line"], dict
        ):
            return False, "production_line must be a dictionary"

        # All checks passed
        return True, "Valid state"

    def export_state(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Export the current state to a JSON file.

        Args:
            file_path: Optional path to save the state. If not provided, uses results_manager.

        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            filename = "state_default.json"

            # Generate filename with timestamp if not provided
            if file_path:
                save_path = Path(file_path)
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"state_{timestamp}.json"
                save_path = Path(filename)

            # Export state to file
            with open(save_path, "w") as f:
                json.dump(self.current_state, f, indent=2)

            # If using results_manager, save and clean up
            if file_path is None:
                results_manager.save_file(save_path, "digital_twin")
                save_path.unlink()  # Remove temporary file
                self.logger.info(f"State exported to results manager: {filename}")
            else:
                self.logger.info(f"State exported to {save_path}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to export state: {str(e)}")
            return False

    def import_state(self, file_path: Union[str, Path]) -> bool:
        """
        Import a state from a JSON file.

        Args:
            file_path: Path to the state file

        Returns:
            bool: True if import was successful, False otherwise
        """
        try:
            # Handle results_manager paths if path is not absolute
            path = Path(file_path)
            if not path.is_absolute():
                # Check if file exists in the digital_twin directory
                dt_dir = results_manager.get_path("digital_twin")
                full_path = dt_dir / path
                if full_path.exists():
                    path = full_path

            # Load and set state
            with open(path, "r") as f:
                state = json.load(f)
            self.set_state(state)
            self.logger.info(f"State imported from {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import state: {str(e)}")
            return False

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively update a dictionary.

        Args:
            d: Base dictionary to update
            u: Dictionary with updates to apply

        Returns:
            Dict[str, Any]: Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                # Recursive update for nested dictionaries
                self._deep_update(d[k], v)
            else:
                # Direct update for non-dictionary values or new keys
                d[k] = v
        return d

    @classmethod
    def _reset(cls):
        """Reset the singleton state (for testing only)."""
        with cls._init_lock:
            cls._instance = None
            cls._initialized = False
