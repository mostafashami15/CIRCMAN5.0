# src/circman5/manufacturing/digital_twin/event_notification/event_persistence.py

from typing import Dict, List, Optional, Any, Union
import datetime
import json
import os
from pathlib import Path

from ....utils.logging_config import setup_logger
from ....utils.results_manager import results_manager
from .event_types import Event, EventCategory, EventSeverity


# Custom JSON encoder for handling Timestamp objects and other non-serializable types
class CustomEncoder(json.JSONEncoder):
    """JSON encoder that handles Timestamp objects and other non-serializable types."""

    def default(self, o):
        # Handle pandas Timestamp
        if hasattr(o, "__class__") and o.__class__.__name__ == "Timestamp":
            return str(o)
        # Handle datetime objects
        elif isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        # Handle other pandas/numpy types
        elif hasattr(o, "dtype") or hasattr(o, "iloc"):
            return str(o)
        # Let the base encoder handle it or raise TypeError
        return super().default(o)


class EventPersistence:
    """
    Handles persistence of events using results_manager.

    This class is responsible for saving, retrieving, and managing
    persisted events in the system.
    """

    def __init__(self, max_events: int = 1000):
        """
        Initialize event persistence.

        Args:
            max_events: Maximum number of events to store in memory
        """
        self.logger = setup_logger("event_persistence")
        self.max_events = max_events
        self.events: List[Event] = []
        self.events_dir = self._ensure_events_directory()

        # Load existing events on startup
        self._load_events()

        self.logger.info(
            f"Event persistence initialized with {len(self.events)} events"
        )

    def _ensure_events_directory(self) -> Path:
        """
        Ensure events directory exists in results structure.

        Returns:
            Path: Path to events directory
        """
        try:
            # First check if events dir exists in results_manager
            try:
                events_dir = results_manager.get_path("events")
                return events_dir
            except KeyError:
                # Create events directory if it doesn't exist
                run_dir = results_manager.get_run_dir()
                events_dir = run_dir / "events"
                events_dir.mkdir(parents=True, exist_ok=True)

                # Add to results_manager paths
                results_manager.run_dirs["events"] = events_dir
                return events_dir

        except Exception as e:
            self.logger.error(f"Error ensuring events directory: {str(e)}")
            # Fallback to default location
            default_dir = Path("./events")
            default_dir.mkdir(parents=True, exist_ok=True)
            return default_dir

    def set_max_events(self, max_events: int) -> None:
        """
        Set maximum number of events to store in memory.

        Args:
            max_events: Maximum number of events
        """
        self.max_events = max_events
        # Trim events if needed
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def save_event(self, event: Event) -> None:
        """
        Save an event to persistence.

        Args:
            event: Event to save
        """
        try:
            # Add to in-memory list
            self.events.append(event)

            # Trim if needed
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events :]

            # Save to file
            self._save_to_file(event)

        except Exception as e:
            self.logger.error(f"Error saving event: {str(e)}")

    def _save_to_file(self, event: Event) -> None:
        """
        Save event to file using results_manager.

        Args:
            event: Event to save
        """
        try:
            # Create filename based on date and event ID
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"events_{date_str}.jsonl"
            file_path = self.events_dir / filename

            # Convert event to dict
            event_dict = event.to_dict()

            # Append to file - Using CustomEncoder for Timestamp objects
            with open(file_path, "a") as f:
                f.write(json.dumps(event_dict, cls=CustomEncoder) + "\n")

        except Exception as e:
            self.logger.error(f"Error saving event to file: {str(e)}")

    def get_events(
        self,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get events with optional filtering.

        Args:
            category: Optional category filter
            severity: Optional severity filter
            acknowledged: Optional acknowledged state filter
            limit: Maximum number of events to return

        Returns:
            List[Event]: List of events matching criteria
        """
        # Filter events based on criteria
        filtered_events = self.events

        if category:
            filtered_events = [e for e in filtered_events if e.category == category]

        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]

        if acknowledged is not None:
            filtered_events = [
                e for e in filtered_events if e.acknowledged == acknowledged
            ]

        # Return most recent events up to limit
        return filtered_events[-limit:] if limit > 0 else filtered_events

    def acknowledge_event(self, event_id: str) -> bool:
        """
        Mark an event as acknowledged.

        Args:
            event_id: ID of event to acknowledge

        Returns:
            bool: True if event was found and acknowledged
        """
        for event in self.events:
            if event.event_id == event_id:
                event.acknowledged = True
                self._update_event_in_file(event)
                return True
        return False

    def _update_event_in_file(self, event: Event) -> None:
        """
        Update an event in its file.

        Args:
            event: Event to update
        """
        # This is a simplified implementation
        # In a real implementation, we would need to:
        # 1. Locate the file containing the event
        # 2. Read all events
        # 3. Update the specific event
        # 4. Write all events back
        # This can be inefficient for large event logs

        # For simplicity, we'll just log that this would happen
        self.logger.debug(f"Event {event.event_id} would be updated in persistence")

    def clear_events(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear events from persistence.

        Args:
            older_than_days: Optional, clear events older than this many days

        Returns:
            int: Number of events cleared
        """
        if older_than_days is None:
            # Clear all events
            count = len(self.events)
            self.events = []
            return count

        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        cutoff_str = cutoff_date.isoformat()

        # Filter events
        original_count = len(self.events)
        self.events = [e for e in self.events if e.timestamp > cutoff_str]

        return original_count - len(self.events)

    def _load_events(self) -> None:
        """Load existing events from files."""
        try:
            # Get all event files
            event_files = list(self.events_dir.glob("events_*.jsonl"))

            loaded_events = []

            # Load events from each file
            for file_path in event_files:
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            try:
                                event_dict = json.loads(line.strip())
                                event = Event.from_dict(event_dict)
                                loaded_events.append(event)
                            except Exception as e:
                                self.logger.error(f"Error parsing event: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Error reading event file {file_path}: {str(e)}")

            # Sort events by timestamp
            loaded_events.sort(key=lambda e: e.timestamp)

            # Limit to max_events
            if len(loaded_events) > self.max_events:
                loaded_events = loaded_events[-self.max_events :]

            self.events = loaded_events
            self.logger.info(f"Loaded {len(self.events)} events from persistence")

        except Exception as e:
            self.logger.error(f"Error loading events: {str(e)}")
