# src/circman5/manufacturing/digital_twin/event_notification/publishers.py

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

from ....utils.logging_config import setup_logger
from circman5.adapters.services.constants_service import ConstantsService
from .event_manager import event_manager
from .event_types import (
    Event,
    EventCategory,
    EventSeverity,
    SystemStateEvent,
    ThresholdEvent,
    OptimizationEvent,
    ErrorEvent,
    UserActionEvent,
)


class Publisher:
    """Base class for event publishers."""

    def __init__(self, source: str):
        """
        Initialize publisher.

        Args:
            source: Source identifier for events
        """
        self.logger = setup_logger(f"publisher_{source}")
        self.constants = ConstantsService()
        self.source = source

    def publish_event(self, event: Event) -> None:
        """
        Publish an event.

        Args:
            event: Event to publish
        """
        # Set source if not already set
        if not event.source or event.source == "system":
            event.source = self.source

        # Publish to event manager
        event_manager.publish(event)
        self.logger.debug(f"Published event: {event.event_id} - {event.message}")

    def publish_system_state_event(
        self,
        previous_state: str,
        new_state: str,
        severity: EventSeverity = EventSeverity.INFO,
        **kwargs,
    ) -> None:
        """
        Publish a system state change event.

        Args:
            previous_state: Previous system state
            new_state: New system state
            severity: Event severity
            **kwargs: Additional event details
        """
        event = SystemStateEvent(
            previous_state=previous_state,
            new_state=new_state,
            severity=severity,
            source=self.source,
            **kwargs,
        )
        self.publish_event(event)

    def publish_threshold_event(
        self,
        parameter: str,
        threshold: float,
        actual_value: float,
        severity: EventSeverity = EventSeverity.WARNING,
        **kwargs,
    ) -> None:
        """
        Publish a threshold breach event.

        Args:
            parameter: Parameter that breached threshold
            threshold: Threshold value
            actual_value: Actual value
            severity: Event severity
            **kwargs: Additional event details
        """
        event = ThresholdEvent(
            parameter=parameter,
            threshold=threshold,
            actual_value=actual_value,
            severity=severity,
            source=self.source,
            **kwargs,
        )
        self.publish_event(event)

    def publish_optimization_event(
        self,
        optimization_type: str,
        potential_improvement: float,
        recommended_action: str,
        severity: EventSeverity = EventSeverity.INFO,
        **kwargs,
    ) -> None:
        """
        Publish an optimization opportunity event.

        Args:
            optimization_type: Type of optimization
            potential_improvement: Potential improvement percentage
            recommended_action: Recommended action
            severity: Event severity
            **kwargs: Additional event details
        """
        event = OptimizationEvent(
            optimization_type=optimization_type,
            potential_improvement=potential_improvement,
            recommended_action=recommended_action,
            severity=severity,
            source=self.source,
            **kwargs,
        )
        self.publish_event(event)

    def publish_error_event(
        self,
        error_type: str,
        error_message: str,
        stacktrace: Optional[str] = None,
        severity: EventSeverity = EventSeverity.ERROR,
        **kwargs,
    ) -> None:
        """
        Publish an error event.

        Args:
            error_type: Type of error
            error_message: Error message
            stacktrace: Optional stacktrace
            severity: Event severity
            **kwargs: Additional event details
        """
        event = ErrorEvent(
            error_type=error_type,
            error_message=error_message,
            stacktrace=stacktrace,
            severity=severity,
            source=self.source,
            **kwargs,
        )
        self.publish_event(event)

    def publish_user_action_event(
        self,
        user_id: str,
        action: str,
        action_details: Dict[str, Any],
        severity: EventSeverity = EventSeverity.INFO,
        **kwargs,
    ) -> None:
        """
        Publish a user action event.

        Args:
            user_id: User identifier
            action: Action performed
            action_details: Details of the action
            severity: Event severity
            **kwargs: Additional event details
        """
        event = UserActionEvent(
            user_id=user_id,
            action=action,
            action_details=action_details,
            severity=severity,
            source=self.source,
            **kwargs,
        )
        self.publish_event(event)


class DigitalTwinPublisher(Publisher):
    """Publisher for Digital Twin events."""

    def __init__(self):
        """Initialize Digital Twin publisher."""
        super().__init__(source="digital_twin")

    def publish_state_update(
        self, previous_state: Dict[str, Any], new_state: Dict[str, Any]
    ) -> None:
        """
        Publish state update event.

        Args:
            previous_state: Previous Digital Twin state
            new_state: New Digital Twin state
        """
        # Extract relevant state information
        prev_status = self._extract_status(previous_state)
        new_status = self._extract_status(new_state)

        # Only publish if status changed
        if prev_status != new_status:
            self.publish_system_state_event(
                previous_state=prev_status,
                new_state=new_status,
                details={"previous_state": previous_state, "new_state": new_state},
            )

    def _extract_status(self, state: Dict[str, Any]) -> str:
        """
        Extract system status from state.

        Args:
            state: Digital Twin state

        Returns:
            str: Status string
        """
        return state.get("system_status", "unknown")

    def publish_parameter_threshold_event(
        self,
        parameter_path: str,
        parameter_name: str,
        threshold: float,
        actual_value: float,
        state: Dict[str, Any],
        severity: EventSeverity = EventSeverity.WARNING,
    ) -> None:
        """
        Publish parameter threshold event.

        Args:
            parameter_path: Path to parameter in state
            parameter_name: Human-readable parameter name
            threshold: Threshold value
            actual_value: Actual value
            state: Current state
            severity: Event severity
        """
        self.publish_threshold_event(
            parameter=parameter_name,
            threshold=threshold,
            actual_value=actual_value,
            severity=severity,
            details={
                "parameter_path": parameter_path,
                "state_timestamp": state.get("timestamp", ""),
                "state_snapshot": state,
            },
        )

    def publish_simulation_result(
        self,
        simulation_id: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        improvement: Optional[float] = None,
    ) -> None:
        """
        Publish simulation result event.

        Args:
            simulation_id: Simulation identifier
            parameters: Simulation parameters
            results: Simulation results
            improvement: Optional improvement percentage
        """
        # Determine if this is an optimization event
        if improvement is not None and improvement > 0:
            self.publish_optimization_event(
                optimization_type="process_parameters",
                potential_improvement=improvement,
                recommended_action="Apply optimized parameters",
                details={
                    "simulation_id": simulation_id,
                    "parameters": parameters,
                    "results": results,
                },
            )
        else:
            # Regular simulation event
            self.publish_event(
                Event(
                    category=EventCategory.PROCESS,
                    message=f"Simulation {simulation_id} completed",
                    details={
                        "simulation_id": simulation_id,
                        "parameters": parameters,
                        "results": results,
                    },
                )
            )
