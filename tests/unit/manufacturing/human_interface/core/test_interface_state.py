# tests/unit/manufacturing/human_interface/core/test_interface_state.py

import pytest
from unittest.mock import patch, MagicMock


def test_interface_state_singleton():
    """Test that interface state follows singleton pattern."""
    from circman5.manufacturing.human_interface.core.interface_state import (
        InterfaceState,
    )
    from circman5.manufacturing.human_interface.core.interface_state import (
        interface_state,
    )

    # Getting the instance twice should return the same object
    instance1 = InterfaceState()
    instance2 = InterfaceState()

    assert instance1 is instance2
    assert instance1 is interface_state


def test_active_view_management():
    """Test active view management."""
    from circman5.manufacturing.human_interface.core.interface_state import (
        interface_state,
        InterfaceState,
    )

    # Create a mock interface state
    mock_state = MagicMock()
    mock_state.active_view = "main_dashboard"
    mock_state.set_active_view = InterfaceState.set_active_view.__get__(
        mock_state, InterfaceState
    )
    mock_state.get_active_view = InterfaceState.get_active_view.__get__(
        mock_state, InterfaceState
    )
    mock_state._save_state = MagicMock()

    # Patch the global instance
    with patch(
        "circman5.manufacturing.human_interface.core.interface_state.interface_state",
        mock_state,
    ):
        # Test setting active view
        mock_state.set_active_view("test_view")
        assert mock_state.get_active_view() == "test_view"
        mock_state._save_state.assert_called_once()


def test_panel_expanded_state():
    """Test panel expanded state management."""
    from circman5.manufacturing.human_interface.core.interface_state import (
        interface_state,
        InterfaceState,
    )

    # Create a mock interface state
    mock_state = MagicMock()
    mock_state.expanded_panels = set(["status", "kpi"])
    mock_state.toggle_panel_expanded = InterfaceState.toggle_panel_expanded.__get__(
        mock_state, InterfaceState
    )
    mock_state.is_panel_expanded = InterfaceState.is_panel_expanded.__get__(
        mock_state, InterfaceState
    )
    mock_state._save_state = MagicMock()

    # Patch the global instance
    with patch(
        "circman5.manufacturing.human_interface.core.interface_state.interface_state",
        mock_state,
    ):
        # Add a panel to expanded set
        assert mock_state.toggle_panel_expanded("test_panel") == True
        assert mock_state.is_panel_expanded("test_panel") == True

        # Remove panel from expanded set
        assert mock_state.toggle_panel_expanded("test_panel") == False
        assert mock_state.is_panel_expanded("test_panel") == False

        # Verify _save_state was called
        assert mock_state._save_state.call_count == 2


def test_alert_filters():
    """Test alert filter management."""
    from circman5.manufacturing.human_interface.core.interface_state import (
        interface_state,
        InterfaceState,
    )

    # Set up default filters
    default_filters = {
        "severity_levels": ["critical", "error", "warning", "info"],
        "categories": ["system", "process", "user"],
        "show_acknowledged": False,
    }

    # Create a mock interface state
    mock_state = MagicMock()
    mock_state.alert_filters = default_filters.copy()
    mock_state.update_alert_filters = InterfaceState.update_alert_filters.__get__(
        mock_state, InterfaceState
    )
    mock_state.get_alert_filters = InterfaceState.get_alert_filters.__get__(
        mock_state, InterfaceState
    )
    mock_state._save_state = MagicMock()

    # Patch the global instance
    with patch(
        "circman5.manufacturing.human_interface.core.interface_state.interface_state",
        mock_state,
    ):
        # Get default filters
        current_filters = mock_state.get_alert_filters()
        assert current_filters == default_filters

        # Update filters
        new_filters = {"severity_levels": ["critical", "error"]}
        mock_state.update_alert_filters(new_filters)

        # Verify filters were updated
        updated_filters = mock_state.get_alert_filters()
        assert updated_filters["severity_levels"] == ["critical", "error"]

        # Other filter settings should be preserved
        for key in default_filters:
            if key != "severity_levels":
                assert key in updated_filters

        # Verify _save_state was called
        mock_state._save_state.assert_called_once()


def test_process_control_mode():
    """Test process control mode management."""
    from circman5.manufacturing.human_interface.core.interface_state import (
        interface_state,
        InterfaceState,
    )

    # Create a mock interface state
    mock_state = MagicMock()
    mock_state.process_control_mode = "monitor"
    mock_state.set_process_control_mode = (
        InterfaceState.set_process_control_mode.__get__(mock_state, InterfaceState)
    )
    mock_state.get_process_control_mode = (
        InterfaceState.get_process_control_mode.__get__(mock_state, InterfaceState)
    )
    mock_state._save_state = MagicMock()

    # Patch the global instance
    with patch(
        "circman5.manufacturing.human_interface.core.interface_state.interface_state",
        mock_state,
    ):
        # Default should be "monitor"
        assert mock_state.get_process_control_mode() == "monitor"

        # Set mode to "manual"
        mock_state.set_process_control_mode("manual")
        assert mock_state.get_process_control_mode() == "manual"

        # Invalid mode should raise ValueError
        with pytest.raises(ValueError):
            mock_state.set_process_control_mode("invalid_mode")

        # Verify _save_state was called
        assert mock_state._save_state.call_count == 1
