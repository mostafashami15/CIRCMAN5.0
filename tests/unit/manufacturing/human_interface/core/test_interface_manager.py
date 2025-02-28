# tests/unit/manufacturing/human_interface/core/test_interface_manager.py

import pytest
from unittest.mock import patch, MagicMock


def test_interface_manager_singleton():
    """Test that interface manager follows singleton pattern."""
    from circman5.manufacturing.human_interface.core.interface_manager import (
        InterfaceManager,
    )
    from circman5.manufacturing.human_interface.core.interface_manager import (
        interface_manager,
    )

    # Getting the instance twice should return the same object
    instance1 = InterfaceManager()
    instance2 = InterfaceManager()

    assert instance1 is instance2
    assert instance1 is interface_manager


def test_register_component():
    """Test component registration in interface manager."""
    from circman5.manufacturing.human_interface.core.interface_manager import (
        InterfaceManager,
    )

    # Create a mock with all required attributes
    mock_manager = MagicMock(spec=InterfaceManager)
    mock_manager.components = {}
    mock_manager.logger = MagicMock()  # Add this line
    mock_manager._lock = MagicMock()  # Add this line

    # Bind the original method
    original_method = InterfaceManager.register_component
    mock_manager.register_component = original_method.__get__(
        mock_manager, InterfaceManager
    )

    # Patch the interface_manager
    with patch(
        "circman5.manufacturing.human_interface.core.interface_manager.interface_manager",
        mock_manager,
    ):
        # Create mock component
        mock_component = MagicMock()

        # Register component
        mock_manager.register_component("test_component", mock_component)

        # Verify registration
        assert "test_component" in mock_manager.components
        assert mock_manager.components["test_component"] is mock_component


def test_handle_command():
    """Test command handling."""
    from circman5.manufacturing.human_interface.core.interface_manager import (
        InterfaceManager,
    )

    # Create a mock with all required attributes
    mock_manager = MagicMock(spec=InterfaceManager)
    mock_manager.components = {}
    mock_manager.logger = MagicMock()  # Add this line
    mock_manager._lock = MagicMock()  # Add this line

    # Bind the original method
    original_method = InterfaceManager.handle_command
    mock_manager.handle_command = original_method.__get__(
        mock_manager, InterfaceManager
    )

    # Create mock component with command handler
    mock_component = MagicMock()
    mock_component.handle_command.return_value = {
        "handled": True,
        "success": True,
        "data": "test",
    }

    # Add component to mock manager
    mock_manager.components["test_component"] = mock_component

    # Patch the interface_manager
    with patch(
        "circman5.manufacturing.human_interface.core.interface_manager.interface_manager",
        mock_manager,
    ):
        # Handle command
        result = mock_manager.handle_command("test_command", {"param": "value"})

        # Verify component's handle_command was called
        mock_component.handle_command.assert_called_once_with(
            "test_command", {"param": "value"}
        )

        # Verify result
        assert result["success"] is True
        assert result["data"] == "test"


def test_register_event_handler():
    """Test event handler registration."""
    from circman5.manufacturing.human_interface.core.interface_manager import (
        InterfaceManager,
    )

    # Create a mock with all required attributes
    mock_manager = MagicMock(spec=InterfaceManager)
    mock_manager.event_handlers = {}
    mock_manager.logger = MagicMock()  # Add this line
    mock_manager._lock = MagicMock()  # Add this line

    # Bind the original methods
    register_method = InterfaceManager.register_event_handler
    mock_manager.register_event_handler = register_method.__get__(
        mock_manager, InterfaceManager
    )

    trigger_method = InterfaceManager.trigger_event
    mock_manager.trigger_event = trigger_method.__get__(mock_manager, InterfaceManager)

    # Patch the interface_manager
    with patch(
        "circman5.manufacturing.human_interface.core.interface_manager.interface_manager",
        mock_manager,
    ):
        # Create mock handler
        mock_handler = MagicMock()

        # Register handler
        mock_manager.register_event_handler("test_event", mock_handler)

        # Verify registration
        assert "test_event" in mock_manager.event_handlers
        assert mock_handler in mock_manager.event_handlers["test_event"]

        # Test triggering event
        event_data = {"test": "data"}
        mock_manager.trigger_event("test_event", event_data)

        # Verify handler was called
        mock_handler.assert_called_once_with(event_data)
