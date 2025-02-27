# tests/unit/manufacturing/digital_twin/configuration/test_config_manager.py

import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from circman5.manufacturing.digital_twin.configuration.config_manager import (
    ConfigurationManager,
    ParameterDefinition,
    ParameterType,
)
from circman5.manufacturing.digital_twin.event_notification.event_types import (
    EventCategory,
    EventSeverity,
)


@pytest.fixture
def config_manager():
    """Get a ConfigurationManager instance for testing."""
    # Reset the singleton
    ConfigurationManager._instance = None

    # Mock constants service to return test configuration
    with patch(
        "circman5.manufacturing.digital_twin.configuration.config_manager.ConstantsService"
    ) as mock_cs:
        # Setup mock constants service
        mock_constants = MagicMock()
        mock_constants.get_digital_twin_config.return_value = {
            "PARAMETERS": {
                "PARAMETER_GROUPS": [
                    {
                        "name": "Test Group",
                        "description": "Test group description",
                        "category": "process",
                        "parameters": [
                            {
                                "name": "test_param",
                                "description": "Test parameter",
                                "type": "float",
                                "default_value": 10.0,
                                "path": "test.param",
                                "min_value": 0.0,
                                "max_value": 100.0,
                            }
                        ],
                    }
                ]
            }
        }
        mock_cs.return_value = mock_constants

        # Mock results_manager to avoid file operations
        with patch(
            "circman5.manufacturing.digital_twin.configuration.config_manager.results_manager"
        ) as mock_rm:
            # Create a temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Setup mock results_manager
                mock_rm.get_path.return_value = Path(temp_dir)

                # Mock event publisher
                with patch(
                    "circman5.manufacturing.digital_twin.configuration.config_manager.ConfigurationPublisher"
                ):
                    manager = ConfigurationManager()
                    yield manager


def test_get_parameter_value(config_manager):
    """Test getting parameter value."""
    # Parameter should exist from the mock setup
    value = config_manager.get_parameter_value("test_param")
    assert value == 10.0

    # Test non-existent parameter
    with pytest.raises(KeyError):
        config_manager.get_parameter_value("nonexistent_param")


def test_set_parameter_value(config_manager):
    """Test setting parameter value."""
    # Valid value
    success, error = config_manager.set_parameter_value("test_param", 50.0)
    assert success is True
    assert error is None

    # Check that value was updated
    value = config_manager.get_parameter_value("test_param")
    assert value == 50.0

    # Invalid value (below minimum)
    success, error = config_manager.set_parameter_value("test_param", -10.0)
    assert success is False
    assert "below minimum" in error

    # Check that value was not updated
    value = config_manager.get_parameter_value("test_param")
    assert value == 50.0


def test_get_parameter_definition(config_manager):
    """Test getting parameter definition."""
    param_def = config_manager.get_parameter_definition("test_param")
    assert param_def.name == "test_param"
    assert param_def.parameter_type == ParameterType.FLOAT
    assert param_def.default_value == 10.0
    assert param_def.min_value == 0.0
    assert param_def.max_value == 100.0


def test_reset_parameter_to_default(config_manager):
    """Test resetting parameter to default value."""
    # Change parameter value
    config_manager.set_parameter_value("test_param", 50.0)
    assert config_manager.get_parameter_value("test_param") == 50.0

    # Reset to default
    config_manager.reset_parameter_to_default("test_param")
    assert config_manager.get_parameter_value("test_param") == 10.0


def test_apply_parameters_to_state(config_manager):
    """Test applying parameters to state."""
    # Initial state
    state = {"test": {"other": "value"}}

    # Apply parameters
    updated_state = config_manager.apply_parameters_to_state(state)

    # Check that parameter was applied
    assert "param" in updated_state["test"]
    assert updated_state["test"]["param"] == 10.0

    # Check that other values were preserved
    assert updated_state["test"]["other"] == "value"
