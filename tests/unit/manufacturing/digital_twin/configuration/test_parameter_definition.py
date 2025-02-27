# tests/unit/manufacturing/digital_twin/configuration/test_parameter_definition.py

import pytest
from circman5.manufacturing.digital_twin.configuration.parameter_definition import (
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
    ParameterGroup,
)


def test_parameter_definition_validation():
    """Test parameter validation for different types."""
    # Float parameter
    float_param = ParameterDefinition(
        name="test_float",
        description="Test float parameter",
        parameter_type=ParameterType.FLOAT,
        default_value=10.0,
        path="test.float",
        min_value=0.0,
        max_value=100.0,
    )

    # Valid value
    is_valid, error = float_param.validate(50.0)
    assert is_valid is True
    assert error is None

    # Integer converted to float is valid
    is_valid, error = float_param.validate(50)
    assert is_valid is True
    assert error is None

    # Invalid type
    is_valid, error = float_param.validate("50")
    assert is_valid is False
    assert error is not None and "must be a number" in error

    # Below minimum
    is_valid, error = float_param.validate(-10.0)
    assert is_valid is False
    assert error is not None and "below minimum" in error

    # Above maximum
    is_valid, error = float_param.validate(150.0)
    assert is_valid is False
    assert error is not None and "above maximum" in error

    # Integer parameter
    int_param = ParameterDefinition(
        name="test_int",
        description="Test integer parameter",
        parameter_type=ParameterType.INTEGER,
        default_value=10,
        path="test.int",
        min_value=0,
        max_value=100,
    )

    # Valid value
    is_valid, error = int_param.validate(50)
    assert is_valid is True
    assert error is None

    # Float not valid for integer
    is_valid, error = int_param.validate(50.5)
    assert is_valid is False
    assert error is not None and "must be an integer" in error

    # String parameter with pattern
    string_param = ParameterDefinition(
        name="test_string",
        description="Test string parameter",
        parameter_type=ParameterType.STRING,
        default_value="test",
        path="test.string",
        pattern="^[a-z]+$",
    )

    # Valid value
    is_valid, error = string_param.validate("test")
    assert is_valid is True
    assert error is None

    # Invalid pattern
    is_valid, error = string_param.validate("Test123")
    assert is_valid is False
    assert error is not None and "does not match pattern" in error

    # Boolean parameter
    bool_param = ParameterDefinition(
        name="test_bool",
        description="Test boolean parameter",
        parameter_type=ParameterType.BOOLEAN,
        default_value=True,
        path="test.bool",
    )

    # Valid value
    is_valid, error = bool_param.validate(False)
    assert is_valid is True
    assert error is None

    # Invalid type
    is_valid, error = bool_param.validate("True")
    assert is_valid is False
    assert error is not None and "must be a boolean" in error

    # Enum parameter
    enum_param = ParameterDefinition(
        name="test_enum",
        description="Test enum parameter",
        parameter_type=ParameterType.ENUM,
        default_value="option1",
        path="test.enum",
        enum_values=["option1", "option2", "option3"],
    )

    # Valid value
    is_valid, error = enum_param.validate("option2")
    assert is_valid is True
    assert error is None

    # Invalid value
    is_valid, error = enum_param.validate("option4")
    assert is_valid is False
    assert error is not None and "must be one of" in error


def test_parameter_format_value():
    """Test parameter value formatting with units."""
    # Parameter with units
    param = ParameterDefinition(
        name="test",
        description="Test parameter",
        parameter_type=ParameterType.FLOAT,
        default_value=10.0,
        path="test",
        units="°C",
    )

    formatted = param.format_value(25.5)
    assert formatted == "25.5 °C"

    # Parameter without units
    param = ParameterDefinition(
        name="test",
        description="Test parameter",
        parameter_type=ParameterType.FLOAT,
        default_value=10.0,
        path="test",
    )

    formatted = param.format_value(25.5)
    assert formatted == "25.5"
