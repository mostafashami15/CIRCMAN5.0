# src/circman5/manufacturing/human_interface/utils/validation.py

"""
Validation utilities for CIRCMAN5.0 Human-Machine Interface.

This module provides functions for validating user input, checking
parameter values, and ensuring data integrity.
"""

from typing import Dict, Any, Optional, Union, Tuple, List, Pattern, Type
import re
from enum import Enum


def validate_number(
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_zero: bool = True,
    allow_negative: bool = False,
    required: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a numeric value.

    Args:
        value: Value to validate
        min_value: Optional minimum value
        max_value: Optional maximum value
        allow_zero: Whether zero is allowed
        allow_negative: Whether negative values are allowed
        required: Whether a value is required

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check if required
    if value is None or value == "":
        if required:
            return False, "Value is required"
        return True, None

    # Try to convert to float
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        return False, "Value must be a number"

    # Check zero constraint
    if not allow_zero and num_value == 0:
        return False, "Zero is not allowed"

    # Check negative constraint
    if not allow_negative and num_value < 0:
        return False, "Negative values are not allowed"

    # Check minimum
    if min_value is not None and num_value < min_value:
        return False, f"Value must be at least {min_value}"

    # Check maximum
    if max_value is not None and num_value > max_value:
        return False, f"Value must be at most {max_value}"

    return True, None


def validate_integer(
    value: Any,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    allow_zero: bool = True,
    allow_negative: bool = False,
    required: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate an integer value.

    Args:
        value: Value to validate
        min_value: Optional minimum value
        max_value: Optional maximum value
        allow_zero: Whether zero is allowed
        allow_negative: Whether negative values are allowed
        required: Whether a value is required

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # First validate as number
    is_valid, error = validate_number(
        value, min_value, max_value, allow_zero, allow_negative, required
    )

    if not is_valid:
        return False, error

    # If not required and no value provided, it's valid
    if (value is None or value == "") and not required:
        return True, None

    # Check if it's an integer
    try:
        num_value = float(value)
        if num_value != int(num_value):
            return False, "Value must be an integer"
    except (ValueError, TypeError):
        return False, "Value must be an integer"

    return True, None


def validate_range(
    value: Any,
    allowed_range: Union[Tuple[float, float], List[float]],
    required: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a value against a range.

    Args:
        value: Value to validate
        allowed_range: Tuple or List with (min, max) values
        required: Whether a value is required

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if len(allowed_range) != 2:
        raise ValueError("Range must have exactly 2 elements")

    min_value, max_value = allowed_range

    return validate_number(
        value, min_value=min_value, max_value=max_value, required=required
    )


def validate_string(
    value: Any,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[Union[str, Pattern]] = None,
    required: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a string value.

    Args:
        value: Value to validate
        min_length: Optional minimum length
        max_length: Optional maximum length
        pattern: Optional regex pattern to match
        required: Whether a value is required

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check if required
    if value is None or value == "":
        if required:
            return False, "Value is required"
        return True, None

    # Convert to string
    str_value = str(value)

    # Check minimum length
    if min_length is not None and len(str_value) < min_length:
        return False, f"Value must be at least {min_length} characters"

    # Check maximum length
    if max_length is not None and len(str_value) > max_length:
        return False, f"Value must be at most {max_length} characters"

    # Check pattern
    if pattern:
        # Compile pattern if it's a string
        if isinstance(pattern, str):
            try:
                pattern = re.compile(pattern)
            except re.error:
                raise ValueError(f"Invalid regex pattern: {pattern}")

        # Check if pattern matches
        if not pattern.match(str_value):
            return False, "Value must match the required pattern"

    return True, None


def validate_enum(
    value: Any,
    allowed_values: Union[List[Any], Tuple[Any, ...], set, Type[Enum]],
    required: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a value against a set of allowed values.

    Args:
        value: Value to validate
        allowed_values: List, Tuple, Set or Enum class of allowed values
        required: Whether a value is required

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check if required
    if value is None or value == "":
        if required:
            return False, "Value is required"
        return True, None

    # Handle Enum class case
    if isinstance(allowed_values, type) and issubclass(allowed_values, Enum):
        enum_values = [item.value for item in allowed_values]

        if value not in enum_values:
            values_str = ", ".join(str(v) for v in enum_values)
            return False, f"Value must be one of: {values_str}"
    else:
        # Handle normal iterable case
        if value not in allowed_values:
            values_str = ", ".join(str(v) for v in allowed_values)
            return False, f"Value must be one of: {values_str}"

    return True, None


def validate_boolean(value: Any, required: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate a boolean value.

    Args:
        value: Value to validate
        required: Whether a value is required

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check if required
    if value is None or value == "":
        if required:
            return False, "Value is required"
        return True, None

    # Check if value is a boolean
    if isinstance(value, bool):
        return True, None

    # Check if value is a string representing a boolean
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value in ["true", "yes", "1", "t", "y"]:
            return True, None
        elif lower_value in ["false", "no", "0", "f", "n"]:
            return True, None

    # Try to convert to int and check if it's 0 or 1
    try:
        int_value = int(value)
        if int_value in [0, 1]:
            return True, None
    except (ValueError, TypeError):
        pass

    return False, "Value must be a boolean"


def validate_parameter(
    param_type: str,
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allowed_values: Optional[List[Any]] = None,
    pattern: Optional[str] = None,
    required: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a parameter based on its type.

    Args:
        param_type: Parameter type ("float", "integer", "string", "boolean", "enum")
        value: Value to validate
        min_value: Optional minimum value for numeric types
        max_value: Optional maximum value for numeric types
        allowed_values: Optional list of allowed values for enum type
        pattern: Optional regex pattern for string type
        required: Whether a value is required

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Normalize parameter type to lowercase
    param_type = param_type.lower()

    # Validate based on parameter type
    if param_type == "float":
        return validate_number(
            value, min_value=min_value, max_value=max_value, required=required
        )
    elif param_type == "integer":
        # Convert min/max values to int if they're not None
        int_min = int(min_value) if min_value is not None else None
        int_max = int(max_value) if max_value is not None else None

        return validate_integer(
            value, min_value=int_min, max_value=int_max, required=required
        )
    elif param_type == "string":
        return validate_string(value, pattern=pattern, required=required)
    elif param_type == "boolean":
        return validate_boolean(value, required=required)
    elif param_type == "enum":
        if not allowed_values:
            raise ValueError("allowed_values must be provided for enum type")
        return validate_enum(value, allowed_values=allowed_values, required=required)
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")
