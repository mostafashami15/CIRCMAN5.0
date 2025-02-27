# src/circman5/manufacturing/digital_twin/configuration/parameter_definition.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
import re


class ParameterType(Enum):
    """Types of parameters that can be configured."""

    FLOAT = "float"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"
    ENUM = "enum"


@dataclass
class ParameterDefinition:
    """
    Definition of a configurable parameter.

    Attributes:
        name: Name of the parameter
        description: Human-readable description
        parameter_type: Type of parameter (float, int, string, etc.)
        default_value: Default value if not specified
        path: Path to the parameter in the state (e.g., "production_line.temperature")
        min_value: Minimum allowed value (for numeric types)
        max_value: Maximum allowed value (for numeric types)
        enum_values: Allowed values for enum types
        pattern: Regex pattern for string validation
        units: Units of measurement (e.g., "°C", "kWh")
        tags: Tags for categorizing parameters
        requires_restart: Whether changing this parameter requires system restart
    """

    name: str
    description: str
    parameter_type: ParameterType
    default_value: Any
    path: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    units: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    requires_restart: bool = False

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a value against this parameter definition.

        Args:
            value: Value to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        # Initialize variables to handle type issues
        value_num = 0.0
        value_str = ""

        # Type validation
        if self.parameter_type == ParameterType.FLOAT:
            if not isinstance(value, (float, int)):
                return False, f"Value must be a number, got {type(value).__name__}"

            # Convert to float for comparison
            value_num = float(value)

        elif self.parameter_type == ParameterType.INTEGER:
            if not isinstance(value, int):
                return False, f"Value must be an integer, got {type(value).__name__}"
            value_num = value

        elif self.parameter_type == ParameterType.STRING:
            if not isinstance(value, str):
                return False, f"Value must be a string, got {type(value).__name__}"
            value_str = value

        elif self.parameter_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Value must be a boolean, got {type(value).__name__}"

        elif self.parameter_type == ParameterType.ENUM:
            if self.enum_values is None:
                return False, "Enum values not defined for enum parameter"

            if value not in self.enum_values:
                return False, f"Value must be one of {self.enum_values}, got {value}"

        # Range validation for numeric types
        if self.parameter_type in [ParameterType.FLOAT, ParameterType.INTEGER]:
            if self.min_value is not None and value_num < self.min_value:
                return False, f"Value {value} is below minimum {self.min_value}"

            if self.max_value is not None and value_num > self.max_value:
                return False, f"Value {value} is above maximum {self.max_value}"

        # Pattern validation for string types
        if self.parameter_type == ParameterType.STRING and self.pattern:
            if not re.match(self.pattern, value_str):
                return False, f"Value '{value}' does not match pattern '{self.pattern}'"

        return True, None

    def format_value(self, value: Any) -> str:
        """
        Format a value with units for display.

        Args:
            value: Value to format

        Returns:
            str: Formatted value with units
        """
        if self.units:
            return f"{value} {self.units}"
        return str(value)


class ParameterCategory(Enum):
    """Categories for grouping parameters."""

    PROCESS = "process"
    ENVIRONMENT = "environment"
    MATERIAL = "material"
    ENERGY = "energy"
    QUALITY = "quality"
    SYSTEM = "system"


@dataclass
class ParameterGroup:
    """
    Group of related parameters.

    Attributes:
        name: Name of the group
        description: Human-readable description
        category: Category this group belongs to
        parameters: List of parameter definitions
    """

    name: str
    description: str
    category: ParameterCategory
    parameters: List[ParameterDefinition] = field(default_factory=list)
