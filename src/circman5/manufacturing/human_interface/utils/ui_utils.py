# src/circman5/manufacturing/human_interface/utils/ui_utils.py

"""
UI utility functions for CIRCMAN5.0 Human-Machine Interface.

This module provides helper functions for UI components, including
formatting, styling, and display utilities.
"""

from typing import Dict, Any, Optional, Union, Tuple
import datetime


def format_value(value: Any, unit: Optional[str] = None, precision: int = 2) -> str:
    """
    Format a value for display, with optional unit.

    Args:
        value: Value to format
        unit: Optional unit string
        precision: Number of decimal places for float values

    Returns:
        str: Formatted value string
    """
    if value is None:
        return "N/A"

    if isinstance(value, float):
        # Format float with specified precision
        formatted = f"{value:.{precision}f}"
    elif isinstance(value, int):
        # Format integer
        formatted = f"{value:,}"
    elif isinstance(value, bool):
        # Format boolean as Yes/No
        formatted = "Yes" if value else "No"
    elif isinstance(value, datetime.datetime) or isinstance(value, datetime.date):
        # Format datetime
        formatted = value.isoformat()
    else:
        # Default to string representation
        formatted = str(value)

    # Add unit if provided
    if unit:
        formatted = f"{formatted} {unit}"

    return formatted


def format_timestamp(
    timestamp: Union[str, datetime.datetime], format_str: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format a timestamp for display.

    Args:
        timestamp: Timestamp to format (ISO string or datetime)
        format_str: Format string for strftime

    Returns:
        str: Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
            return dt.strftime(format_str)
        except ValueError:
            return timestamp
    elif isinstance(timestamp, datetime.datetime):
        return timestamp.strftime(format_str)
    else:
        return str(timestamp)


def get_trend_icon(trend: str) -> str:
    """
    Get an icon string for a trend direction.

    Args:
        trend: Trend direction ("up", "down", "stable", "unknown")

    Returns:
        str: Icon string for the trend
    """
    icons = {"up": "↑", "down": "↓", "stable": "→", "unknown": "•"}

    return icons.get(trend.lower(), "•")


def get_severity_style(severity: str) -> Dict[str, str]:
    """
    Get styling for a severity level.

    Args:
        severity: Severity level ("info", "warning", "error", "critical")

    Returns:
        Dict[str, str]: Style dictionary with color, background, and icon
    """
    styles = {
        "info": {
            "color": "#2196F3",
            "background": "#E3F2FD",
            "icon": "ℹ️",
            "border": "1px solid #BBDEFB",
        },
        "warning": {
            "color": "#FF9800",
            "background": "#FFF3E0",
            "icon": "⚠️",
            "border": "1px solid #FFE0B2",
        },
        "error": {
            "color": "#F44336",
            "background": "#FFEBEE",
            "icon": "❌",
            "border": "1px solid #FFCDD2",
        },
        "critical": {
            "color": "#B71C1C",
            "background": "#FFEBEE",
            "icon": "🚨",
            "border": "1px solid #FFCDD2",
        },
    }

    return styles.get(severity.lower(), styles["info"])


def get_status_style(status: str) -> Dict[str, str]:
    """
    Get styling for a status value.

    Args:
        status: Status string (e.g., "running", "idle", "error")

    Returns:
        Dict[str, str]: Style dictionary with color, background, and icon
    """
    styles = {
        "running": {
            "color": "#4CAF50",
            "background": "#E8F5E9",
            "icon": "▶️",
            "border": "1px solid #C8E6C9",
        },
        "idle": {
            "color": "#9E9E9E",
            "background": "#F5F5F5",
            "icon": "⏸️",
            "border": "1px solid #EEEEEE",
        },
        "error": {
            "color": "#F44336",
            "background": "#FFEBEE",
            "icon": "❌",
            "border": "1px solid #FFCDD2",
        },
        "warning": {
            "color": "#FF9800",
            "background": "#FFF3E0",
            "icon": "⚠️",
            "border": "1px solid #FFE0B2",
        },
        "unknown": {
            "color": "#9E9E9E",
            "background": "#F5F5F5",
            "icon": "❓",
            "border": "1px solid #EEEEEE",
        },
    }

    return styles.get(status.lower(), styles["unknown"])


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a value as a percentage.

    Args:
        value: Value to format (0-1 or 0-100)
        precision: Number of decimal places

    Returns:
        str: Formatted percentage string
    """
    # Normalize value to 0-100 range
    if 0 <= value <= 1:
        value = value * 100

    return f"{value:.{precision}f}%"


def get_color_for_value(
    value: float, min_value: float = 0, max_value: float = 100, invert: bool = False
) -> str:
    """
    Get a color for a value on a red-yellow-green gradient.

    Args:
        value: Value to get color for
        min_value: Minimum value in range
        max_value: Maximum value in range
        invert: Whether to invert the color scale (green to red)

    Returns:
        str: Hex color string
    """
    # Normalize value to 0-1 range
    range_size = max_value - min_value
    if range_size == 0:
        normalized = 0.5
    else:
        normalized = (value - min_value) / range_size

    # Clamp to 0-1
    normalized = max(0, min(1, normalized))

    # Invert if requested
    if invert:
        normalized = 1 - normalized

    # Red component (decreases as value increases)
    r = int(255 * (1 - normalized))

    # Green component (increases, then decreases)
    g = int(255 * (normalized if normalized <= 0.5 else 2 - 2 * normalized))

    # Convert to hex
    return f"#{r:02x}{g:02x}00"
