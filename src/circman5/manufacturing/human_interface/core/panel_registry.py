from typing import Dict, Any, Callable, Optional
import threading

# Dictionary to store panel renderers
_panel_renderers: Dict[str, Callable] = {}
_lock = threading.RLock()


def register_panel_renderer(panel_type: str, renderer_func: Callable) -> None:
    """
    Register a panel renderer function.

    Args:
        panel_type: Type identifier for the panel
        renderer_func: Function that renders panel data
    """
    with _lock:
        _panel_renderers[panel_type] = renderer_func


def get_panel_renderer(panel_type: str) -> Optional[Callable]:
    """
    Get a panel renderer function.

    Args:
        panel_type: Type identifier for the panel

    Returns:
        Renderer function or None if not found
    """
    return _panel_renderers.get(panel_type)


def render_panel(
    panel_type: str, panel_config: Dict[str, Any], digital_twin_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Render a panel using the registered renderer.

    Args:
        panel_type: Type identifier for the panel
        panel_config: Panel configuration
        digital_twin_state: Current digital twin state

    Returns:
        Rendered panel data or error information
    """
    renderer = get_panel_renderer(panel_type)

    if renderer:
        try:
            return renderer(panel_config, digital_twin_state)
        except Exception as e:
            return {"error": f"Error rendering panel: {str(e)}", "config": panel_config}
    else:
        return {"error": f"Unknown panel type: {panel_type}", "config": panel_config}
