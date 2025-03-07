# tests/performance/test_hmi_performance.py

import pytest
import time
import statistics
from circman5.manufacturing.human_interface.core.dashboard_manager import (
    dashboard_manager,
)


def test_dashboard_rendering_performance(setup_test_environment):
    """Test the performance of dashboard rendering."""
    # Number of rendering iterations
    iterations = 50
    render_times = []

    # Run multiple rendering cycles
    for _ in range(iterations):
        start_time = time.time()
        dashboard_data = dashboard_manager.render_dashboard("main_dashboard")
        end_time = time.time()

        render_times.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate statistics
    avg_render_time = statistics.mean(render_times)
    max_render_time = max(render_times)
    min_render_time = min(render_times)
    percentile_95 = statistics.quantiles(render_times, n=20)[18]  # 95th percentile

    # Log results
    print(f"Dashboard Rendering Performance:")
    print(f"  Average: {avg_render_time:.2f} ms")
    print(f"  Min: {min_render_time:.2f} ms")
    print(f"  Max: {max_render_time:.2f} ms")
    print(f"  95th percentile: {percentile_95:.2f} ms")

    # Assert performance requirements
    assert avg_render_time < 200.0, "Average rendering time exceeds 200ms threshold"
    assert (
        percentile_95 < 300.0
    ), "95th percentile rendering time exceeds 300ms threshold"
