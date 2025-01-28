import pytest
from datetime import datetime
from circman5.monitoring import ManufacturingMonitor


@pytest.fixture
def monitor():
    """Create a monitoring instance for testing."""
    return ManufacturingMonitor()


def test_batch_monitoring(monitor):
    """Test basic batch monitoring functionality."""
    batch_id = "TEST_001"
    monitor.start_batch_monitoring(batch_id)

    # Record efficiency metrics
    efficiency_metrics = monitor.record_efficiency_metrics(
        output_quantity=100.0, cycle_time=60.0, energy_consumption=500.0
    )

    assert efficiency_metrics["batch_id"] == batch_id
    assert efficiency_metrics["output_quantity"] == 100.0
    assert efficiency_metrics["production_rate"] == pytest.approx(100.0 / 60.0)


def test_quality_metrics(monitor):
    """Test quality metrics recording."""
    batch_id = "TEST_001"
    monitor.start_batch_monitoring(batch_id)

    quality_metrics = monitor.record_quality_metrics(
        defect_rate=2.5, yield_rate=97.5, uniformity_score=95.0
    )

    assert quality_metrics["batch_id"] == batch_id
    assert quality_metrics["defect_rate"] == 2.5
    assert quality_metrics["yield_rate"] == 97.5


def test_resource_metrics(monitor):
    """Test resource utilization metrics."""
    batch_id = "TEST_001"
    monitor.start_batch_monitoring(batch_id)

    resource_metrics = monitor.record_resource_metrics(
        material_consumption=1000.0, water_usage=500.0, waste_generated=50.0
    )

    assert resource_metrics["batch_id"] == batch_id
    assert resource_metrics["resource_efficiency"] == pytest.approx(
        0.95
    )  # (1000-50)/1000


def test_batch_summary(monitor):
    """Test batch summary generation."""
    batch_id = "TEST_001"
    monitor.start_batch_monitoring(batch_id)

    # Record some metrics
    monitor.record_efficiency_metrics(100.0, 60.0, 500.0)
    monitor.record_quality_metrics(2.5, 97.5, 95.0)
    monitor.record_resource_metrics(1000.0, 500.0, 50.0)

    summary = monitor.get_batch_summary(batch_id)

    assert "efficiency" in summary
    assert "quality" in summary
    assert "resources" in summary
    assert summary["quality"]["avg_defect_rate"] == 2.5
