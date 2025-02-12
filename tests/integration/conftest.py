# tests/integration/conftest.py
import pytest
from pathlib import Path
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.config.project_paths import project_paths


@pytest.fixture(scope="module")
def test_analyzer():
    """Create manufacturing analyzer instance."""
    return SoliTekManufacturingAnalysis()


@pytest.fixture(scope="module")
def test_data():
    """Generate test manufacturing data."""
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=5)
    return {
        "production": generator.generate_production_data(),
        "quality": generator.generate_quality_data(),
        "energy": generator.generate_energy_data(),
        "material": generator.generate_material_flow_data(),
    }


@pytest.fixture(scope="module")
def test_run_dir():
    """Create and maintain test run directory."""
    run_dir = project_paths.get_run_directory()
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    return run_dir
