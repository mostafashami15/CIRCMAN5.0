# tests/integration/conftest.py

import pytest
from pathlib import Path
from circman5.utils.results_manager import results_manager
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture(scope="session")
def test_data():
    """Generate test manufacturing data."""
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=5)
    return {
        "production": generator.generate_production_data(),
        "quality": generator.generate_quality_data(),
        "energy": generator.generate_energy_data(),
        "material": generator.generate_material_flow_data(),
        "lca": generator.generate_complete_lca_dataset(),
    }


@pytest.fixture(scope="session")
def test_analyzer():
    """Create manufacturing analyzer instance."""
    return SoliTekManufacturingAnalysis()


@pytest.fixture(scope="session")
def test_paths():
    """Get standardized test paths."""
    return {
        "reports": results_manager.get_path("reports"),
        "visualizations": results_manager.get_path("visualizations"),
        "metrics": results_manager.get_path("metrics"),
        "lca_results": results_manager.get_path("lca_results"),
        "input_data": results_manager.get_path("input_data"),
    }


@pytest.fixture(scope="session")
def analyzer_with_data(test_analyzer, test_data):
    """Create analyzer instance with test data."""
    test_analyzer.production_data = test_data["production"]
    test_analyzer.quality_data = test_data["quality"]
    test_analyzer.energy_data = test_data["energy"]
    test_analyzer.material_flow = test_data["material"]
    test_analyzer.lca_data = {
        "material_flow": test_data["material"],
        "energy_consumption": test_data["energy"],
        "lca": test_data["lca"],
    }
    return test_analyzer


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_outputs():
    """Clean up test outputs after session."""
    yield
    results_manager.cleanup_old_runs(keep_last=3)


@pytest.fixture(scope="function")
def temp_test_dir():
    """Provide temporary directory for test files."""
    temp_path = results_manager.get_path("temp")
    yield temp_path
    # Clean temporary files after each test
    for file in temp_path.glob("*"):
        if file.is_file():
            file.unlink()
