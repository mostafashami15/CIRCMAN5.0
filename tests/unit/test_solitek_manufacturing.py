import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.config.project_paths import project_paths


@pytest.fixture
def analyzer():
    """Create analyzer instance for testing."""
    return SoliTekManufacturingAnalysis()


def test_report_generation(analyzer):
    """Test that reports are generated in correct location."""
    analyzer.export_analysis_report()
    run_dir = project_paths.get_run_directory()
    assert (run_dir / "reports" / "analysis_report.xlsx").exists()


def test_default_paths(analyzer):
    """Test that default paths are used when none provided."""
    # Test production data loading
    test_data_path = Path(project_paths.get_path("SYNTHETIC_DATA"))
    test_data_path.mkdir(parents=True, exist_ok=True)

    # Create test CSV with output always less than input
    test_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5),
            "batch_id": [f"BATCH_{i}" for i in range(5)],
            "stage": ["test"] * 5,
            "input_amount": np.random.uniform(90, 110, 5),
            # Ensure output is always less than input
            "output_amount": np.random.uniform(80, 90, 5),
            "energy_used": np.random.uniform(140, 160, 5),
        }
    )

    test_file = test_data_path / "test_production_data.csv"
    test_df.to_csv(test_file, index=False)

    # Test loading
    analyzer.load_production_data()
    assert not analyzer.production_data.empty
