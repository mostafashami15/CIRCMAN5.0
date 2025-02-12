"""
Test suite for LCA integration in SoliTekManufacturingAnalysis.
Tests all aspects of lifecycle assessment calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.circman5.test_data_generator import ManufacturingDataGenerator
from circman5.manufacturing.core import SoliTekManufacturingAnalysis


@pytest.fixture
def test_data_generator():
    """Create a data generator for testing."""
    return ManufacturingDataGenerator(
        start_date="2024-01-01", days=5  # Small dataset for faster testing
    )


@pytest.fixture
def analyzer():
    """Create SoliTekManufacturingAnalysis instance for testing."""
    return SoliTekManufacturingAnalysis()


def test_lca_data_loading(analyzer, test_data_generator):
    """Test loading and validation of LCA data."""
    # Generate test data
    test_data = test_data_generator.generate_complete_lca_dataset()

    # Save test data to temporary CSV files
    test_data["material_flow"].to_csv("test_material_data.csv", index=False)
    test_data["energy_consumption"].to_csv("test_energy_data.csv", index=False)
    test_data["process_data"].to_csv("test_process_data.csv", index=False)

    # Test loading data
    analyzer.load_lca_data(
        material_data_path="test_material_data.csv",
        energy_data_path="test_energy_data.csv",
        process_data_path="test_process_data.csv",
    )

    # Verify data was loaded correctly
    assert not analyzer.lca_data["material_flow"].empty
    assert not analyzer.lca_data["energy_consumption"].empty
    assert not analyzer.lca_data["process_data"].empty


def test_recycling_rates_calculation(analyzer, test_data_generator):
    """Test calculation of recycling rates from material data."""
    # Generate test data
    material_data = test_data_generator.generate_lca_material_data()

    # Calculate recycling rates
    recycling_rates = analyzer._calculate_recycling_rates(material_data)

    # Verify results
    assert isinstance(recycling_rates, dict)
    assert all(0 <= rate <= 1 for rate in recycling_rates.values())


def test_lifecycle_assessment(analyzer, test_data_generator):
    """Test full lifecycle assessment calculation."""
    # Generate and load test data
    test_data = test_data_generator.generate_complete_lca_dataset()
    analyzer.lca_data = test_data

    # Perform lifecycle assessment
    impact = analyzer.perform_lifecycle_assessment()

    # Verify results
    assert hasattr(impact, "manufacturing_impact")
    assert hasattr(impact, "use_phase_impact")
    assert hasattr(impact, "end_of_life_impact")
    assert hasattr(impact, "total_carbon_footprint")


def test_lca_report_generation(analyzer, test_data_generator, tmp_path):
    """Test generation of LCA report."""
    # Generate and load test data
    test_data = test_data_generator.generate_complete_lca_dataset()
    analyzer.lca_data = test_data

    # Generate report
    report_path = tmp_path / "lca_report.xlsx"
    analyzer.generate_lca_report(str(report_path))

    # Verify report was created
    assert report_path.exists()

    # Load and verify report content
    xlsx = pd.ExcelFile(report_path)
    assert "Manufacturing Impact" in xlsx.sheet_names
    assert "Use Phase" in xlsx.sheet_names
    assert "End of Life" in xlsx.sheet_names


def test_material_impacts_calculation(analyzer, test_data_generator):
    """Test calculation of material production impacts."""
    # Generate test data
    test_data = test_data_generator.generate_complete_lca_dataset()
    analyzer.lca_data = test_data

    # Calculate material impacts
    impact = analyzer._calculate_material_impacts()

    # Verify results
    assert isinstance(impact, float)
    assert impact >= 0


def test_energy_impacts_calculation(analyzer, test_data_generator):
    """Test calculation of energy consumption impacts."""
    # Generate test data
    test_data = test_data_generator.generate_complete_lca_dataset()
    analyzer.lca_data = test_data

    # Calculate energy impacts
    impact = analyzer._calculate_energy_impacts()

    # Verify results
    assert isinstance(impact, float)
    assert impact >= 0


def test_process_impacts_calculation(analyzer, test_data_generator):
    """Test calculation of manufacturing process impacts."""
    # Generate test data
    test_data = test_data_generator.generate_complete_lca_dataset()
    analyzer.lca_data = test_data

    # Calculate process impacts
    impact = analyzer._calculate_process_impacts()

    # Verify results
    assert isinstance(impact, float)
    assert impact >= 0


def test_batch_specific_assessment(analyzer, test_data_generator):
    """Test lifecycle assessment for specific batch."""
    # Generate test data
    test_data = test_data_generator.generate_complete_lca_dataset()
    analyzer.lca_data = test_data

    # Get a test batch ID
    test_batch = analyzer.lca_data["material_flow"]["batch_id"].iloc[0]

    # Perform assessment for specific batch
    impact = analyzer.perform_lifecycle_assessment(batch_id=test_batch)

    # Verify results
    assert hasattr(impact, "manufacturing_impact")
    assert hasattr(impact, "use_phase_impact")
    assert hasattr(impact, "end_of_life_impact")
    assert hasattr(impact, "total_carbon_footprint")


def test_error_handling(analyzer):
    """Test error handling for invalid data scenarios."""
    # Test with empty data
    empty_df = pd.DataFrame()

    # Verify empty data handling
    assert analyzer._calculate_recycling_rates(empty_df) == {}
    assert analyzer._calculate_material_impacts() == 0.0
    assert analyzer._calculate_energy_impacts() == 0.0
    assert analyzer._calculate_process_impacts() == 0.0
