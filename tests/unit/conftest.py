import pytest


@pytest.fixture(autouse=True)
def reset_constants_service():
    """Reset ConstantsService singleton before and after each test."""
    from circman5.adapters.services.constants_service import ConstantsService

    ConstantsService._reset_instance()
    yield
    ConstantsService._reset_instance()


@pytest.fixture
def mock_constants(mocker):
    """Create a mock constants service with proper config values."""
    from circman5.adapters.services.constants_service import ConstantsService

    # Reset singleton first
    ConstantsService._reset_instance()

    constants_mock = mocker.Mock()

    # Define the mock data with all required keys
    mock_config = {
        "GRID_CARBON_INTENSITIES": {"eu_average": 0.275, "us_average": 0.417},
        "QUALITY_WEIGHTS": {
            "defect": 0.4,
            "efficiency": 0.4,
            "uniformity": 0.2,
            "defect_rate": 0.4,
            "efficiency_score": 0.4,
            "uniformity_score": 0.2,
        },
        "SUSTAINABILITY_WEIGHTS": {
            "carbon_footprint": 0.4,
            "material_efficiency": 0.4,
            "energy_efficiency": 0.3,
            "recycling_rate": 0.3,
        },
        "RECYCLING_BENEFIT_FACTORS": {
            "silicon_wafer": 0.7,
            "solar_glass": 0.8,
            "aluminum_frame": 0.9,
        },
        "MATERIAL_IMPACT_FACTORS": {
            "silicon_wafer": 32.8,
            "solar_glass": 0.9,
            "backsheet": 4.8,
            "aluminum_frame": 8.9,
        },
        "CARBON_INTENSITY_FACTORS": {"electricity": 0.5, "natural_gas": 0.2},
    }

    # Better implementation of get_constant that matches the real behavior
    def get_constant_side_effect(adapter, key):
        if key in mock_config:
            return mock_config[key]
        raise KeyError(f"Key not found in {adapter} config: {key}")

    constants_mock.get_constant.side_effect = get_constant_side_effect
    constants_mock.get_impact_factors.return_value = mock_config

    # Patch the singleton
    mocker.patch.object(ConstantsService, "__new__", return_value=constants_mock)

    return constants_mock
