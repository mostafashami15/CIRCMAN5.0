# src/circman5/adapters/config/impact_factors.py

from pathlib import Path
from typing import Dict, Any, Optional
import json

from ..base.adapter_base import ConfigAdapterBase


class ImpactFactorsAdapter(ConfigAdapterBase):
    """Adapter for Life Cycle Assessment (LCA) impact factors configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize impact factors adapter.

        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.config_path = (
            config_path or Path(__file__).parent / "json" / "impact_factors.json"
        )

    def load_config(self) -> Dict[str, Any]:
        """
        Load impact factors configuration.

        Returns:
            Dict[str, Any]: Impact factors configuration

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        if not self.config_path.exists():
            self.logger.warning(
                f"Config file not found: {self.config_path}, using defaults"
            )
            return self.get_defaults()

        return self._load_json_config(self.config_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate impact factors configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """

        # Log the actual config keys for debugging
        self.logger.info(f"Validating config with keys: {config.keys()}")

        required_factors = {
            "MATERIAL_IMPACT_FACTORS",
            "ENERGY_IMPACT_FACTORS",
            "RECYCLING_BENEFIT_FACTORS",
            "TRANSPORT_IMPACT_FACTORS",
            "PROCESS_IMPACT_FACTORS",
            "GRID_CARBON_INTENSITIES",
            "DEGRADATION_RATES",
            "SUSTAINABILITY_WEIGHTS",
            "QUALITY_WEIGHTS",
            "CARBON_INTENSITY_FACTORS",
            "MONITORING_WEIGHTS",
            "WATER_FACTOR",
            "WASTE_FACTOR",
        }

        # Check required top-level keys
        if not all(key in config for key in required_factors):
            self.logger.error(
                f"Missing required factors: {required_factors - set(config.keys())}"
            )
            return False

        # Validate material impact factors
        material_factors = config.get("MATERIAL_IMPACT_FACTORS", {})
        if not material_factors or not all(
            isinstance(v, (int, float)) for v in material_factors.values()
        ):
            self.logger.error("Invalid material impact factors")
            return False

        # Validate energy impact factors
        energy_factors = config.get("ENERGY_IMPACT_FACTORS", {})
        required_energy_sources = {
            "grid_electricity",
            "natural_gas",
            "solar_pv",
            "wind",
        }
        if not all(source in energy_factors for source in required_energy_sources):
            self.logger.error("Missing required energy sources")
            return False

        # Validate recycling benefit factors
        recycling_factors = config.get("RECYCLING_BENEFIT_FACTORS", {})
        required_materials = {"silicon", "glass", "aluminum", "copper", "plastic"}
        if not all(material in recycling_factors for material in required_materials):
            self.logger.error("Missing required recycling materials")
            return False

        return True

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default impact factors configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "MATERIAL_IMPACT_FACTORS": {
                "silicon_wafer": 32.8,
                "polysilicon": 45.2,
                "metallization_paste": 23.4,
                "solar_glass": 0.9,
                "tempered_glass": 1.2,
                "eva_sheet": 2.6,
                "backsheet": 4.8,
                "aluminum_frame": 8.9,
                "mounting_structure": 2.7,
                "junction_box": 7.5,
                "copper_wiring": 3.2,
            },
            "ENERGY_IMPACT_FACTORS": {
                "grid_electricity": 0.5,
                "natural_gas": 0.2,
                "solar_pv": 0.0,
                "wind": 0.0,
            },
            "TRANSPORT_IMPACT_FACTORS": {
                "road": 0.062,
                "rail": 0.022,
                "sea": 0.008,
                "air": 0.602,
            },
            "RECYCLING_BENEFIT_FACTORS": {
                "silicon": -28.4,
                "glass": -0.7,
                "aluminum": -8.1,
                "copper": -2.8,
                "plastic": -1.8,
                "silicon_wafer": 0.7,
                "solar_glass": 0.8,
                "aluminum_frame": 0.9,
                "copper_wire": 0.85,
            },
            "PROCESS_IMPACT_FACTORS": {
                "wafer_cutting": 0.8,
                "cell_processing": 1.2,
                "module_assembly": 0.5,
                "testing": 0.1,
            },
            "GRID_CARBON_INTENSITIES": {
                "eu_average": 0.275,
                "us_average": 0.417,
                "china": 0.555,
                "india": 0.708,
            },
            "DEGRADATION_RATES": {
                "mono_perc": 0.5,
                "poly_bsf": 0.6,
                "thin_film": 0.7,
                "bifacial": 0.45,
            },
            "CARBON_INTENSITY_FACTORS": {
                "grid": 0.5,
                "solar": 0.0,
                "wind": 0.0,
                "electricity": 0.5,
                "natural_gas": 0.2,
                "petroleum": 0.25,
            },
            "SUSTAINABILITY_WEIGHTS": {
                "material": 0.4,
                "recycling": 0.3,
                "energy": 0.3,
                "material_efficiency": 0.4,
                "carbon_footprint": 0.4,
                "energy_efficiency": 0.3,
                "recycling_rate": 0.3,
            },
            "QUALITY_WEIGHTS": {
                "defect": 0.4,
                "efficiency": 0.4,
                "uniformity": 0.2,
                "defect_rate": 0.4,
                "efficiency_score": 0.4,
                "uniformity_score": 0.2,
            },
            "MONITORING_WEIGHTS": {"defect": 0.4, "yield": 0.4, "uniformity": 0.2},
            "WATER_FACTOR": 0.1,
            "WASTE_FACTOR": 0.05,
        }
