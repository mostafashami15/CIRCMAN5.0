# src/circman5/manufacturing/lifecycle/impact_factors.py
"""
Standard impact factors for PV manufacturing LCA calculations.
Values based on published literature and industry standards.
"""

from typing import Dict

# Manufacturing phase impact factors (kg CO2-eq per kg material)
MATERIAL_IMPACT_FACTORS: Dict[str, float] = {
    # Silicon-based materials
    "silicon_wafer": 32.8,
    "polysilicon": 45.2,
    "metallization_paste": 23.4,
    # Glass and encapsulation
    "solar_glass": 0.9,
    "tempered_glass": 1.2,
    "eva_sheet": 2.6,
    "backsheet": 4.8,
    # Frame and mounting
    "aluminum_frame": 8.9,
    "mounting_structure": 2.7,
    # Electronics
    "junction_box": 7.5,
    "copper_wiring": 3.2,
}

# Energy source impact factors (kg CO2-eq per kWh)
ENERGY_IMPACT_FACTORS: Dict[str, float] = {
    "grid_electricity": 0.5,
    "natural_gas": 0.2,
    "solar_pv": 0.0,
    "wind": 0.0,
}

# Transport impact factors (kg CO2-eq per tonne-km)
TRANSPORT_IMPACT_FACTORS: Dict[str, float] = {
    "road": 0.062,
    "rail": 0.022,
    "sea": 0.008,
    "air": 0.602,
}

# Recycling benefit factors (kg CO2-eq avoided per kg recycled)
RECYCLING_BENEFIT_FACTORS: Dict[str, float] = {
    "silicon": -28.4,
    "glass": -0.7,
    "aluminum": -8.1,
    "copper": -2.8,
    "plastic": -1.8,
}

# Manufacturing process impact factors (kg CO2-eq per process unit)
PROCESS_IMPACT_FACTORS: Dict[str, float] = {
    "wafer_cutting": 0.8,
    "cell_processing": 1.2,
    "module_assembly": 0.5,
    "testing": 0.1,
}

# Use phase factors
GRID_CARBON_INTENSITIES: Dict[str, float] = {
    "eu_average": 0.275,
    "us_average": 0.417,
    "china": 0.555,
    "india": 0.708,
}

# Performance degradation factors (% per year)
DEGRADATION_RATES: Dict[str, float] = {
    "mono_perc": 0.5,
    "poly_bsf": 0.6,
    "thin_film": 0.7,
    "bifacial": 0.45,
}
