# src/core/constants.py

MANUFACTURING_STAGES = {
    'silicon_purification': {
        'input': 'raw_silicon',
        'output': 'purified_silicon',
        'expected_yield': 0.90
    },
    'wafer_production': {
        'input': 'purified_silicon',
        'output': 'silicon_wafer',
        'expected_yield': 0.95
    },
    'cell_production': {
        'input': 'silicon_wafer',
        'output': 'solar_cell',
        'expected_yield': 0.98
    }
}

QUALITY_THRESHOLDS = {
    'min_efficiency': 18.0,
    'max_defect_rate': 5.0,
    'min_thickness_uniformity': 90.0,
    'max_contamination_level': 1.0
}

OPTIMIZATION_TARGETS = {
    'min_yield_rate': 92.0,
    'min_energy_efficiency': 0.7,
    'min_water_reuse': 80.0,
    'min_recycled_content': 30.0
}