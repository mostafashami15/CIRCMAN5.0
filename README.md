# CIRCMAN5.0
**Human-Centered AI-aided Framework for the Photovoltaic (PV) Manufacturing Industry**

## Overview
CIRCMAN5.0 integrates **AI-driven analytics** with **circular manufacturing principles** to optimize **PV production**, reduce waste, and enhance sustainability through comprehensive lifecycle assessment.

## Project Goals
- **AI-driven modeling** for circular manufacturing.
- **Process optimization** using **machine learning**.
- **Real-time monitoring** and **control systems**.
- **Circular economy** principles for **waste reduction**.
- **Digital Product/Material Passport** implementation.
- **Life Cycle Assessment (LCA)** for environmental impact.

## Features

### **Current Implementation**
✅ **Manufacturing Process Tracking**
- Batch processing and monitoring.
- Resource utilization tracking.
- Production efficiency analysis.
- Real-time data processing (r2: 0.99).

✅ **Quality Control System**
- Real-time quality monitoring.
- Defect detection and analysis.
- Performance metrics calculation.
- AI-driven quality assessment.

✅ **Circularity Metrics**
- Material efficiency tracking.
- Water reuse monitoring.
- Waste reduction analysis.
- Environmental impact tracking.

✅ **Advanced Analytics**
- Performance visualization.
- Trend analysis.
- Optimization recommendations.
- AI-powered predictions.

### **In Progress**
🔄 **AI/ML Integration**
- Basic predictive analytics implemented.
- Initial process optimization framework.
- Synthetic data validation complete.
- Real data integration pending.

🔄 **LCA Development**
- Environmental impact assessment.
- Resource impact calculations.
- Lifecycle phase tracking.
- Carbon footprint analysis.

### **Planned Features**
🚀 **Digital Twin Development**
- Real-time simulation for PV manufacturing.
- Process modeling for optimization.
- Scenario-based **What-if analysis**.

🚀 **Enhanced Circularity**
- Advanced Life Cycle Assessment (LCA).
- Digital Product Passport implementation.
- Optimized resource management.

## Project Structure
```plaintext
circman5-0-py3.11(base) mostafashami@Mostafas-MacBook-Pro CIRCMAN5.0 % tree
.
├── LICENSE
├── README.md
├── data
│   ├── processed
│   ├── raw
│   └── synthetic
│       ├── test_energy_data.csv
│       ├── test_material_data.csv
│       ├── test_production_data.csv
│       └── test_quality_data.csv
├── demo_script.py
├── docs
│   ├── api
│   │   └── README.md
│   ├── diagrams
│   │   └── architecture.md
│   └── guides
│       ├── development_roadmap.md
│       ├── implementation_details.md
│       ├── system_analysis.md
│       └── system_documentation.md
├── logs
│   ├── manufacturing_20250128_103459.log
│   ├── manufacturing_20250128_103735.log
│   ├── manufacturing_20250128_104132.log
│   ├── manufacturing_20250128_115153.log
│   ├── manufacturing_20250129_230409.log
│   ├── manufacturing_20250203_230717.log
│   ├── manufacturing_20250203_231150.log
│   ├── manufacturing_20250203_231724.log
│   ├── manufacturing_20250203_232237.log
│   ├── manufacturing_20250203_232702.log
│   ├── manufacturing_20250203_233936.log
│   ├── manufacturing_20250204_084124.log
│   ├── manufacturing_20250204_084359.log
│   ├── manufacturing_20250204_085438.log
│   ├── manufacturing_20250204_111135.log
│   ├── manufacturing_20250204_111932.log
│   ├── manufacturing_20250204_131207.log
│   ├── manufacturing_20250204_131547.log
│   ├── manufacturing_20250204_133800.log
│   ├── manufacturing_20250204_135605.log
│   ├── manufacturing_20250204_135727.log
│   ├── manufacturing_20250204_140122.log
│   ├── manufacturing_20250204_140526.log
│   ├── manufacturing_20250204_170632.log
│   ├── manufacturing_20250204_170902.log
│   └── manufacturing_20250204_171152.log
├── notebooks
│   └── requirements.txt
├── poetry.lock
├── pyproject.toml
├── pyrightconfig.json
├── pytest.ini
├── setup.py
├── src
│   ├── __init__.py
│   ├── circman5
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── errors.cpython-311.pyc
│   │   │   ├── logging_config.cpython-311.pyc
│   │   │   ├── manufacturing.cpython-311.pyc
│   │   │   ├── monitoring.cpython-311.pyc
│   │   │   ├── solitek_manufacturing.cpython-311.pyc
│   │   │   ├── test_data_generator.cpython-311-pytest-7.4.4.pyc
│   │   │   ├── test_data_generator.cpython-311.pyc
│   │   │   └── visualization.cpython-311.pyc
│   │   ├── ai
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   └── optimization.cpython-311.pyc
│   │   │   └── optimization.py
│   │   ├── analysis
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── efficiency.cpython-311.pyc
│   │   │   │   ├── quality.cpython-311.pyc
│   │   │   │   └── sustainability.cpython-311.pyc
│   │   │   ├── efficiency.py
│   │   │   ├── quality.py
│   │   │   └── sustainability.py
│   │   ├── constants.py
│   │   ├── data_types.py
│   │   ├── errors.py
│   │   ├── logging_config.py
│   │   ├── manufacturing.py
│   │   ├── monitoring.py
│   │   ├── solitek_manufacturing.py
│   │   ├── test_data_generator.py
│   │   ├── test_framework.py
│   │   └── visualization.py
│   └── utils
│       └── __init__.py
└── tests
    ├── README.md
    ├── __init__.py
    ├── ai
    │   ├── __init__.py
    │   └── test_optimization.py
    ├── integration
    │   ├── __pycache__
    │   │   └── test_system_integration.cpython-311-pytest-7.4.4.pyc
    │   ├── system
    │   │   └── test_script.py
    │   ├── test_data_pipeline.py
    │   ├── test_manufacturing_optimization.py
    │   └── test_system_integration.py
    ├── performance
    │   ├── __pycache__
    │   │   └── test_performance.cpython-311-pytest-7.4.4.pyc
    │   └── test_performance.py
    ├── results
    │   ├── latest
    │   │   ├── analysis_report.xlsx
    │   │   ├── energy_analysis.png
    │   │   ├── production_analysis.png
    │   │   ├── quality_analysis.png
    │   │   └── sustainability_analysis.png
    │   └── visualizations
    └── unit
        ├── __pycache__
        │   ├── test_efficiency_analyzer.cpython-311-pytest-7.4.4.pyc
        │   ├── test_manufacturing.cpython-311-pytest-7.4.4.pyc
        │   ├── test_monitoring.cpython-311-pytest-7.4.4.pyc
        │   ├── test_production_data.cpython-311-pytest-7.4.4.pyc
        │   ├── test_quality_analyzer.cpython-311-pytest-7.4.4.pyc
        │   ├── test_sustainability_analyzer.cpython-311-pytest-7.4.4.pyc
        │   └── test_visualization.cpython-311-pytest-7.4.4.pyc
        ├── test_efficiency_analyzer.py
        ├── test_manufacturing.py
        ├── test_monitoring.py
        ├── test_production_data.py
        ├── test_quality_analyzer.py
        ├── test_sustainability_analyzer.py
        └── test_visualization.py

31 directories, 109 files
```
## Technology Stack
- **Python 3.11+**
- **Pandas** for data processing
- **Matplotlib & Seaborn** for visualization
- **Scikit-learn** for machine learning
- **Poetry** for package management
- **Pytest** for testing framework
- **Openpyxl** for report generation

## Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/mostafashami15/CIRCMAN5.0.git
cd CIRCMAN5.0
```

### **2️⃣ Install Dependencies Using Poetry**
```bash
poetry install
```

### **3️⃣ Activate the Virtual Environment**
```bash
poetry shell
```

### **4️⃣ Run Tests**
```bash
poetry run pytest tests/
```

### **5️⃣ Start Using the System**
```python
from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis

# Initialize analyzer
analyzer = SoliTekManufacturingAnalysis()

# Generate analysis with synthetic data
analyzer.generate_comprehensive_report("analysis_report.xlsx")
```

## Documentation**
For detailed documentation and API reference, see the docs/ folder or visit:
🔗 GitHub Wiki (under constructuion)

## Contributing
1. Fork the repository.
2. Create a new branch (feature-branch).
3. Commit your changes.
4. Push to your branch and create a PR.

## License
This project is licensed under the **MIT** License. See **LICENSE** for details.
