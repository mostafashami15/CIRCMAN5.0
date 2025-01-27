# CIRCMAN5.0
**Human-Centered AI-aided Framework for the Photovoltaic (PV) Manufacturing Industry**

## Overview
CIRCMAN5.0 integrates **AI-driven analytics** with **circular manufacturing principles** to optimize **PV production**, reduce waste, and enhance sustainability.

## Project Goals
- **AI-driven modeling** for circular manufacturing.
- **Process optimization** using **machine learning**.
- **Real-time monitoring** and **control systems**.
- **Circular economy** principles for **waste reduction**.
- **Digital Product/Material Passport** implementation.

## Features

### **Current Implementation**
✅ **Manufacturing Process Tracking**
- Batch processing and monitoring.
- Resource utilization tracking.
- Production efficiency analysis.

✅ **Quality Control System**
- Real-time quality monitoring.
- Defect detection and analysis.
- Performance metrics calculation.

✅ **Circularity Metrics**
- Material efficiency tracking.
- Water reuse monitoring.
- Waste reduction analysis.

✅ **Advanced Analytics**
- Performance visualization.
- Trend analysis.
- Optimization recommendations.

### **Planned Features**
🚀 **AI/ML Integration**
- Predictive analytics for manufacturing processes.
- Process optimization using reinforcement learning.
- Automated decision support.

🚀 **Digital Twin Development**
- Real-time simulation for PV manufacturing.
- Process modeling for optimization.
- Scenario-based **What-if analysis**.

🚀 **Enhanced Circularity**
- Life Cycle Assessment (LCA).
- Digital Product Passport implementation.
- Optimized resource management.

## Project Structure
.
├── LICENSE
├── README.md
├── data
│   ├── processed
│   └── raw
├── docs
│   ├── development_roadmap.md
│   ├── system_analysis.md
│   └── system_documentation.md
├── notebooks
│   └── requirements.txt
├── poetry.lock
├── pyproject.toml
├── pytest.ini
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── analysis
│   │   └── __init__.py
│   ├── circman5
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── manufacturing.cpython-311.pyc
│   │   ├── constants.py
│   │   ├── data_types.py
│   │   ├── manufacturing.py
│   │   ├── solitek_manufacturing.py
│   │   ├── test_data_generator.py
│   │   └── test_framework.py
│   ├── documentation
│   │   ├── __init__.py
│   │   ├── conclusions.py
│   │   ├── gap_analysis.py
│   │   ├── literature_review.py
│   │   └── spi_framework.py
│   └── utils
│       └── __init__.py
└── tests
    ├── README.md
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-311.pyc
    ├── integration
    ├── results
    │   ├── data
    │   ├── reports
    │   ├── run_20241127_172331
    │   │   ├── analysis_report.xlsx
    │   │   ├── input_data
    │   │   │   ├── test_energy_data.csv
    │   │   │   ├── test_material_data.csv
    │   │   │   ├── test_production_data.csv
    │   │   │   └── test_quality_data.csv
    │   │   ├── test_log.txt
    │   │   └── visualizations
    │   │       ├── energy_analysis.png
    │   │       ├── production_analysis.png
    │   │       ├── quality_analysis.png
    │   │       └── sustainability_analysis.png
    │   ├── test_results
    │   │   ├── reports
    │   │   └── visualizations
    │   └── visualizations
    └── unit
        ├── __pycache__
        │   ├── test_manufacturing.cpython-311-pytest-7.4.4.pyc
        │   └── test_manufacturing.cpython-311-pytest-8.3.4.pyc
        └── test_manufacturing.py


## Technology Stack
- **Python 3.11+**
- **Pandas** for data processing.
- **Matplotlib & Seaborn** for visualization.
- **Scikit-learn** for machine learning.
- **Poetry** for package management.

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
from circman5.manufacturing import AdvancedPVManufacturing

factory = AdvancedPVManufacturing()
factory.start_batch("TEST_001", "silicon_purification", 100)
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
