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
CIRCMAN5.0/
├── data/
│   ├── processed/
│   ├── raw/
│   └── synthetic/
│   └── synthetic
│       ├── test_energy_data.csv
│       ├── test_material_data.csv
│       ├── test_process_data.csv
│       └── test_production_data.csv
├── docs/
│   ├── api/
│   │   └── README.md (empty)
│   ├── diagrams/
│   │   └── architecture.md (empty)
│   └── guides/
│       ├── development_roadmap.md
│       ├── implementation_details.md
│       ├── system_analysis.md
│       └── system_documentation.md
├── logs/
├── examples/
│   └── demo_script.py
├── notebooks/
│   └── requirements.txt
├── src/
│   └── utils/
│   │   ├── cleanup.py
│   │   └── results_paths.py
│   └── circman5/
│       ├── ai/
│       │   ├── optimization_base.py
│       │   ├── optimization_core.py
│       │   ├── optimization_prediction.py
│       │   ├── optimization_training.py
│       │   └── optimization_types.py
│       ├── analysis/
│       │   └── lca/
│       │   │   ├── core.py
│       │   │   └── impact_factors.py
│       │   ├── efficiency.py
│       │   ├── quality.py
│       │   └── sustainability.py
│       ├── config/
│       │   └── project_paths.py
│       ├── visualization/
│       │   ├── lca_visualizer.py
│       │   └── manufacturing_visualizer.py
│       ├── constants.py
│       ├── data_types.py
│       ├── errors.py
│       ├── logging_config.py
│       ├── monitoring.py
│       ├── solitek_manufacturing.py
│       ├── test_data_generator.py
│       └── test_framework.py
├── tests/
│   ├── ai/
│   │   └── test_optimization.py
│   ├── integration/
│   │   ├── test_data_pipeline.py (empty)
│   │   ├── test_data_saving.py
│   │   ├── test_manufacturing_optimization.py
│   │   └── test_system_integration.py
│   ├── performance/
│   │   └── test_performance.py
│   ├── results/
│   │   ├── archive/
│   │   ├── latest/
│   │   └── runs/
│   └── unit/
│   │   └── test_lca_core.py
│   ├── test_data_generator.py
│   ├── test_data_generator.py
│   ├── test_efficiency_analyzer.py
│   ├── test_lca_data_generator.py
│   ├── test_lca_integration.py
│   ├── test_lca_visualization.py
│   ├── test_logging_config.py
│   ├── test_manufacturing.py
│   ├── test_monitoring.py
│   ├── test_production_data.py
│   ├── test_project_paths.py
│   ├── test_quality_analyzer.py
│   ├── test_solitek_manufacturing.py
│   ├── test_sustainability_analyzer.py
│   ├──test_project_imports.py
│   └── test_visualization.py
├── poetry.lock
├── pyproject.toml
├── pyrightconfig.json
├── pytest.ini
├── LICENSE
├── README.md
├── setup.py
├── .pre-commit-config.yaml
├── .env
└── .gitignore

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
