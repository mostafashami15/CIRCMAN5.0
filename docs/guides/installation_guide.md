# CIRCMAN5.0 Installation Guide

## 1. Introduction

This guide provides detailed instructions for installing CIRCMAN5.0, a human-centered AI-aided framework for PV manufacturing. It covers different installation methods, system requirements, environment setup, and troubleshooting common installation issues.

CIRCMAN5.0 is a Python-based system that includes components for manufacturing analytics, digital twin simulation, optimization, and lifecycle assessment. The installation process will set up all necessary dependencies and prepare the system for configuration and deployment.

## 2. System Requirements

### 2.1 Hardware Requirements

CIRCMAN5.0 has the following hardware recommendations:

| Component | Minimum | Recommended | Enterprise |
|-----------|---------|-------------|-----------|
| CPU | 4 cores, 2.5 GHz | 8 cores, 3.0+ GHz | 16+ cores, 3.5+ GHz |
| RAM | 8 GB | 16 GB | 32+ GB |
| Storage | 100 GB SSD | 500 GB SSD | 1+ TB SSD |
| Graphics | Integrated graphics | Dedicated GPU (for advanced visualization) | Dedicated GPU with 8+ GB VRAM |
| Network | 100 Mbps | 1 Gbps | 10+ Gbps |

### 2.2 Software Requirements

CIRCMAN5.0 has the following software requirements:

| Component | Requirement |
|-----------|-------------|
| Operating System | Linux (Ubuntu 20.04+, CentOS 8+), macOS 11+, Windows 10/11 |
| Python | Python 3.11+ |
| Database (Optional) | PostgreSQL 13+, MySQL 8+, or SQLite 3.30+ |
| Version Control | Git 2.20+ |
| Build Tools | C/C++ compiler, make |

## 3. Installation Methods

CIRCMAN5.0 can be installed using different methods, depending on your requirements and preferences.

### 3.1 Installation with Poetry (Recommended)

The recommended way to install CIRCMAN5.0 is using Poetry, which provides dependency management and virtual environment handling.

#### 3.1.1 Install Poetry

First, install Poetry if you don't have it already:

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Verify Poetry installation:

```bash
poetry --version
```

#### 3.1.2 Clone the Repository

Clone the CIRCMAN5.0 repository:

```bash
git clone https://github.com/example/circman5.git
cd circman5
```

#### 3.1.3 Install Dependencies

Install CIRCMAN5.0 and its dependencies:

```bash
poetry install
```

This will create a virtual environment and install all dependencies specified in the `pyproject.toml` file:

```toml
[tool.poetry]
name = "circman5"
version = "0.1.0"
description = "Human-Centered AI-aided Framework for PV Manufacturing"
authors = ["Mostafa Shami <mostafashami15@gmail.com>"]
packages = [
    { include = "circman5", from = "src" },
]

[tool.poetry.dependencies]
python = "^3.11"
numpy = "^2.2.2"
pandas = "^2.2.3"
matplotlib = "^3.5.0"
seaborn = "^0.12.0"
scikit-learn = "^1.6.1"
psutil = "^6.1.1"
openpyxl = "^3.1.5"

[tool.poetry.group.dev.dependencies]
pre-commit = "^4.1.0"
pytest = "^7.1.3"
pytest-html = "^4.1.1"
pytest-cov = "^6.0.0"
pytest-xdist = "^3.6.1"
pytest-mock = "^3.14.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 3.2 Installation with pip

Alternatively, you can install CIRCMAN5.0 using pip.

#### 3.2.1 Clone the Repository

Clone the CIRCMAN5.0 repository:

```bash
git clone https://github.com/example/circman5.git
cd circman5
```

#### 3.2.2 Create Virtual Environment

Create and activate a virtual environment:

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

#### 3.2.3 Install Dependencies

Install CIRCMAN5.0 and its dependencies:

```bash
pip install -e .
```

This will install the package in development mode, using the `setup.py` file:

```python
from setuptools import setup, find_packages

setup(
    name="circman5",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn>=1.6.1",
        "pandas>=2.0.0",
        "numpy>=1.21.0",
    ],
)
```

For additional dependencies, install them from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3.3 Installation from Source

For a manual installation from source:

#### 3.3.1 Clone the Repository

Clone the CIRCMAN5.0 repository:

```bash
git clone https://github.com/example/circman5.git
cd circman5
```

#### 3.3.2 Create Virtual Environment

Create and activate a virtual environment:

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

#### 3.3.3 Install Dependencies Manually

Install the core dependencies:

```bash
pip install pandas>=2.0.0 numpy>=2.2.2 matplotlib>=3.5.0 seaborn>=0.12.0 scikit-learn>=1.6.1 psutil>=6.1.1 openpyxl>=3.1.5
```

#### 3.3.4 Install the Package

Install the package in development mode:

```bash
pip install -e .
```

## 4. First-Time Setup

After installing CIRCMAN5.0, perform the first-time setup to prepare the system for use.

### 4.1 Initialize Project Structure

Run the project structure initialization script to set up the required directories and configurations:

```bash
# Using Poetry
poetry run python scripts/fix_project_structure.py

# Using pip or manual installation
python scripts/fix_project_structure.py
```

This script performs the following actions:

```python
def fix_project_structure():
    """Fix project structure to match desired organization."""

    src_root = Path("src/circman5")

    # Fix imports in manufacturing core
    core_file = src_root / "manufacturing/core.py"
    if core_file.exists():
        content = core_file.read_text()
        # Update relative imports to use absolute imports
        content = content.replace("from ..utils", "from circman5.utils")
        content = content.replace(
            "from .analyzers", "from circman5.manufacturing.analyzers"
        )
        core_file.write_text(content)

    # Clean up backup files
    backup_files = src_root.rglob("*.bak")
    for file in backup_files:
        file.unlink()

    # Remove circman5_backup if it exists
    backup_dir = Path("src/circman5_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    # Create required __init__.py files
    init_locations = [
        "src/circman5/manufacturing/analyzers",
        "src/circman5/manufacturing/reporting",
        "src/circman5/manufacturing/lifecycle",
        "src/circman5/manufacturing/optimization",
        "src/circman5/utils",
        "src/circman5/config",
    ]

    for loc in init_locations:
        init_file = Path(loc) / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch()
```

### 4.2 Create Data Directories

Create the necessary data directories:

```bash
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/synthetic
```

### 4.3 Configure Project Paths

The system uses a project paths configuration that is automatically managed by the Results Manager. The default configuration is in `src/circman5/config/project_paths.py`:

```python
# src/circman5/config/project_paths.py
from pathlib import Path
from ..utils.results_manager import results_manager


class ProjectPaths:
    """Legacy wrapper around ResultsManager."""

    def __init__(self):
        self.PROJECT_ROOT = results_manager.project_root

    def get_run_directory(self) -> Path:
        return results_manager.get_run_dir()

    def get_path(self, key: str) -> str:
        return str(results_manager.get_path(key))

    def get_synthetic_data_path(self, filename: str) -> str:
        synthetic_dir = results_manager.get_path("SYNTHETIC_DATA")
        return str(synthetic_dir / filename)


# Keep global instance for backward compatibility
project_paths = ProjectPaths()
```

## 5. Verification

After installation, verify that CIRCMAN5.0 is correctly installed and configured.

### 5.1 Run Tests

Run the tests to verify the installation:

```bash
# Using Poetry
poetry run pytest tests/unit

# Using pip or manual installation
pytest tests/unit
```

You should see output indicating that the tests have passed.

### 5.2 Run a Simple Example

Run a simple example to verify the system functionality:

```bash
# Using Poetry
poetry run python examples/demo_script.py

# Using pip or manual installation
python examples/demo_script.py
```

## 6. Environment-Specific Setup

### 6.1 Development Environment

For a development environment, additional tools can be useful:

```bash
# Using Poetry
poetry install --with dev

# Using pip
pip install -r dev-requirements.txt
```

Set up pre-commit hooks for code quality:

```bash
pre-commit install
```

### 6.2 Production Environment

For a production environment, optimize the installation:

```bash
# Using Poetry
poetry install --no-dev

# Using pip
pip install -e .
```

### 6.3 Containerized Environment

For a containerized environment, CIRCMAN5.0 can be installed using Docker. Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Poetry
RUN pip install poetry

# Install dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

# Initialize project structure
RUN python scripts/fix_project_structure.py

# Create required directories
RUN mkdir -p data/raw data/processed data/synthetic

CMD ["python", "-m", "circman5"]
```

Build and run the Docker container:

```bash
docker build -t circman5 .
docker run -p 8080:8080 circman5
```

## 7. Troubleshooting

### 7.1 Common Installation Issues

#### 7.1.1 Dependency Conflicts

Issue: Installation fails due to dependency conflicts.

Solution: Try using Poetry, which provides better dependency resolution:

```bash
poetry install
```

#### 7.1.2 Missing Compiled Extensions

Issue: Installation fails due to missing compiled extensions.

Solution: Install the necessary development tools:

```bash
# Ubuntu
sudo apt-get install python3-dev build-essential

# CentOS
sudo yum install python3-devel gcc

# Windows
# Install Visual C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### 7.1.3 Permission Issues

Issue: Installation fails due to permission issues.

Solution: Use a virtual environment or install with `--user` flag:

```bash
pip install --user -e .
```

### 7.2 Module Import Errors

Issue: Import errors when running the system.

Solution: Run the project structure fix script:

```bash
python scripts/fix_project_structure.py
```

Alternatively, check your Python path:

```bash
# Add the project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/circman5
```

### 7.3 Data Directory Issues

Issue: The system cannot find or access data directories.

Solution: Ensure the data directories exist with proper permissions:

```bash
mkdir -p data/raw data/processed data/synthetic
chmod -R 755 data
```

## 8. Next Steps

After installing CIRCMAN5.0, proceed with the following steps:

1. **Configure the System**: Follow the [Configuration Guide](configuration_guide.md) to set up the system for your specific needs.

2. **Deploy the System**: Follow the [Deployment Guide](deployment_guide.md) to deploy the system in a production environment.

3. **Explore Examples**: Review the examples in the `examples/` directory to understand the system's capabilities.

4. **Read Documentation**: Explore the documentation in the `docs/` directory for detailed information on system components.

## 9. Support and Resources

If you encounter issues during installation or need additional assistance, the following resources are available:

- **GitHub Issues**: Report issues on the GitHub repository.
- **Documentation**: Refer to the documentation in the `docs/` directory.
- **Community Forum**: Join the CIRCMAN5.0 community forum for discussions and support.

## 10. Conclusion

You have successfully installed CIRCMAN5.0, a human-centered AI-aided framework for PV manufacturing. The system is now ready for configuration and deployment according to your specific requirements.

For more information on using and configuring CIRCMAN5.0, refer to the other guides in the documentation.
