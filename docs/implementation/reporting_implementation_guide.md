# Reporting Implementation Guide

## 1. Introduction

The CIRCMAN5.0 Reporting System provides comprehensive tools for generating reports, creating visualizations, and presenting manufacturing data in accessible formats. This guide describes how to implement and extend the reporting capabilities within the CIRCMAN5.0 framework.

The reporting system is designed to:
- Generate structured reports from manufacturing data
- Create visual representations of performance metrics
- Export data in standardized formats (Excel, CSV, PNG)
- Support decision-making through clear data presentation
- Integrate with other system components like the Digital Twin and Optimization modules

## 2. Architecture Overview

The Reporting System consists of four main components:

```
┌───────────────────────────┐     ┌───────────────────────────┐
│                           │     │                           │
│   Results Manager         │     │   Report Generator        │
│   (Path Management)       │◄────┤   (Data Export)           │
│                           │     │                           │
└─────────────┬─────────────┘     └───────────────────────────┘
              │
              │
┌─────────────▼─────────────┐     ┌───────────────────────────┐
│                           │     │                           │
│   Visualization Path      │◄────┤   Manufacturing           │
│   Manager                 │     │   Visualizer              │
│                           │     │                           │
└───────────────────────────┘     └──────────────┬────────────┘
                                                 │
                                  ┌──────────────▼────────────┐
                                  │                           │
                                  │   Optimization            │
                                  │   Visualizer              │
                                  │                           │
                                  └───────────────────────────┘
```

### 2.1 Component Responsibilities

1. **ReportGenerator**: Transforms manufacturing data into structured reports in Excel and other formats.
2. **ManufacturingVisualizer**: Creates visualizations for manufacturing metrics and performance data.
3. **OptimizationVisualizer**: Specialized visualizer for optimization results and comparisons.
4. **VisualizationPathManager**: Manages file paths for visualization outputs.
5. **ResultsManager**: Provides centralized path and file management for all system outputs.

### 2.2 Dependencies

The reporting system has the following dependencies:

- **pandas**: For data manipulation and Excel export
- **matplotlib**: For visualization creation
- **seaborn**: For enhanced visualization styling
- **numpy**: For numerical operations
- **Path (pathlib)**: For cross-platform path management
- **ResultsManager**: For centralized file management
- **datetime**: For timestamp generation

## 3. Core Components Implementation

### 3.1 ReportGenerator

The `ReportGenerator` class is responsible for creating structured reports in various formats.

#### 3.1.1 Class Definition

```python
# src/circman5/manufacturing/reporting/reports.py

class ReportGenerator:
    def __init__(self):
        self.logger = setup_logger("manufacturing_reports")
        self.reports_dir = results_manager.get_path("reports")
```

#### 3.1.2 Key Methods

**Generate Comprehensive Report**
```python
def generate_comprehensive_report(
    self, metrics: Dict, output_dir: Optional[Path] = None
) -> None:
    """Generate a comprehensive analysis report including all metrics."""
    try:
        # If output_dir is a file path, use it directly
        if output_dir and output_dir.suffix == ".xlsx":
            output_path = output_dir
        else:
            # If it's a directory or None, append filename
            base_dir = output_dir if output_dir else self.reports_dir
            output_path = base_dir / "comprehensive_analysis.xlsx"

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(str(output_path)) as writer:
            for metric_type, data in metrics.items():
                if isinstance(data, dict) and not any(
                    key == "error" for key in data.keys()
                ):
                    pd.DataFrame([data]).to_excel(writer, sheet_name=metric_type)

        self.logger.info(f"Comprehensive report generated at: {output_path}")

    except Exception as e:
        self.logger.error(f"Error generating comprehensive report: {str(e)}")
        raise ProcessError(f"Report generation failed: {str(e)}")
```

**Generate Performance Report**
```python
def generate_performance_report(
    self, metrics: Dict[str, float], save_path: Optional[Union[str, Path]] = None
) -> None:
    """Generate visual performance report."""
    try:
        if save_path is None:
            save_path = self.reports_dir / "performance_report.xlsx"

        excel_data = {
            "Overall Metrics": pd.DataFrame([metrics]),
            "Detailed Analysis": pd.DataFrame(
                [
                    {
                        "Manufacturing Efficiency": metrics.get("efficiency", 0),
                        "Quality Score": metrics.get("quality_score", 0),
                        "Resource Efficiency": metrics.get(
                            "resource_efficiency", 0
                        ),
                        "Energy Efficiency": metrics.get("energy_efficiency", 0),
                    }
                ]
            ),
        }

        with pd.ExcelWriter(str(save_path)) as writer:
            for sheet_name, df in excel_data.items():
                df.to_excel(writer, sheet_name=sheet_name)

        self.logger.info(f"Performance report generated at: {save_path}")

    except Exception as e:
        self.logger.error(f"Error generating performance report: {str(e)}")
        raise ProcessError(f"Performance report generation failed: {str(e)}")
```

### 3.2 VisualizationPathManager

The `VisualizationPathManager` class handles file paths for visualization outputs.

#### 3.2.1 Class Definition

```python
# src/circman5/manufacturing/reporting/visualization_paths.py

class VisualizationPathManager:
    """Manages paths for visualization outputs."""

    def __init__(self):
        self.logger = setup_logger("visualization_path_manager")
```

#### 3.2.2 Key Methods

**Get Visualization Path**
```python
def get_visualization_path(
    self, metric_type: str, filename: Optional[str] = None
) -> Path:
    """Get the full path for saving visualizations.

    Args:
        metric_type: Type of metric being visualized
        filename: Optional specific filename to use

    Returns:
        Path: Full path for saving visualization
    """
    if filename is None:
        filename = f"{metric_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    return results_manager.get_path("visualizations") / filename
```

**Ensure Visualization Directory**
```python
def ensure_visualization_directory(
    self, run_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Ensure visualization directory exists and return its path.

    Args:
        run_dir: Optional path to run directory (deprecated, kept for backwards compatibility)

    Returns:
        Path: Path to visualization directory
    """
    return results_manager.get_path("visualizations")
```

### 3.3 ManufacturingVisualizer

The `ManufacturingVisualizer` class creates visualizations for manufacturing data.

#### 3.3.1 Class Definition

```python
# src/circman5/manufacturing/reporting/visualizations.py

class ManufacturingVisualizer:
    """Creates visualizations for manufacturing metrics and performance data."""

    def __init__(self):
        """Initialize visualization settings."""
        self.logger = setup_logger("manufacturing_visualizer")
        self.viz_dir = results_manager.get_path("visualizations")
        VisualizationConfig.setup_style()
        self.colors = VisualizationConfig.COLOR_PALETTE
```

#### 3.3.2 Key Methods

**Visualize Production Trends**
```python
def visualize_production_trends(
    self, production_data: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """Enhanced production efficiency and output trends visualization with xlim handling."""
    try:
        if production_data.empty:
            self.logger.warning("No production data available for visualization")
            raise DataError("No production data available for visualization")

        # Ensure timestamps are unique and sorted
        production_data = production_data.sort_values("timestamp")

        fig = plt.figure(figsize=(12, 8))

        # Daily output plot with explicit xlim
        plt.subplot(2, 2, 1)
        daily_output = production_data.groupby(
            pd.Grouper(key="timestamp", freq="D")
        )["output_amount"].sum()
        if len(daily_output) > 1:  # Only plot if we have multiple points
            daily_output.plot(style=".-", title="Daily Production Output")
            plt.xlim(daily_output.index.min(), daily_output.index.max())
            self._add_plot_padding(plt.gca())
        plt.ylabel("Output Amount")

        # Additional plot sections...

        plt.tight_layout()

        if save_path:
            self._save_visualization(fig, save_path)
            plt.close()
            self.logger.info(
                f"Saved production trends visualization to {save_path}"
            )
        else:
            plt.show()

    except Exception as e:
        self.logger.error(f"Error creating visualization: {str(e)}")
        raise
```

**Create KPI Dashboard**
```python
def create_kpi_dashboard(
    self, metrics_data: Dict[str, float], save_path: Optional[str] = None
) -> None:
    """Create a KPI dashboard with key manufacturing metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Manufacturing KPIs Dashboard", fontsize=16)

    # Efficiency KPIs
    self._create_gauge_chart(
        axes[0, 0], metrics_data.get("efficiency", 0), "Production Efficiency", "%"
    )

    # Quality KPIs
    self._create_gauge_chart(
        axes[0, 1], metrics_data.get("quality_score", 0), "Quality Score", "%"
    )

    # Resource KPIs
    self._create_gauge_chart(
        axes[1, 0],
        metrics_data.get("resource_efficiency", 0),
        "Resource Efficiency",
        "%",
    )

    # Energy KPIs
    self._create_gauge_chart(
        axes[1, 1],
        metrics_data.get("energy_efficiency", 0),
        "Energy Efficiency",
        "%",
    )

    plt.tight_layout()

    if save_path:
        self._save_visualization(fig, save_path)
        plt.close()
    else:
        plt.show()
```

### 3.4 OptimizationVisualizer

The `OptimizationVisualizer` class creates specialized visualizations for optimization results.

#### 3.4.1 Class Definition

```python
# src/circman5/manufacturing/reporting/optimization_visualizer.py

class OptimizationVisualizer:
    """Visualization component for optimization results."""

    def __init__(self):
        self.results_dir = results_manager.get_path("visualizations")
        self.setup_style()
```

#### 3.4.2 Key Methods

**Plot Optimization Impact**
```python
def plot_optimization_impact(self, results: OptimizationResults) -> Path:
    """Create bar plot showing impact of optimization on key parameters."""
    fig, ax = plt.subplots(figsize=(12, 6))

    params = list(results["original_params"].keys())
    improvements = [results["improvement"][param] for param in params]

    # Create color gradient based on improvement values
    colors = plt.cm.get_cmap("coolwarm")(np.linspace(0, 1, len(improvements)))
    bars = ax.bar(params, improvements, color=colors)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.title("Optimization Impact by Parameter")
    plt.xlabel("Parameters")
    plt.ylabel("Improvement (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = self.results_dir / "optimization_impact.png"
    plt.savefig(output_path)
    plt.close()

    return output_path
```

**Create Optimization Dashboard**
```python
def create_optimization_dashboard(
    self, results: OptimizationResults, metrics: MetricsDict
) -> Path:
    """Create comprehensive dashboard with all optimization visualizations."""
    fig = plt.figure(figsize=(20, 15))

    # Feature importance (top left)
    ax1 = plt.subplot(221)
    importance_dict = metrics["feature_importance"]
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    y_pos = np.arange(len(features))
    ax1.barh(y_pos, importance)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.set_title("Feature Importance")

    # Additional plot sections...

    plt.tight_layout()

    output_path = self.results_dir / "optimization_dashboard.png"
    plt.savefig(output_path)
    plt.close()

    return output_path
```

## 4. Implementation Details

### 4.1 Report Generation Implementation

#### 4.1.1 Excel Report Generation

The `ReportGenerator` uses pandas and `ExcelWriter` to create Excel-based reports:

```python
def export_analysis_report(
    self, report_data: Dict, output_path: Optional[Union[str, Path]] = None
) -> None:
    """Generate and export comprehensive analysis report."""
    try:
        if output_path is None:
            output_path = self.reports_dir / "analysis_report.xlsx"

        with pd.ExcelWriter(str(output_path)) as writer:
            has_data = False
            for metric_type, data in report_data.items():
                if isinstance(data, dict) and not any(
                    key == "error" for key in data.keys()
                ):
                    pd.DataFrame([data]).to_excel(writer, sheet_name=metric_type)
                    has_data = True

            if not has_data:
                pd.DataFrame(["No data available"]).to_excel(
                    writer, sheet_name="Empty_Report"
                )

        self.logger.info(f"Analysis report exported to: {output_path}")

    except Exception as e:
        self.logger.error(f"Error exporting analysis report: {str(e)}")
        raise ProcessError(f"Report export failed: {str(e)}")
```

Key implementation aspects:
- Uses the `with` statement to ensure proper resource management
- Creates multiple sheets in the Excel workbook, one for each metric type
- Converts dictionaries to pandas DataFrames for Excel export
- Handles the case where no data is available
- Implements comprehensive error handling

#### 4.1.2 LCA Report Generation

The LCA (Life Cycle Assessment) report generation has specific formatting:

```python
def generate_lca_report(
    self, impact_data: Dict, batch_id: Optional[str] = None
) -> None:
    """Generate comprehensive LCA report."""
    try:
        output_path = (
            self.reports_dir
            / f"lca_report{'_' + batch_id if batch_id else ''}.xlsx"
        )

        with pd.ExcelWriter(str(output_path)) as writer:
            for category, data in impact_data.items():
                pd.DataFrame([data]).to_excel(writer, sheet_name=category)

        self.logger.info(f"LCA report generated successfully at {output_path}")

    except Exception as e:
        self.logger.error(f"Error generating LCA report: {str(e)}")
        raise ProcessError(f"LCA report generation failed: {str(e)}")
```

Key implementation aspects:
- Creates a unique filename based on the batch ID if provided
- Creates a sheet for each impact category in the LCA data
- Uses a consistent approach to error handling

### 4.2 Visualization Implementation

#### 4.2.1 Matplotlib Configuration

The `ManufacturingVisualizer` configures matplotlib for consistent styling:

```python
def setup_style(self):
    """Configure matplotlib style for visualizations."""
    # Use default style with customizations instead of seaborn
    plt.style.use("default")
    # Set custom parameters
    plt.rcParams.update(
        {
            "figure.figsize": [10, 6],
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "figure.autolayout": True,
        }
    )
```

Key implementation aspects:
- Sets consistent figure size and DPI for all visualizations
- Configures grid, labels, and line properties
- Ensures a professional appearance across all visualizations

#### 4.2.2 Handling Empty Datasets

The visualization system includes robust handling for empty datasets:

```python
def visualize_quality_metrics(
    self, quality_data: pd.DataFrame, analyzer=None, save_path: Optional[str] = None
) -> None:
    """Enhanced quality control metrics visualization."""
    if quality_data.empty:
        self.logger.warning("No quality data available for visualization")
        raise DataError("No quality data available for visualization")

    # Visualization code...
```

Key implementation aspects:
- Checks for empty DataFrames before attempting visualization
- Raises specific exceptions for better error handling
- Logs warnings to aid in troubleshooting

#### 4.2.3 Plot Padding for Edge Cases

The visualizer includes special handling for edge cases in plots:

```python
def _add_plot_padding(self, ax, padding=0.05):
    """Add padding to plot limits to avoid singular transformations."""
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Add padding if limits are identical
    if xlims[0] == xlims[1]:
        value = xlims[0]
        delta = max(abs(value) * padding, 0.1)  # At least 0.1 padding
        ax.set_xlim(value - delta, value + delta)

    if ylims[0] == ylims[1]:
        value = ylims[0]
        delta = max(abs(value) * padding, 0.1)
        ax.set_ylim(value - delta, value + delta)
```

Key implementation aspects:
- Handles the case where plot limits are identical (a common source of errors)
- Adds appropriate padding based on the value magnitude
- Prevents matplotlib errors from singular transformations

### 4.3 Optimization Visualization Implementation

#### 4.3.1 Loading Optimization Results

The `OptimizationVisualizer` loads results from JSON files:

```python
def load_optimization_results(
    self, results_path: Union[str, Path]
) -> OptimizationResults:
    """Load optimization results from JSON file."""
    with open(results_path, "r") as f:
        data = json.load(f)
    return OptimizationResults(**data)
```

Key implementation aspects:
- Opens and reads JSON files with proper error handling
- Converts JSON data to the `OptimizationResults` type
- Uses Python's type system for better code organization

#### 4.3.2 Color Gradients in Visualizations

The optimization visualizer uses color gradients for enhanced visualizations:

```python
def plot_feature_importance(self, metrics: MetricsDict) -> Path:
    """Create horizontal bar plot of feature importance scores."""
    importance_dict = metrics["feature_importance"]
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))

    # Create horizontal bars with color gradient
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(features)))
    ax.barh(y_pos, importance, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()

    plt.title("Feature Importance Scores")
    plt.xlabel("Importance Score")

    output_path = self.results_dir / "feature_importance.png"
    plt.savefig(output_path)
    plt.close()

    return output_path
```

Key implementation aspects:
- Uses the viridis colormap for a modern, perceptually uniform color gradient
- Maps colors to the number of features using numpy's linspace
- Returns the output path for further reference

## 5. Extending the Reporting System

### 5.1 Adding New Report Types

To add a new report type to the `ReportGenerator` class:

```python
def generate_custom_report(
    self, data: Dict[str, Any], report_name: str, output_dir: Optional[Path] = None
) -> Path:
    """Generate custom report with specific format.

    Args:
        data: Data to include in the report
        report_name: Name of the report
        output_dir: Optional output directory

    Returns:
        Path: Path to the generated report
    """
    try:
        # Determine output path
        if output_dir is None:
            output_path = self.reports_dir / f"{report_name}.xlsx"
        else:
            output_path = output_dir / f"{report_name}.xlsx"

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process data as needed
        processed_data = self._process_custom_data(data)

        # Write to Excel
        with pd.ExcelWriter(str(output_path)) as writer:
            for section_name, section_data in processed_data.items():
                pd.DataFrame(section_data).to_excel(
                    writer, sheet_name=section_name
                )

        self.logger.info(f"Custom report generated at: {output_path}")
        return output_path

    except Exception as e:
        self.logger.error(f"Error generating custom report: {str(e)}")
        raise ProcessError(f"Custom report generation failed: {str(e)}")

def _process_custom_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Process data for custom report format."""
    # Implementation of custom data processing
    processed_data = {}
    # ...
    return processed_data
```

### 5.2 Adding New Visualization Types

To add a new visualization type to the `ManufacturingVisualizer` class:

```python
def visualize_custom_metric(
    self, data: pd.DataFrame, metric_column: str, save_path: Optional[str] = None
) -> None:
    """Create visualization for custom metric.

    Args:
        data: DataFrame containing the data
        metric_column: Column name for the metric to visualize
        save_path: Optional path to save the visualization
    """
    try:
        if data.empty:
            self.logger.warning("No data available for visualization")
            raise DataError("No data available for visualization")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot time series data
        data.plot(x="timestamp", y=metric_column, ax=ax, marker='o')

        # Add trend line
        z = np.polyfit(range(len(data)), data[metric_column], 1)
        p = np.poly1d(z)
        ax.plot(data["timestamp"], p(range(len(data))), "r--",
                label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")

        # Add annotations
        ax.axhline(y=data[metric_column].mean(), color='g', linestyle='-',
                  label=f"Mean: {data[metric_column].mean():.2f}")

        # Formatting
        ax.set_title(f"{metric_column.replace('_', ' ').title()} Trend")
        ax.set_xlabel("Time")
        ax.set_ylabel(metric_column.replace('_', ' ').title())
        ax.legend()
        ax.grid(True)

        # Save or show
        if save_path:
            self._save_visualization(fig, save_path)
            plt.close()
        else:
            plt.show()

    except Exception as e:
        self.logger.error(f"Error creating custom visualization: {str(e)}")
        plt.close()
        raise ProcessError(f"Visualization generation failed: {str(e)}")
```

### 5.3 Customizing Visualization Styles

To create a custom visualization style:

```python
def create_custom_style(self):
    """Create custom visualization style for specific needs."""
    # Define custom color palette
    self.custom_colors = ["#3366CC", "#DC3912", "#FF9900", "#109618", "#990099"]

    # Create custom style
    plt.style.use("default")  # Start with default style

    # Update parameters for custom style
    plt.rcParams.update({
        "figure.figsize": [12, 8],
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.7,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2
    })

    return self
```

## 6. Integration with Other Components

### 6.1 Integration with Digital Twin

To integrate the reporting system with the Digital Twin:

```python
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwinCore
from circman5.manufacturing.reporting.reports import ReportGenerator
from circman5.manufacturing.reporting.visualizations import ManufacturingVisualizer

class DigitalTwinReporting:
    """Integrates Digital Twin with the reporting system."""

    def __init__(self, digital_twin: DigitalTwinCore):
        self.digital_twin = digital_twin
        self.report_generator = ReportGenerator()
        self.visualizer = ManufacturingVisualizer()

    def generate_twin_state_report(self, state_id: str) -> Path:
        """Generate report from Digital Twin state.

        Args:
            state_id: State identifier

        Returns:
            Path: Path to the generated report
        """
        # Get state from Digital Twin
        state_data = self.digital_twin.state_manager.get_state(state_id)

        # Transform state data to report format
        report_data = self._transform_state_to_report(state_data)

        # Generate report
        output_path = self.report_generator.reports_dir / f"twin_state_{state_id}.xlsx"
        self.report_generator.generate_comprehensive_report(report_data, output_path)

        return output_path

    def visualize_twin_state(self, state_id: str) -> Path:
        """Create visualization of Digital Twin state.

        Args:
            state_id: State identifier

        Returns:
            Path: Path to the visualization
        """
        # Get state from Digital Twin
        state_data = self.digital_twin.state_manager.get_state(state_id)

        # Transform state data to visualization format
        vis_data = self._transform_state_to_visualization(state_data)

        # Generate visualization
        save_path = self.visualizer.viz_dir / f"twin_state_{state_id}.png"
        self.visualizer.create_kpi_dashboard(vis_data, str(save_path))

        return save_path

    def _transform_state_to_report(self, state_data: Dict) -> Dict:
        """Transform Digital Twin state to report format."""
        # Implementation details
        return transformed_data

    def _transform_state_to_visualization(self, state_data: Dict) -> Dict:
        """Transform Digital Twin state to visualization format."""
        # Implementation details
        return transformed_data
```

### 6.2 Integration with Optimization Module

To integrate with the optimization module:

```python
from circman5.manufacturing.optimization.model import OptimizationModel
from circman5.manufacturing.optimization.types import OptimizationResults
from circman5.manufacturing.reporting.optimization_visualizer import OptimizationVisualizer

class OptimizationReporter:
    """Reporting integration for optimization module."""

    def __init__(self, optimization_model: OptimizationModel):
        self.optimization_model = optimization_model
        self.visualizer = OptimizationVisualizer()

    def generate_optimization_report(self, results: OptimizationResults) -> List[Path]:
        """Generate comprehensive optimization report.

        Args:
            results: Optimization results

        Returns:
            List[Path]: Paths to generated visualizations
        """
        output_paths = []

        # Generate individual visualizations
        impact_path = self.visualizer.plot_optimization_impact(results)
        output_paths.append(impact_path)

        feature_path = self.visualizer.plot_feature_importance(
            self.optimization_model.get_metrics()
        )
        output_paths.append(feature_path)

        convergence_path = self.visualizer.plot_convergence_history(results)
        output_paths.append(convergence_path)

        param_path = self.visualizer.plot_parameter_comparison(results)
        output_paths.append(param_path)

        # Generate dashboard
        dashboard_path = self.visualizer.create_optimization_dashboard(
            results, self.optimization_model.get_metrics()
        )
        output_paths.append(dashboard_path)

        return output_paths
```

### 6.3 Integration with Monitoring System

To integrate with the monitoring system:

```python
from circman5.monitoring import ManufacturingMonitor
from circman5.manufacturing.reporting.reports import ReportGenerator
from circman5.manufacturing.reporting.visualizations import ManufacturingVisualizer

class MonitoringReporter:
    """Reporting integration for monitoring system."""

    def __init__(self, monitor: ManufacturingMonitor):
        self.monitor = monitor
        self.report_generator = ReportGenerator()
        self.visualizer = ManufacturingVisualizer()

    def generate_batch_report(self, batch_id: str) -> Dict[str, Path]:
        """Generate comprehensive batch report.

        Args:
            batch_id: Batch identifier

        Returns:
            Dict[str, Path]: Paths to generated reports and visualizations
        """
        outputs = {}

        # Get batch summary from monitor
        batch_summary = self.monitor.get_batch_summary(batch_id)

        # Generate report
        report_path = self.report_generator.reports_dir / f"batch_{batch_id}_report.xlsx"
        self.report_generator.generate_comprehensive_report(batch_summary, report_path)
        outputs["report"] = report_path

        # Generate efficiency visualization
        efficiency_data = self.monitor.metrics_history["efficiency"]
        batch_efficiency = efficiency_data[efficiency_data["batch_id"] == batch_id]
        efficiency_path = self.visualizer.viz_dir / f"batch_{batch_id}_efficiency.png"
        self.visualizer.plot_efficiency_trends(batch_efficiency, str(efficiency_path))
        outputs["efficiency_viz"] = efficiency_path

        # Generate quality visualization
        quality_data = self.monitor.metrics_history["quality"]
        batch_quality = quality_data[quality_data["batch_id"] == batch_id]
        quality_path = self.visualizer.viz_dir / f"batch_{batch_id}_quality.png"
        self.visualizer.plot_quality_metrics(batch_quality, str(quality_path))
        outputs["quality_viz"] = quality_path

        # Generate KPI dashboard
        kpi_path = self.visualizer.viz_dir / f"batch_{batch_id}_kpi.png"
        self.visualizer.create_kpi_dashboard({
            "efficiency": batch_summary["efficiency"]["avg_production_rate"],
            "quality_score": batch_summary["quality"]["avg_quality_score"],
            "resource_efficiency": batch_summary["resources"]["avg_resource_efficiency"],
            "energy_efficiency": batch_summary["efficiency"]["avg_energy_efficiency"]
        }, str(kpi_path))
        outputs["kpi_dashboard"] = kpi_path

        return outputs
```

## 7. Error Handling and Validation

### 7.1 Error Handling Pattern

The reporting system uses a consistent error handling pattern:

```python
try:
    # Operation that might fail
    # ...

    # Log success
    self.logger.info("Operation succeeded")

except SpecificException as e:
    # Log specific error
    self.logger.error(f"Specific error occurred: {str(e)}")
    # Handle or re-raise as appropriate
    raise ProcessError(f"Operation failed due to: {str(e)}")

except Exception as e:
    # Log unexpected error
    self.logger.error(f"Unexpected error: {str(e)}")
    # Handle or re-raise as appropriate
    raise ProcessError(f"Operation failed unexpectedly: {str(e)}")
```

### 7.2 Data Validation

Implement data validation to ensure visualization quality:

```python
def _validate_visualization_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate data for visualization.

    Args:
        data: DataFrame to validate
        required_columns: List of required column names

    Returns:
        bool: True if valid, False otherwise
    """
    # Check if empty
    if data.empty:
        self.logger.warning("Data frame is empty")
        return False

    # Check required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        self.logger.warning(f"Missing required columns: {missing_columns}")
        return False

    # Check data types
    if not isinstance(data["timestamp"].iloc[0], (pd.Timestamp, datetime)):
        self.logger.warning("Timestamp column has incorrect type")
        return False

    # Check for NaN values in critical columns
    for col in required_columns:
        if data[col].isna().any():
            self.logger.warning(f"Column {col} contains NaN values")
            return False

    return True
```

### 7.3 Output Path Validation

Ensure output paths are valid:

```python
def _validate_save_path(self, save_path: Union[str, Path]) -> Path:
    """Validate and standardize save path.

    Args:
        save_path: Path to validate

    Returns:
        Path: Validated and standardized path
    """
    # Convert to Path object
    if isinstance(save_path, str):
        path = Path(save_path)
    else:
        path = save_path

    # Check extension for appropriate file type
    if path.suffix.lower() not in [".png", ".jpg", ".pdf", ".svg"]:
        self.logger.warning(f"Unsupported file extension: {path.suffix}")
        # Append default extension
        path = path.with_suffix(".png")

    # Ensure parent directory exists
    if not path.parent.exists():
        self.logger.info(f"Creating directory: {path.parent}")
        path.parent.mkdir(parents=True, exist_ok=True)

    return path
```

## 8. Testing Strategy

### 8.1 Unit Testing

Unit tests for the reporting system should cover:

1. **Report Generation**: Test each report type with valid and invalid inputs
2. **Visualization Creation**: Test each visualization type with different datasets
3. **Path Management**: Test path creation and validation
4. **Error Handling**: Test error conditions and recovery

Example test for the `ReportGenerator`:

```python
def test_generate_comprehensive_report(report_generator, sample_metrics, reports_dir):
    """Test comprehensive report generation."""
    # Prepare test data
    output_file = reports_dir / "test_report.xlsx"

    # Execute function under test
    report_generator.generate_comprehensive_report(sample_metrics, output_file)

    # Verify results
    assert output_file.exists()

    # Verify content
    df_dict = pd.read_excel(output_file, sheet_name=None)
    assert set(df_dict.keys()) == set(sample_metrics.keys())

    # Verify each sheet has expected data
    for key in sample_metrics:
        assert key in df_dict
        # Additional verification as needed
```

### 8.2 Integration Testing

Integration tests should verify:

1. **System Integration**: Test integration with Digital Twin, Optimization, etc.
2. **Data Flow**: Verify correct data flow through the system
3. **Error Propagation**: Test error propagation across components

Example integration test:

```python
def test_digital_twin_reporting_integration(digital_twin, report_generator, visualizer):
    """Test integration between Digital Twin and reporting."""
    # Setup
    dt_reporting = DigitalTwinReporting(digital_twin)

    # Create test state in Digital Twin
    state_id = digital_twin.create_test_state()

    # Generate report
    report_path = dt_reporting.generate_twin_state_report(state_id)

    # Verify report
    assert report_path.exists()

    # Generate visualization
    viz_path = dt_reporting.visualize_twin_state(state_id)

    # Verify visualization
    assert viz_path.exists()
```

## 9. Best Practices

### 9.1 Report Generation

1. **Consistent Structure**: Maintain consistent report structure across different types
2. **Clear Sections**: Organize reports into logical sections
3. **Appropriate Formats**: Use Excel for detailed data, PDF for formal reports
4. **Data Validation**: Validate input data before generating reports
5. **Error Handling**: Implement robust error handling for all operations
6. **Descriptive Filenames**: Use descriptive, timestamp-based filenames
7. **Metadata**: Include metadata in reports (generation time, source, version)

### 9.2 Visualization Creation

1. **Consistent Styling**: Use consistent styling across all visualizations
2. **Clear Labels**: Provide clear titles, axis labels, and legends
3. **Color Accessibility**: Use colorblind-friendly palettes
4. **Data Filtering**: Filter data to show relevant information only
5. **Plot Types**: Use appropriate plot types for different data characteristics
6. **Empty Data**: Handle empty datasets gracefully
7. **Resource Management**: Close matplotlib figures to prevent memory leaks

### 9.3 Path Management

1. **Centralized Management**: Use ResultsManager for centralized path management
2. **Standard Directories**: Follow standard directory structure
3. **Path Validation**: Validate paths before use
4. **Directory Creation**: Ensure directories exist before saving files
5. **Relative Paths**: Use relative paths for portability
6. **Path Objects**: Use Path objects instead of strings for better cross-platform support

### 9.4 Performance Considerations

1. **Batch Processing**: Process data in batches for large datasets
2. **Resource Cleanup**: Close files and resources properly
3. **Memory Management**: Use iterators for large data processing
4. **Figure Size**: Balance figure size and DPI for appropriate file sizes
5. **Caching**: Consider caching for frequently accessed data

## 10. Conclusion

The CIRCMAN5.0 Reporting System provides a flexible framework for generating reports and visualizations from manufacturing data. By following this implementation guide, you can effectively extend and customize the reporting capabilities to meet specific project requirements.

Key points to remember:
- Use the `ReportGenerator` for structured data reports
- Use the `ManufacturingVisualizer` for general manufacturing visualizations
- Use the `OptimizationVisualizer` for optimization-specific visualizations
- Implement proper error handling and validation
- Follow consistent styling for professional output
- Integrate with other system components for comprehensive reporting

## 11. Related Documentation

- [Adapter API Reference](../api/adapter_api_reference.md) - Details on configuration adapters
- [Utilities API Reference](../api/utilities_api_reference.md) - Information on utility functions including ResultsManager
- [Digital Twin Integration Guide](../implementation/dt_integration_guide.md) - Digital Twin integration details
- [Monitoring Guide](../guides/monitoring_guide.md) - Information on the monitoring system that provides data
- [Optimization Implementation Guide](../implementation/optimization_implementation_guide.md) - Details on the optimization system
