# scripts/visualization/synthetic_data_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Import the results_manager from your project
from circman5.utils.results_manager import results_manager
from circman5.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger("synthetic_data_analysis")

# Use a standard matplotlib backend
plt.switch_backend("agg")

# Get paths from results_manager
visualizations_dir = results_manager.get_path("visualizations")
synthetic_data_dir = results_manager.get_path("SYNTHETIC_DATA")
temp_dir = results_manager.get_path("temp")

logger.info(f"Loading data from {synthetic_data_dir}")
logger.info(f"Saving visualizations to {visualizations_dir}")

# Load synthetic datasets
production_data = pd.read_csv(synthetic_data_dir / "production_data.csv")
quality_data = pd.read_csv(synthetic_data_dir / "quality_data.csv")
time_series_data = pd.read_csv(synthetic_data_dir / "time_series_data.csv")
edge_cases = pd.read_csv(synthetic_data_dir / "edge_cases.csv")

# Convert timestamps to datetime
for df in [production_data, quality_data, time_series_data]:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

logger.info("Data loaded successfully, generating improved visualizations...")

# Figure 1: Production metrics distribution (IMPROVED)
plt.figure(figsize=(10, 6))
# Create separate histograms with transparency
plt.hist(
    production_data["input_amount"],
    bins=15,
    alpha=0.7,
    label="Input Amount",
    color="orange",
)
plt.hist(
    production_data["output_amount"],
    bins=15,
    alpha=0.7,
    label="Output Amount",
    color="blue",
)
plt.xlabel("Amount (units)")
plt.ylabel("Frequency")
plt.title("Distribution of Production Metrics")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
temp_file = temp_dir / "production_distribution.png"
plt.savefig(temp_file, dpi=300)
plt.close()
results_manager.save_file(temp_file, "visualizations")

# Figure 2: Statistical validation of yield rate (IMPROVED)
plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    x="production_line", y="yield_rate", data=production_data, palette="Set2"
)
# Add individual points for transparency
sns.stripplot(
    x="production_line",
    y="yield_rate",
    data=production_data,
    size=4,
    color=".3",
    linewidth=0,
    alpha=0.3,
)
plt.xlabel("Production Line")
plt.ylabel("Yield Rate (%)")
plt.title("Yield Rate by Production Line")
# Add mean values as text
for i, line in enumerate(production_data["production_line"].unique()):
    mean_val = production_data[production_data["production_line"] == line][
        "yield_rate"
    ].mean()
    plt.text(i, mean_val + 0.5, f"Mean: {mean_val:.2f}%", ha="center")
plt.grid(axis="y", linestyle="--", alpha=0.7)
temp_file = temp_dir / "yield_rate_validation.png"
plt.savefig(temp_file, dpi=300)
plt.close()
results_manager.save_file(temp_file, "visualizations")

# Figure 3: Energy consumption pattern (IMPROVED)
plt.figure(figsize=(10, 6))
energy_daily = production_data.copy()
energy_daily["date"] = energy_daily["timestamp"].dt.date
daily_energy = energy_daily.groupby("date")["energy_used"].mean().reset_index()
daily_energy["date"] = pd.to_datetime(daily_energy["date"])
# Add day of week for pattern visibility
daily_energy["day_of_week"] = daily_energy["date"].dt.day_name()

plt.plot(daily_energy["date"], daily_energy["energy_used"], marker="o", markersize=4)
# Highlight weekends
weekend_mask = daily_energy["day_of_week"].isin(["Saturday", "Sunday"])
if weekend_mask.any():
    plt.scatter(
        daily_energy.loc[weekend_mask, "date"],
        daily_energy.loc[weekend_mask, "energy_used"],
        color="red",
        label="Weekends",
        s=80,
        alpha=0.5,
    )
    plt.legend()

plt.xlabel("Date")
plt.ylabel("Average Energy Consumption (kWh)")
plt.title("Daily Energy Consumption Pattern")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.7)
temp_file = temp_dir / "energy_consumption_pattern.png"
plt.savefig(temp_file, dpi=300, bbox_inches="tight")
plt.close()
results_manager.save_file(temp_file, "visualizations")

# Figure 4: Correlation matrix of manufacturing parameters (IMPROVED)
plt.figure(figsize=(10, 8))
corr_cols = ["input_amount", "output_amount", "energy_used", "cycle_time", "yield_rate"]
corr_matrix = production_data[corr_cols].corr()
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# Using nicer colormap and annotations with custom formatting
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
)
plt.title("Correlation Matrix of Manufacturing Parameters", pad=20)
temp_file = temp_dir / "parameter_correlation.png"
plt.savefig(temp_file, dpi=300)
plt.close()
results_manager.save_file(temp_file, "visualizations")

# Figure 5: Time series analysis - Efficiency and defect rate (IMPROVED)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# Get a subset for better visibility
time_series_subset = time_series_data.iloc[:500]

# First subplot - Efficiency
ax1.plot(time_series_subset["timestamp"], time_series_subset["efficiency"])

# Simplified approach for maintenance markers - avoid the complex indexing
# Just mark positions based on timestamp index directly
for i in range(
    0, len(time_series_subset), 30 * 24
):  # Every ~7 days (assuming hourly data)
    if i < len(time_series_subset):
        # Simply mark based on position in dataframe, not date logic
        maintenance_date = time_series_subset["timestamp"].iloc[i]
        ax1.axvline(x=maintenance_date, color="r", linestyle="--", alpha=0.5)
        ax1.text(
            maintenance_date,
            0.84,
            "Maintenance",
            rotation=90,
            va="top",
            ha="right",
            color="r",
        )

ax1.set_ylabel("Efficiency")
ax1.set_title("Manufacturing Efficiency Over Time")
ax1.grid(True, linestyle="--", alpha=0.3)

# Second subplot - Defect Rate
ax2.plot(time_series_subset["timestamp"], time_series_subset["defect_rate"])
# Highlight anomalies if present
anomaly_mask = time_series_subset["anomaly_present"] == True
if anomaly_mask.any():
    anomaly_timestamps = time_series_subset.loc[anomaly_mask, "timestamp"]
    anomaly_values = time_series_subset.loc[anomaly_mask, "defect_rate"]
    ax2.scatter(anomaly_timestamps, anomaly_values, color="r", label="Anomalies", s=50)
    ax2.legend()

ax2.set_ylabel("Defect Rate (%)")
ax2.set_xlabel("Time")
ax2.set_title("Defect Rate Over Time")
ax2.grid(True, linestyle="--", alpha=0.3)

# Format x-axis to show readable dates
fig.autofmt_xdate()
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))

plt.tight_layout()
temp_file = temp_dir / "efficiency_defect_timeseries.png"
plt.savefig(temp_file, dpi=300)
plt.close()
results_manager.save_file(temp_file, "visualizations")

# Figure 6: Quality metrics comparison (IMPROVED)
plt.figure(figsize=(10, 6))
# Normalize quality metrics for better comparison
quality_norm = quality_data.copy()
for col in ["efficiency", "defect_rate", "thickness_uniformity", "contamination_level"]:
    if quality_data[col].std() > 0:  # Prevent division by zero
        quality_norm[col] = (quality_data[col] - quality_data[col].min()) / (
            quality_data[col].max() - quality_data[col].min()
        )
    else:
        quality_norm[col] = 0  # Set to constant if no variation

# Plot normalized metrics
sns.boxplot(data=quality_norm)
plt.ylabel("Normalized Value (0-1)")
plt.title("Normalized Distribution of Quality Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.7)
# Add original mean values as text
for i, col in enumerate(quality_norm.columns):
    if col in [
        "efficiency",
        "defect_rate",
        "thickness_uniformity",
        "contamination_level",
    ]:
        mean_val = quality_data[col].mean()
        plt.text(i, 1.05, f"Mean: {mean_val:.2f}", ha="center")
temp_file = temp_dir / "quality_metrics_normalized.png"
plt.savefig(temp_file, dpi=300)
plt.close()
results_manager.save_file(temp_file, "visualizations")

# Figure 7: Edge cases analysis (IMPROVED)
plt.figure(figsize=(12, 8))
# Use better categorical colors
sns.scatterplot(
    data=edge_cases,
    x="input_amount",
    y="energy_used",
    hue="case_type",
    size="defect_rate",
    sizes=(20, 200),
    alpha=0.8,
    palette="tab10",
)

# Add annotations for extreme points
for case_type in ["extreme_input", "zero_energy", "maximum_defect_rate"]:
    case_data = edge_cases[edge_cases["case_type"] == case_type]
    if not case_data.empty:
        for idx, row in case_data.iterrows():
            plt.annotate(
                case_type,
                (row["input_amount"], row["energy_used"]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

plt.xlabel("Input Amount")
plt.ylabel("Energy Used")
plt.title("Edge Cases by Type, Input Amount, and Energy Usage")
plt.grid(True, linestyle="--", alpha=0.3)
# Move legend outside plot area
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
temp_file = temp_dir / "edge_cases_analysis.png"
plt.savefig(temp_file, dpi=300, bbox_inches="tight")
plt.close()
results_manager.save_file(temp_file, "visualizations")

# Figure 8: Working hours vs. non-working hours efficiency (IMPROVED)
plt.figure(figsize=(10, 6))

# Just use histograms instead of KDE plots to avoid the issues
plt.hist(
    time_series_data[time_series_data["is_working_hours"] == True]["efficiency"],
    bins=30,
    alpha=0.6,
    label="Working Hours",
    color="blue",
)
plt.hist(
    time_series_data[time_series_data["is_working_hours"] == False]["efficiency"],
    bins=30,
    alpha=0.6,
    label="Non-Working Hours",
    color="orange",
)

# Add statistical details
working_mean = time_series_data[time_series_data["is_working_hours"] == True][
    "efficiency"
].mean()
non_working_mean = time_series_data[time_series_data["is_working_hours"] == False][
    "efficiency"
].mean()

plt.axvline(working_mean, color="blue", linestyle="--", alpha=0.7)
plt.axvline(non_working_mean, color="orange", linestyle="--", alpha=0.7)

plt.text(
    working_mean,
    plt.gca().get_ylim()[1] * 0.9,
    f"Mean: {working_mean:.3f}",
    color="blue",
    ha="center",
    bbox=dict(facecolor="white", alpha=0.5),
)
plt.text(
    non_working_mean,
    plt.gca().get_ylim()[1] * 0.8,
    f"Mean: {non_working_mean:.3f}",
    color="orange",
    ha="center",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.xlabel("Efficiency")
plt.ylabel("Frequency")
plt.title("Efficiency Distribution: Working vs. Non-Working Hours")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
temp_file = temp_dir / "working_hours_efficiency.png"
plt.savefig(temp_file, dpi=300)
plt.close()
results_manager.save_file(temp_file, "visualizations")

logger.info(f"All visualizations saved to {visualizations_dir}")

# Generate metadata for the visualization run
metadata = {
    "visualization_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "datasets_used": [
        "production_data.csv",
        "quality_data.csv",
        "time_series_data.csv",
        "edge_cases.csv",
    ],
    "figures_generated": 8,
    "data_records_processed": {
        "production": len(production_data),
        "quality": len(quality_data),
        "time_series": len(time_series_data),
        "edge_cases": len(edge_cases),
    },
}

# Save metadata to a report
metadata_df = pd.DataFrame([metadata])
metadata_file = temp_dir / "visualization_metadata.csv"
metadata_df.to_csv(metadata_file, index=False)
results_manager.save_file(metadata_file, "reports")

logger.info("Improved analysis complete")
print(f"All visualizations saved to {visualizations_dir}")
print(f"Metadata report saved to {results_manager.get_path('reports')}")
