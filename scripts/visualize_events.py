"""
Digital Twin Event Visualization Script

This script reads the event data from the JSONL file and creates visualizations
to help analyze the Digital Twin behavior during testing.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# Set the path to your events file
events_file = "tests/results/runs/run_20250410_114810/events/events_20250410.jsonl"

# Check if file exists
if not os.path.exists(events_file):
    print(f"Error: File {events_file} not found")
    exit(1)

# Read events from JSONL file
events = []
with open(events_file, "r") as f:
    for line in f:
        events.append(json.loads(line))

print(f"Loaded {len(events)} events")

# Convert to DataFrame for easier analysis
df = pd.DataFrame(events)

# Extract timestamps and convert to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Create directory for outputs if it doesn't exist
output_dir = "validation_results"
os.makedirs(output_dir, exist_ok=True)

# 1. Event type distribution
plt.figure(figsize=(10, 6))
event_counts = df["category"].value_counts()
event_counts.plot(kind="bar", color="skyblue")
plt.title("Distribution of Event Types")
plt.xlabel("Event Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/event_distribution.png")

# 2. Event severity distribution
plt.figure(figsize=(10, 6))
severity_counts = df["severity"].value_counts()
colors = {"info": "green", "warning": "orange", "error": "red"}
severity_counts.plot(
    kind="bar", color=[colors.get(s, "blue") for s in severity_counts.index]
)
plt.title("Distribution of Event Severities")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/severity_distribution.png")

# 3. Extract temperature values from threshold events
temperature_events = []
energy_events = []

for event in events:
    if event["category"] == "threshold":
        details = event["details"]
        parameter = details.get("parameter", "")
        if "Temperature" in parameter:
            temperature_events.append(
                {
                    "timestamp": event["timestamp"],
                    "value": details["actual_value"],
                    "threshold": details["threshold"],
                }
            )
        elif "Energy" in parameter:
            energy_events.append(
                {
                    "timestamp": event["timestamp"],
                    "value": details["actual_value"],
                    "threshold": details["threshold"],
                }
            )

# Plot temperature breaches if any
if temperature_events:
    temp_df = pd.DataFrame(temperature_events)
    temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"])

    plt.figure(figsize=(12, 6))
    plt.plot(temp_df["timestamp"], temp_df["value"], "ro-", label="Actual Temperature")
    plt.axhline(
        y=temp_df["threshold"].iloc[0], color="r", linestyle="--", label="Threshold"
    )
    plt.title("Temperature Threshold Breaches")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temperature_breaches.png")

# Plot energy consumption breaches if any
if energy_events:
    energy_df = pd.DataFrame(energy_events)
    energy_df["timestamp"] = pd.to_datetime(energy_df["timestamp"])

    plt.figure(figsize=(12, 6))
    plt.plot(energy_df["timestamp"], energy_df["value"], "bo-", label="Actual Energy")
    plt.axhline(
        y=energy_df["threshold"].iloc[0], color="b", linestyle="--", label="Threshold"
    )
    plt.title("Energy Consumption Threshold Breaches")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_breaches.png")

# 4. Create a timeline of events
plt.figure(figsize=(14, 8))
categories = df["category"].unique()
colors = {"threshold": "red", "process": "blue", "state": "green", "error": "orange"}

for i, category in enumerate(categories):
    category_events = df[df["category"] == category]
    plt.scatter(
        category_events["timestamp"],
        [i] * len(category_events),
        label=category,
        color=colors.get(category, "gray"),
        s=100,
    )

plt.yticks(range(len(categories)), list(categories))
plt.title("Timeline of Digital Twin Events")
plt.xlabel("Time")
plt.ylabel("Event Category")
plt.grid(True, axis="x")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/event_timeline.png")

# 5. Extract simulation IDs
simulation_events = df[df["category"] == "process"]
simulation_ids = []
for _, event in simulation_events.iterrows():
    if "simulation_id" in event["details"]:
        simulation_ids.append(event["details"]["simulation_id"])

# Create a summary report
with open(f"{output_dir}/event_analysis_summary.txt", "w") as f:
    f.write("Digital Twin Event Analysis Summary\n")
    f.write("=================================\n\n")
    f.write(f"Total events analyzed: {len(events)}\n")
    f.write(f"Event categories: {', '.join(categories)}\n")
    f.write(f"Event severities: {', '.join(df['severity'].unique())}\n\n")

    f.write("Event Distribution:\n")
    for category, count in event_counts.items():
        f.write(f"  - {category}: {count}\n")

    f.write("\nSeverity Distribution:\n")
    for severity, count in severity_counts.items():
        f.write(f"  - {severity}: {count}\n")

    f.write(f"\nTemperature threshold breaches: {len(temperature_events)}\n")
    f.write(f"Energy threshold breaches: {len(energy_events)}\n")

    f.write(f"\nSimulations completed: {len(simulation_ids)}\n")
    for sim_id in simulation_ids:
        f.write(f"  - {sim_id}\n")

print(f"Analysis complete. Results saved to {output_dir}/ directory")
