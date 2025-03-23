import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from IPython.display import display

# Load spatial data
buildings = gpd.read_file("data/processed/qc-buildings.geojson")
roads = gpd.read_file("data/processed/qc-roads.geojson")
rail = gpd.read_file("data/processed/qc-rail.geojson")

# Load agent route snapshots
snapshots = gpd.read_file("output/snapshots/agent_routes.geojson")  # or CSV + geometry

# Convert timestamp to datetime
snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"])

# Preview one time step
timestep = snapshots["timestamp"].min()
subset = snapshots[snapshots["timestamp"] == timestep]

print(f"Viewing {len(subset)} agents at time {timestep}")
display(subset.head())

# Optional: View agents from a household
hh_id = subset["household_id"].iloc[0]
display(subset[subset["household_id"] == hh_id])

fig, ax = plt.subplots(figsize=(12, 12))

# Plot context layers
buildings.plot(ax=ax, color="lightgray", linewidth=0.2, alpha=0.5)
roads.plot(ax=ax, color="gray", linewidth=0.5, alpha=0.5)
rail.plot(ax=ax, color="darkred", linewidth=1.0, alpha=0.8)

# Plot agents by mode
modes = subset["mode"].unique()
colors = plt.cm.get_cmap("tab10", len(modes))

for i, mode in enumerate(modes):
    subset[subset["mode"] == mode].plot(ax=ax, label=mode, color=colors(i), linewidth=2)

plt.legend(title="Mode")
plt.title(f"Agent Mobility Snapshot — {timestep}")
plt.axis("off")
plt.show()

# View trip segments of one agent
agent_id = subset["agent_id"].iloc[0]
agent_trips = snapshots[snapshots["agent_id"] == agent_id]

# Display full trip
agent_trips.sort_values("timestamp").plot(figsize=(10, 10), linewidth=3)
plt.title(f"Full Trip Path of Agent {agent_id}")
plt.axis("off")
plt.show()

# Optional: show cumulative travel time or cost
agent_trips[["timestamp", "mode", "travel_time", "cost"]]

# Group agent geometries for household-level animation in future versions
hh_groups = snapshots.groupby(["timestamp", "household_id"])

# Example: show one household over time
hh_id = snapshots["household_id"].iloc[0]
hh_trips = snapshots[snapshots["household_id"] == hh_id].sort_values("timestamp")
display(hh_trips)
