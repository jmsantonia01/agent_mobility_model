import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os

movement_snapshots = []

def interpolate_position(route: LineString, progress: float):
    return route.interpolate(progress * route.length)

def generate_movement_snapshot(agent_states, current_time):
    rows = []

    for agent_id, state in agent_states.items():
        if not state.is_traveling or state.route is None:
            continue

        route = state.route
        total_trip_time = state.current_trip_duration
        time_in_trip = (current_time - state.trip_start_time).total_seconds() / 60

        progress = min(1.0, max(0.0, time_in_trip / total_trip_time)) if total_trip_time > 0 else 1.0
        location = interpolate_position(route, progress)

        rows.append({
            "agent_id": agent_id,
            "household_id": state.household_id,
            "mode": state.current_mode,
            "segment_index": state.current_segment,
            "origin_stop_id": state.origin_stop_id,
            "destination_stop_id": state.destination_stop_id,
            "segment_mode": state.segment_mode,
            "segment_time": state.segment_duration,
            "trip_start_time": state.trip_start_time.strftime("%H:%M"),
            "time": current_time.strftime("%H:%M"),
            "geometry": location
        })

    if rows:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        movement_snapshots.append((current_time.strftime("%H%M"), gdf))

        # Optional: save debug map image for the time step
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, color='blue', markersize=5)
        ax.set_title(f"Agent Snapshot @ {current_time.strftime('%H:%M')}")
        plt.axis('off')
        plt.savefig(f"outputs/animation_snapshots/map_{current_time.strftime('%H%M')}.png", dpi=150)
        plt.close()

def export_snapshots_to_geojson(output_dir="outputs/animation_snapshots"):
    os.makedirs(output_dir, exist_ok=True)

    for timestamp, gdf in movement_snapshots:
        gdf.to_file(f"{output_dir}/agents_{timestamp}.geojson", driver="GeoJSON")

    print(f"Exported {len(movement_snapshots)} snapshots to {output_dir}")


import os

output_dir = "outputs/animation_snapshots"
os.makedirs(output_dir, exist_ok=True)

for idx, gdf in enumerate(movement_snapshots):
    timestamp = gdf.iloc[0]["time"].replace(":", "")
    gdf.to_file(f"{output_dir}/agents_{timestamp}.geojson", driver="GeoJSON")
