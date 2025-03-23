from shapely.geometry import Point
import networkx as nx
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

def is_walkable(geom1, geom2, threshold=500):
    return geom1.distance(geom2) <= threshold

def compute_agent_paths_with_transfers(agent_trips_df, buildings_gdf, G_road, G_rail):
    paths = []

    # Find transfer nodes (shared stop_ids in both graphs)
    transfer_stops = set(G_road.nodes).intersection(set(G_rail.nodes))

    for _, row in tqdm(agent_trips_df.iterrows(), total=len(agent_trips_df)):
        agent_id = row["agent_id"]
        trip_no = row["trip_no"]
        origin_id = row["origin_building_id"]
        dest_id = row["destination_building_id"]
        preferred_mode = row["preferred_mode"]

        origin_geom = buildings_gdf.loc[origin_id].geometry
        dest_geom = buildings_gdf.loc[dest_id].geometry

        # --- Case 1: Walkable trip ---
        if is_walkable(origin_geom, dest_geom):
            paths.append({
                "agent_id": agent_id,
                "trip_no": trip_no,
                "mode": "walk",
                "stop_sequence": []
            })
            continue

        # Get nearest stops
        origin_road = buildings_gdf.loc[origin_id]["nearest_road_stop_id"]
        origin_rail = buildings_gdf.loc[origin_id]["nearest_rail_stop_id"]
        dest_road = buildings_gdf.loc[dest_id]["nearest_road_stop_id"]
        dest_rail = buildings_gdf.loc[dest_id]["nearest_rail_stop_id"]

        try:
            # --- Case 2: Single-mode trip ---
            if preferred_mode == "road":
                path = nx.shortest_path(G_road, origin_road, dest_road, weight="length")
                mode_used = "road"
            elif preferred_mode == "rail":
                path = nx.shortest_path(G_rail, origin_rail, dest_rail, weight="length")
                mode_used = "rail"
            else:
                # --- Case 3: Multi-modal with transfer ---
                min_total_length = float("inf")
                best_path = []
                for transfer in transfer_stops:
                    try:
                        part1 = nx.shortest_path(G_road, origin_road, transfer, weight="length")
                        part2 = nx.shortest_path(G_rail, transfer, dest_rail, weight="length")
                        total_length = (
                            nx.path_weight(G_road, part1, "length") +
                            nx.path_weight(G_rail, part2, "length")
                        )
                        if total_length < min_total_length:
                            min_total_length = total_length
                            best_path = part1 + part2[1:]  # avoid duplicate transfer stop
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                path = best_path
                mode_used = "mixed"
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = []
            mode_used = "none"

        paths.append({
            "agent_id": agent_id,
            "trip_no": trip_no,
            "mode": mode_used,
            "stop_sequence": path
        })

    return pd.DataFrame(paths)

from shapely.geometry import LineString
import numpy as np

def get_node_geometry(G, node_id):
    node = G.nodes[node_id]
    return Point(node["x"], node["y"]) if "x" in node else node["geometry"]

def sequence_to_linestring(path, G_road, G_rail):
    coords = []
    for node_id in path:
        if node_id in G_road:
            coords.append(get_node_geometry(G_road, node_id).coords[0])
        elif node_id in G_rail:
            coords.append(get_node_geometry(G_rail, node_id).coords[0])
    if len(coords) < 2:
        return None
    return LineString(coords)

def estimate_travel_time(length_m, mode):
    speed_kph = {
        "walk": 5,
        "road": 20,
        "rail": 30,
        "mixed": 15,
        "none": 0
    }
    speed_mps = (speed_kph[mode] * 1000) / 3600
    return round(length_m / speed_mps / 60, 2) if speed_mps > 0 else None

def compute_agent_paths_geometries(agent_trips_df, buildings_gdf, G_road, G_rail):
    from tqdm import tqdm
    records = []

    transfer_stops = set(G_road.nodes).intersection(set(G_rail.nodes))

    for _, row in tqdm(agent_trips_df.iterrows(), total=len(agent_trips_df)):
        agent_id = row["agent_id"]
        trip_no = row["trip_no"]
        origin_id = row["origin_building_id"]
        dest_id = row["destination_building_id"]
        preferred_mode = row["preferred_mode"]

        origin_geom = buildings_gdf.loc[origin_id].geometry
        dest_geom = buildings_gdf.loc[dest_id].geometry

        if origin_geom.distance(dest_geom) <= 500:
            line = LineString([origin_geom, dest_geom])
            length = line.length
            mode = "walk"
        else:
            origin_road = buildings_gdf.loc[origin_id]["nearest_road_stop_id"]
            origin_rail = buildings_gdf.loc[origin_id]["nearest_rail_stop_id"]
            dest_road = buildings_gdf.loc[dest_id]["nearest_road_stop_id"]
            dest_rail = buildings_gdf.loc[dest_id]["nearest_rail_stop_id"]

            path = []
            mode = "none"
            try:
                if preferred_mode == "road":
                    path = nx.shortest_path(G_road, origin_road, dest_road, weight="length")
                    mode = "road"
                elif preferred_mode == "rail":
                    path = nx.shortest_path(G_rail, origin_rail, dest_rail, weight="length")
                    mode = "rail"
                else:
                    best_path, min_len = [], float("inf")
                    for transfer in transfer_stops:
                        try:
                            p1 = nx.shortest_path(G_road, origin_road, transfer, weight="length")
                            p2 = nx.shortest_path(G_rail, transfer, dest_rail, weight="length")
                            total_len = nx.path_weight(G_road, p1, "length") + nx.path_weight(G_rail, p2, "length")
                            if total_len < min_len:
                                best_path = p1 + p2[1:]
                                min_len = total_len
                                mode = "mixed"
                        except:
                            continue
                    path = best_path

                line = sequence_to_linestring(path, G_road, G_rail)
                length = line.length if line else None

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                line = None
                length = None

        records.append({
            "agent_id": agent_id,
            "trip_no": trip_no,
            "mode": mode,
            "length_m": length,
            "estimated_time_min": estimate_travel_time(length, mode) if length else None,
            "geometry": line
        })

    return gpd.GeoDataFrame(records, geometry="geometry", crs=buildings_gdf.crs)

if __name__ == "__main__":
    agent_trips_df = pd.read_parquet("data/processed/agent_trips.parquet")
    buildings_gdf = gpd.read_file("data/processed/buildings_with_stops.gpkg").set_index("building_id")

    # Load transport graphs
    import pickle
    with open("data/processed/graph_road.pkl", "rb") as f:
        G_road = pickle.load(f)
    with open("data/processed/graph_rail.pkl", "rb") as f:
        G_rail = pickle.load(f)

    agent_paths_df = compute_agent_paths_with_transfers(agent_trips_df, buildings_gdf, G_road, G_rail)

    agent_paths_df.to_parquet("data/processed/agent_trip_paths.parquet", index=False)