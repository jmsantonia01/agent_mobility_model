import pandas as pd
import networkx as nx
from pathlib import Path
from geopy.distance import geodesic
from tqdm import tqdm

GTFS_DIR = Path("data/raw/gtfs")

def load_gtfs(gtfs_path):
    print(f"📂 Loading GTFS from {gtfs_path.name}...")
    stops = pd.read_csv(gtfs_path / "stops.txt")
    stop_times = pd.read_csv(gtfs_path / "stop_times.txt")
    trips = pd.read_csv(gtfs_path / "trips.txt")
    routes = pd.read_csv(gtfs_path / "routes.txt")

    # Merge to get route info per stop time
    stop_times = stop_times.merge(trips, on="trip_id")
    stop_times = stop_times.merge(routes, on="route_id")
    return stops, stop_times

def build_graph(stops, stop_times):
    print("🛠️ Building transport network graph...")
    G = nx.DiGraph()

    # Add nodes with lat/lon
    for _, row in stops.iterrows():
        G.add_node(row["stop_id"], lat=row["stop_lat"], lon=row["stop_lon"], name=row["stop_name"])

    # Sort and add edges by trip sequence
    trip_groups = stop_times.groupby("trip_id")
    for trip_id, group in tqdm(trip_groups, desc="⏱️ Adding trip edges"):
        group = group.sort_values("stop_sequence")
        prev_stop = None
        for _, row in group.iterrows():
            curr_stop = row["stop_id"]
            if prev_stop is not None:
                coord1 = (stops.loc[stops["stop_id"] == prev_stop, "stop_lat"].values[0],
                          stops.loc[stops["stop_id"] == prev_stop, "stop_lon"].values[0])
                coord2 = (stops.loc[stops["stop_id"] == curr_stop, "stop_lat"].values[0],
                          stops.loc[stops["stop_id"] == curr_stop, "stop_lon"].values[0])
                distance = geodesic(coord1, coord2).meters
                G.add_edge(prev_stop, curr_stop, weight=distance, route=row["route_id"])
            prev_stop = curr_stop

    print(f"✅ Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

if __name__ == "__main__":
    # Build road network
    road_gtfs_path = GTFS_DIR / "road"
    road_stops, road_stop_times = load_gtfs(road_gtfs_path)
    road_graph = build_graph(road_stops, road_stop_times)
    nx.write_gpickle(road_graph, "data/processed/graph_road.gpickle")
    
    # Build rail network
    rail_gtfs_path = GTFS_DIR / "rail"
    rail_stops, rail_stop_times = load_gtfs(rail_gtfs_path)
    rail_graph = build_graph(rail_stops, rail_stop_times)
    nx.write_gpickle(rail_graph, "data/processed/graph_rail.gpickle")
