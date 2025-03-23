import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
from sklearn.neighbors import BallTree
import numpy as np

def load_stops(stops_df):
    # Convert GTFS stops to GeoDataFrame
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs="EPSG:4326"
    )
    return stops_gdf

def assign_nearest_stop(buildings_gdf, stops_gdf, label):
    # Reproject to same CRS
    buildings = buildings_gdf.to_crs("EPSG:4326").copy()
    stops = stops_gdf.to_crs("EPSG:4326").copy()

    # Use BallTree for fast nearest-neighbor search
    stop_coords = np.radians(np.array(list(zip(stops.geometry.y, stops.geometry.x))))
    building_coords = np.radians(np.array(list(zip(buildings.geometry.centroid.y, buildings.geometry.centroid.x))))
    tree = BallTree(stop_coords, metric="haversine")

    dist, idx = tree.query(building_coords, k=1)
    buildings[f"nearest_{label}_stop_id"] = stops.iloc[idx.flatten()]["stop_id"].values
    buildings[f"nearest_{label}_stop_dist_m"] = dist.flatten() * 6371000  # convert rad to meters

    return buildings

if __name__ == "__main__":
    # Load building GeoDataFrame
    buildings_gdf = gpd.read_file("data/processed/buildings_qc.gpkg")  # Adjust path as needed

    # Load GTFS stops
    road_stops = pd.read_csv("data/raw/gtfs/road/stops.txt")
    rail_stops = pd.read_csv("data/raw/gtfs/rail/stops.txt")
    road_stops_gdf = load_stops(road_stops)
    rail_stops_gdf = load_stops(rail_stops)

    # Assign nearest stops
    buildings_gdf = assign_nearest_stop(buildings_gdf, road_stops_gdf, "road")
    buildings_gdf = assign_nearest_stop(buildings_gdf, rail_stops_gdf, "rail")

    # Save with nearest stop info
    buildings_gdf.to_file("data/processed/buildings_with_stops.gpkg", driver="GPKG")
