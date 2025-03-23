import geopandas as gpd
import pandas as pd

# Load buildings
buildings = gpd.read_file("data/raw/qc-gdf/qc-buildings.gpkg")

# Filter by relevant tags (e.g., non-residential for destination candidates)
target_tags = ['office', 'commercial', 'school', 'college', 'university', 'industrial', 'shop']
buildings['building_type'] = buildings['building'].fillna(buildings['amenity'])

# Load MUCEP zones shapefile or join with buildings (if not yet done)
zones = gpd.read_file("data/raw/qc-mucep/5_BrgyZones_QC.csv")

# Spatial join to assign MUCEP code to each building
buildings = gpd.sjoin(buildings, zones[['MUCEPCode', 'geometry']], how='left', predicate='intersects')

trip_df = pd.read_csv("data/raw/qc-mucep/3_Trip.csv")

# Keep useful columns
trip_df = trip_df[['Household_No', 'HH_Member_No', 'Trip_No', '4_2_Origin', '8_2_Destination', '9']]

# You can map '9' (trip purpose) to descriptive labels here if needed

import numpy as np

def assign_destination_building(agent_row, buildings_df):
    zone_code = agent_row['8_2_Destination']
    trip_purpose = agent_row['9']
    
    # Get valid building types
    if trip_purpose == 2:  # Work
        valid_types = ['office', 'commercial', 'industrial', 'shop']
    elif trip_purpose == 3:  # School
        valid_types = ['school', 'college', 'university']
    elif trip_purpose == 1:  # Home
        return agent_row.get("home_building_id")  # Assigned earlier
    else:
        valid_types = ['hospital', 'place_of_worship', 'leisure', 'mall']  # Extendable
    
    # Filter candidate buildings
    candidates = buildings_df[
        (buildings_df['MUCEPCode'] == zone_code) &
        (buildings_df['building_type'].isin(valid_types))
    ]
    
    if candidates.empty:
        return np.nan  # fallback logic later
    else:
        selected = candidates.sample(1, random_state=42)
        return selected.index.values[0]  # or selected['building_id'].values[0] if exists

trip_df['destination_building_id'] = trip_df.apply(
    lambda row: assign_destination_building(row, buildings), axis=1
)

trip_df.to_csv("data/processed/trip_with_buildings.csv", index=False)
