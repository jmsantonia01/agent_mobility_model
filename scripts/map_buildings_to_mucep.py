import geopandas as gpd
import pandas as pd

# 1. Load barangay boundaries
brgy_gdf = gpd.read_file("data/raw/qc-gdf/qc-admbnd-brgy.gpkg")
print("Barangay boundaries header:")
print(brgy_gdf.head())

# 2. Load MUCEP zone info and merge
mucep_zones = pd.read_csv("data/raw/qc-mucep/5_BrgyZones_QC.csv")
print("MUCEP zones info:")
print(mucep_zones.head())

brgy_gdf = brgy_gdf.merge(mucep_zones, left_on="ADM4_EN", right_on="Barangay Name", how="left")

# Check if all barangays matched
unmatched = brgy_gdf[brgy_gdf["MUCEPCode"].isna()]
if not unmatched.empty:
    print("Warning: Some barangays didn’t match MUCEP zones:\n", unmatched["ADM4_EN"])

# 3. Load buildings and ensure same CRS
bldgs_gdf = gpd.read_file("data/raw/qc-gdf/qc-buildings.gpkg")
bldgs_gdf = bldgs_gdf.to_crs(brgy_gdf.crs)

# 4. Spatial join: assign each building a barangay & MUCEP zone
bldgs_with_zones = gpd.sjoin(bldgs_gdf, brgy_gdf[["MUCEPCode", "geometry"]], how="left", predicate="within")

# 5. Optional: check how many were unassigned (e.g., on boundary edges)
missing_zone = bldgs_with_zones[bldgs_with_zones["MUCEPCode"].isna()]
print(f"{len(missing_zone)} buildings were not assigned to a zone.")

# 6. Save output
bldgs_with_zones.to_file("data/processed/qc-buildings-mucep.gpkg", driver="GPKG")