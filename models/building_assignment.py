import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# File paths
BUILDINGS_PATH = Path("data/processed/qc_buildings_tagged.gpkg")
TRIPS_PATH = Path("data/processed/mucep_form3_cleaned.csv")
AGENTS_PATH = Path("data/processed/mucep_form2_cleaned.csv")
OUTPUT_PATH = Path("data/processed/trips_with_destination_buildings.csv")

# Load files
print("📦 Loading data...")
buildings = gpd.read_file(BUILDINGS_PATH)
trips = pd.read_csv(TRIPS_PATH)
agents = pd.read_csv(AGENTS_PATH)

# Lowercase tags
buildings["tag"] = buildings["tag"].str.lower().fillna("")

# Merge agent data into trips
trips["hh_member_key"] = trips["household_no"].astype(str) + "-" + trips["hh_member_no"].astype(str)
agents["hh_member_key"] = agents["household_no"].astype(str) + "-" + agents["hh_member_no"].astype(str)
trips = trips.merge(agents[["hh_member_key", "2_age", "6_occupation", "7_employment_sector"]], on="hh_member_key", how="left")

# Purpose-to-default-tags (fallback if no attributes available)
fallback_tags = {
    2: ["office", "commercial"],
    3: ["school", "university"],
    4: ["hospital", "clinic"],
    5: ["mall", "retail", "market"],
    6: ["church", "mosque", "temple"],
    7: ["residential"],
    8: ["restaurant", "cafe"],
    9: ["recreation", "park", "sports_centre"]
}

# Helper: choose school tag by age
def school_tag_by_age(age):
    if pd.isna(age): return "school"
    if 3 <= age <= 5: return "kindergarten"
    elif 6 <= age <= 12: return "elementary"
    elif 13 <= age <= 16: return "high_school"
    elif 17 <= age <= 22: return "university"
    else: return "school"

# Helper: occupation to tags
def work_tag_by_job(occupation, sector):
    occ = str(occupation).lower()
    sec = str(sector).lower()

    if "teacher" in occ: return ["school"]
    elif "driver" in occ or "delivery" in occ: return ["transport_terminal", "garage"]
    elif "nurse" in occ or "doctor" in occ: return ["hospital"]
    elif "cashier" in occ or "sales" in occ: return ["mall", "shop"]
    elif "police" in occ or "fire" in occ: return ["government", "station"]
    elif "it" in occ or "engineer" in occ: return ["office", "tech"]
    else:
        return ["office"] if "private" in sec else ["government"]

# Main assign function
def assign_building(row):
    zone = row["trip_dest_code"]
    purpose = row["trip_purpose"]
    age = row["2_age"]
    occ = row["6_occupation"]
    sector = row["7_employment_sector"]

    if pd.isna(zone): return None
    candidates = buildings[buildings["mucep_zone"] == zone]

    # Attribute-based filtering
    if purpose == 3:  # School
        tag = school_tag_by_age(age)
        candidates = candidates[candidates["tag"].str.contains(tag)]
    elif purpose == 2:  # Work
        tags = work_tag_by_job(occ, sector)
        candidates = candidates[candidates["tag"].isin(tags)]
    else:
        tags = fallback_tags.get(purpose, [])
        candidates = candidates[candidates["tag"].isin(tags)]

    if not candidates.empty:
        return candidates.sample(1).iloc[0]["building_id"]
    else:
        return None

# Assign destination building per trip
print("🎯 Assigning destination buildings using agent attributes...")
tqdm.pandas()
trips["dest_building_id"] = trips.progress_apply(assign_building, axis=1)

# Save output
trips.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved with attributes to: {OUTPUT_PATH}")
