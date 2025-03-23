from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

#Find project root
import os
from pathlib import Path

def find_project_root(marker="README.md"):
    """Finds the project root by searching upwards for a marker file."""
    current_path = Path(__file__).resolve()
    while current_path.parent != current_path:  # Avoid infinite loop at root
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Could not find project root with marker: {marker}")

PROJECT_ROOT = find_project_root()

PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "data/output"

# --- Step 1: Sample 10% of Households ---
print("STEP 1: Sample 10% of Households")
print("🔹 Loading cleaned CPH HH and HHM data...")
hh = pd.read_csv(PROCESSED_DATA_DIR / "CPH_HH_cleaned.csv")
hhm = pd.read_csv(PROCESSED_DATA_DIR / "CPH_HHM_cleaned.csv")

# Sample 10% of households
sampled_hh = hh.sample(frac=0.10, random_state=42).copy()
print(f"✅ Sampled {len(sampled_hh)} households from CPH")

# Filter members
sampled_husn_hsn = sampled_hh[["HOUSING_UNIT_NO", "HH_NO"]].astype(str).agg("_".join, axis=1)
hhm["husn_hsn"] = hhm[["HOUSING_UNT_NO", "HH_NO"]].astype(str).agg("_".join, axis=1)
sampled_hhm = hhm[hhm["HUSN_HSN"].isin(sampled_husn_hsn)].copy()
print(f"✅ Sampled {len(sampled_hhm)} household members from CPH")

# Save intermediate sampled population
sampled_hh.to_csv(OUTPUT_DIR / "synthpop_hh_sampled.csv", index=False)
sampled_hhm.to_csv(OUTPUT_DIR / "synthpop_hhm_sampled.csv", index=False)
print("✅ Saved sampled households and members.")

tqdm.pandas()

# --- Step 2a: Enrich Households using XGBoost ---
def enrich_households_with_xgb(cph_hh, mucep_hh):
    print("🔍 STEP 2a: Enriching households with XGBoost...")

    # Features and target
    features = ["HH_SIZE", "TENURE", "FLR_AREA"]
    target = "MONTHLY_INCOME"

    # Drop NAs for supervised learning
    mucep_hh = mucep_hh.dropna(subset=[target])
    X = mucep_hh[features]
    y = mucep_hh[target]

    # Train classifier
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict on synthetic households
    cph_hh["mucep_monthly_income"] = model.predict(cph_hh[features])
    print("✅ Household enrichment complete.")
    return cph_hh


# --- Step 2b: Enrich Members using XGBoost ---
def enrich_members_with_xgb(cph_hhm, mucep_hhm, mucep_trip):
    print("🚶 STEP 2b: Enriching household members with XGBoost...")

    # Merge MUCEP member info with main trip behavior
    mucep_summary = mucep_trip.groupby(["household_no", "member_no"]).agg({
        "trip_mode_main": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        "trip_dest_code": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    }).reset_index()

    mucep_hhm = pd.merge(mucep_hhm, mucep_summary, how="left", on=["household_no", "member_no"])
    mucep_hhm = mucep_hhm.dropna(subset=["trip_mode_main"])

    # Encode label
    le = LabelEncoder()
    mucep_hhm["trip_mode_main_enc"] = le.fit_transform(mucep_hhm["trip_mode_main"])

    # Train features
    features = ["age", "sex", "occupation"]
    target = "trip_mode_main_enc"
    X = mucep_hhm[features]
    y = mucep_hhm[target]

    # Train classifier
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict on synthetic members
    cph_hhm["trip_mode_main_enc"] = model.predict(cph_hhm[features])
    cph_hhm["trip_mode_main"] = le.inverse_transform(cph_hhm["trip_mode_main_enc"])

    print("✅ People enrichment complete.")
    return cph_hhm


# Load base sampled population
cph_hh = pd.read_csv(PROCESSED_DATA_DIR / "synthpop_hh_base.csv")
cph_hhm = pd.read_csv(PROCESSED_DATA_DIR / "synthpop_hhm_base.csv")

# Load cleaned MUCEP data
mucep_hh = pd.read_csv(PROCESSED_DATA_DIR / "mucep_form1_cleaned.csv")
mucep_hhm = pd.read_csv(PROCESSED_DATA_DIR / "mucep_form2_cleaned.csv")
mucep_trip = pd.read_csv(PROCESSED_DATA_DIR / "mucep_form3_cleaned.csv")
    
# Step 2a: Enrich HHs
enriched_hh = enrich_households_with_xgb(cph_hh, mucep_hh)
    
# Step 2b: Enrich individuals
enriched_hhm = enrich_members_with_xgb(cph_hhm, mucep_hhm, mucep_trip)
    
# Save output
enriched_hh.to_csv(OUTPUT_DIR / "synthpop_hh_enriched.csv", index=False)
enriched_hhm.to_csv(OUTPUT_DIR / "synthpop_hhm_enriched.csv", index=False)
print("📦 Enriched synthetic population saved.")
