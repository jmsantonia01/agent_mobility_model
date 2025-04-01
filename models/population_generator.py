from xgboost import XGBClassifier, XGBRegressor
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
hhm["HUSN_HSN"] = hhm[["HOUSING_UNIT_NO", "HH_NO"]].astype(str).agg("_".join, axis=1)
sampled_hhm = hhm[hhm["HUSN_HSN"].isin(sampled_husn_hsn)].copy()
print(f"✅ Sampled {len(sampled_hhm)} household members from CPH")

# Save intermediate sampled population
sampled_hh.to_csv(OUTPUT_DIR / "synthpop_hh_base.csv", index=False)
sampled_hhm.to_csv(OUTPUT_DIR / "synthpop_hhm_base.csv", index=False)
print("✅ Saved sampled households and members.")

tqdm.pandas()

# --- Step 2a: Enrich Households using XGBoost ---
def enrich_households_with_xgb(cph_hh, mucep_hh):
    print("🔍 STEP 2a: Enriching households with XGBoost...")

    # Mapping dictionary (MUCEP -> CPH)
    column_mapping = {
        'hh_size': 'HH_SIZE',
        '6_occupation': 'MAX_EDUC_LEVEL',
        'avg_age': 'AVG_AGE',
        'male_prop': 'MALE_PROPORTION',
        '7_1_house_ownership': 'OWNERSHIP'
    }

    # Print CPH columns for debugging
    print("Columns in cph_hh:", cph_hh.columns)
    print("Columns in mucep_hh:", mucep_hh.columns)
    print("Unique values in 6_occupation:", mucep_hhm['6_occupation'].unique())

    # Rename CPH columns to match MUCEP (before prediction)
    # Create a copy to modify
    cph_hh_for_prediction = cph_hh.copy()

    # Education Mapping (CPH -> MUCEP Occupation)
    education_to_occupation_map = {
    "No Education": [9, 13],  # Laborer/Unskilled, Unemployed
    "Preschool": [10],  # Student (Elem.)
    "Grade 1": [10],
    "Grade 2": [10],
    "Grade 3": [10],
    "Grade 4": [10],
    "Grade 5": [10],
    "Grade 6": [10],
    "Grade 7": [11], # Student (H.S. & Univ.)
    "Grade 8": [11],
    "Grade 9": [11],  
    "Grade 10": [11],
    "Grade 11": [11],
    "Grade 12": [4, 5, 9, 13],  # Clerical, Service Worker, Laborer, Unemployed
    "High School": [4, 5, 9, 12, 13],  # Clerical, Service Worker, Laborer, Housewife, Unemployed
    "High School Graduate": [4, 5, 7, 9, 12, 13],  # Clerical, Service Worker, Trader, Laborer, Housewife, Unemployed
    "Special Education": [10, 11],  # Student
    "Alternative Education": [10, 11, 9, 13],  # Student, Laborer, Unemployed
    "Post-Secondary": [3, 4, 5, 7, 8],  # Technical, Clerical, Service, Trader, Plant/Machine
    "Tertiary": [2, 3, 1],  # Professional, Technical, Official/Manager
    "College": [2, 3, 1],  # Professional, Technical, Official/Manager
    "Bachelor's Degree": [1, 2, 3],  # Official/Manager, Professional, Technical
    "Master's Degree": [1, 2, 3],  # Official/Manager, Professional, Technical
    "Doctorate Degree": [1, 2],  # Official/Manager, Professional
    }

    # Reverse the mapping for prediction (MUCEP Occupation -> CPH Education)
    occupation_to_education_map = {}
    for education, occupations in education_to_occupation_map.items():
        for occupation in occupations:
            if occupation not in occupation_to_education_map:
                occupation_to_education_map[occupation] = education
            else:
                # Handle cases where an occupation maps to multiple educations (e.g., choose the most common)
                # For simplicity, we'll keep the first mapping here
                pass

    # Rename CPH columns individually (more robust)
    for mucep_col, cph_col in column_mapping.items():
        if cph_col in cph_hh_for_prediction.columns:
            cph_hh_for_prediction = cph_hh_for_prediction.rename(
                columns={cph_col: mucep_col}, errors='ignore'
            )
    print("Columns in cph_hh_for_prediction (after mapping):", cph_hh_for_prediction.columns)

    # Features and target (USE MUCEP COLUMN NAMES!)
    features = ['hh_size', 'max_educ_level', 'avg_age', 'male_prop',
                '7_1_house_ownership']
    target = "4_monthly_hh_income"

    # Drop NaNs
    mucep_hh_clean = mucep_hh.dropna(subset=[target] + features).copy()
    if mucep_hh_clean.empty:
        print("⚠️ WARNING: No MUCEP data available for training. Income prediction skipped.")
        cph_hh["predicted_income"] = np.nan
        return cph_hh

    X = mucep_hh_clean[features].copy()
    y = (mucep_hh_clean[target] - 1).astype(int)

    # Train income prediction model
    income_model = XGBClassifier(n_estimators=100, random_state=42, objective="multi:softmax",
                                num_class=len(y.unique()))
    income_model.fit(X, y)

    # Predict income on synthetic households
    pred_features = cph_hh_for_prediction[features].copy()
    for col in features:
        if col not in pred_features:
            pred_features[col] = 0
    cph_hh["predicted_income"] = income_model.predict(pred_features[features]) + 1

     # Vehicle ownership prediction
    vehicle_cols = [col for col in mucep_hh.columns if "5_" in col and "_owned" in col]
    if vehicle_cols:
        for vcol in vehicle_cols:
            if vcol in mucep_hh:
                print(f"🔍 Predicting {vcol} (Vehicle Count)...")
                print(f"Unique values in {vcol}:", mucep_hh[vcol].unique())
                print(f"Data type of {vcol}:", mucep_hh[vcol].dtype)

                # Ensure vehicle count column is integer type
                mucep_hh[vcol] = mucep_hh[vcol].astype(int)

                vehicle_model = XGBRegressor(n_estimators=100, random_state=42,
                                            objective="reg:squarederror")  # Use regressor
                mucep_veh = mucep_hh[features + [vcol]].dropna(subset=features + [vcol]).copy()
                if not mucep_veh.empty:
                    vehicle_model.fit(mucep_veh[features], mucep_veh[vcol])
                    cph_hh[vcol] = vehicle_model.predict(cph_hh[features]).astype(int) # Predict and convert to int
                else:
                    print(f"⚠️ WARNING: No MUCEP data to predict {vcol}. Skipping.")
                    cph_hh[vcol] = np.nan
            else:
                print(f"⚠️ WARNING: Vehicle column {vcol} not found in MUCEP data.")
                cph_hh[vcol] = np.nan

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
cph_hh = pd.read_csv(OUTPUT_DIR / "synthpop_hh_base.csv")
cph_hhm = pd.read_csv(OUTPUT_DIR / "synthpop_hhm_base.csv")

# Load cleaned MUCEP data
mucep_hh = pd.read_csv(PROCESSED_DATA_DIR / "mucep_form1_qc.csv")
mucep_hhm = pd.read_csv(PROCESSED_DATA_DIR / "mucep_form2_qc.csv")
mucep_trip = pd.read_csv(PROCESSED_DATA_DIR / "mucep_form3_qc.csv")
    
# Step 2a: Enrich HHs
enriched_hh = enrich_households_with_xgb(cph_hh, mucep_hh)
    
# Step 2b: Enrich individuals
enriched_hhm = enrich_members_with_xgb(cph_hhm, mucep_hhm, mucep_trip)
    
# Save output
enriched_hh.to_csv(OUTPUT_DIR / "synthpop_hh_enriched.csv", index=False)
enriched_hhm.to_csv(OUTPUT_DIR / "synthpop_hhm_enriched.csv", index=False)
print("📦 Enriched synthetic population saved.")
