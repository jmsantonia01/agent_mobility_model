from pathlib import Path
import pandas as pd

# === Setup ===
RAW_DIR = Path("data/raw/qc-mucep")
CLEAN_DIR = Path("data/clean/qc-mucep")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# === Load Trip Data ===
form3_path = RAW_DIR / "3_Trip.csv"
form3 = pd.read_csv(form3_path, low_memory=False)

# === Load MUCEP QC Zone Reference ===
zones_path = RAW_DIR / "5_BrgyZones_QC.csv"
zones = pd.read_csv(zones_path)
valid_mucep_codes = set(zones["MUCEPCode"].astype(str))

# === Filter QC Households ===
form3["household_no"] = form3["Household_No"]
form3["member_no"] = form3["HH_Member_No"]
form3["origin_code"] = form3["4_2_Origin"].astype(str)
form3["dest_code"] = form3["8_2_Destination"].astype(str)

qc_form3 = form3[
    form3["origin_code"].isin(valid_mucep_codes) | form3["dest_code"].isin(valid_mucep_codes)
].copy()

# === Rename transport mode fields ===
mode_legs = [1, 2, 3, 4]
rename_map = {}
for i, col_base in zip(mode_legs, [1, 5, 9, 13]):
    rename_map[f"6_{col_base}"] = f"mode_{i}_code"
    rename_map[f"6_{col_base}_Others"] = f"mode_{i}_other"

qc_form3.rename(columns=rename_map, inplace=True)

# === Transport Mode Lookup Dictionary ===
MUCEP_MODE_LOOKUP = {
    1: {"mode_name": "Jeepney", "group": "Public", "type": "Road"},
    2: {"mode_name": "Bus", "group": "Public", "type": "Road"},
    3: {"mode_name": "UV Express", "group": "Public", "type": "Road"},
    4: {"mode_name": "Tricycle", "group": "Public", "type": "Road"},
    5: {"mode_name": "Motorcycle Taxi", "group": "Public", "type": "Road"},
    6: {"mode_name": "Taxi", "group": "Public", "type": "Road"},
    7: {"mode_name": "TNVS", "group": "Public", "type": "Road"},
    8: {"mode_name": "School Service", "group": "Public", "type": "Road"},
    9: {"mode_name": "Shuttle", "group": "Public", "type": "Road"},
    10: {"mode_name": "Carpool", "group": "Private", "type": "Road"},
    11: {"mode_name": "Private Car", "group": "Private", "type": "Road"},
    12: {"mode_name": "Motorcycle", "group": "Private", "type": "Road"},
    13: {"mode_name": "Bicycle", "group": "Active", "type": "Non-Motor"},
    14: {"mode_name": "Walking", "group": "Active", "type": "Non-Motor"},
    15: {"mode_name": "MRT-3", "group": "Public", "type": "Rail"},
    16: {"mode_name": "LRT-1", "group": "Public", "type": "Rail"},
    17: {"mode_name": "LRT-2", "group": "Public", "type": "Rail"},
    18: {"mode_name": "PNR", "group": "Public", "type": "Rail"},
    19: {"mode_name": "Ferry", "group": "Public", "type": "Water"},
    20: {"mode_name": "Airplane", "group": "Public", "type": "Air"},
    21: {"mode_name": "Scooter", "group": "Active", "type": "Non-Motor"},
    22: {"mode_name": "Animal-drawn", "group": "Others", "type": "Road"},
    23: {"mode_name": "Truck", "group": "Private", "type": "Road"},
    24: {"mode_name": "Other", "group": "Others", "type": "Unknown"},
    25: {"mode_name": "Other", "group": "Others", "type": "Unknown"},
    26: {"mode_name": "Other", "group": "Others", "type": "Unknown"},
    27: {"mode_name": "Other", "group": "Others", "type": "Unknown"},
}

# === Apply Lookup Table to Enrich Mode Info ===
for i in mode_legs:
    code_col = f"mode_{i}_code"
    qc_form3[f"mode_{i}_name"] = qc_form3[code_col].map(lambda x: MUCEP_MODE_LOOKUP.get(x, {}).get("mode_name", "Unknown"))
    qc_form3[f"mode_{i}_group"] = qc_form3[code_col].map(lambda x: MUCEP_MODE_LOOKUP.get(x, {}).get("group", "Unknown"))
    qc_form3[f"mode_{i}_type"] = qc_form3[code_col].map(lambda x: MUCEP_MODE_LOOKUP.get(x, {}).get("type", "Unknown"))

# === Save Cleaned Trip Form ===
qc_form3.to_csv(CLEAN_DIR / "3_Trip_Cleaned.csv", index=False)

# === Save Lookup as CSV ===
mode_lookup_df = pd.DataFrame.from_dict(MUCEP_MODE_LOOKUP, orient="index").reset_index().rename(columns={"index": "mode_code"})
mode_lookup_df.to_csv(CLEAN_DIR / "mode_lookup_mucep.csv", index=False)

qc_form3.shape, mode_lookup_df.shape
