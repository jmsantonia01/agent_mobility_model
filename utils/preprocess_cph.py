# utils/preprocess_cph.py

import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw/qc-cph")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Special values to drop
SPECIAL_VALUES = {777777, 888888, 999999}

# Mapping of columns for HH and HHM
HH_RENAME = {
    "B1": "RES_TYPE", "B2": "FLR_NUM", "D1": "FLR_AREA", "B8": "YR_BUILT",
    "B3": "WALL_MAT", "B4": "ROOF_MAT", "B5": "WALL_COND", "B6": "ROOF_COND",
    "B7": "STRUCT_COND", "H1": "OWNERSHIP", "H11A": "OCCUP_1", "H11B": "OCCUP_2",
    "H11C": "OCCUP_3", "H11D": "OCCUP_4", "H13": "LANGUAGE", "H14_PRVMUN": "PREV_LOC",
    "H14_RECODE": "PREV_LOC_RECODE", "HUSN": "HOUSING_UNIT_NO", "HSN": "HH_NO", "REG": "REGION_CODE",
    "PRV": "PROV_CODE", "MUN": "MUN_CODE", "BGY": "BRGY_CODE", "URB": "URBAN_RURAL"
}

HHM_RENAME = {
    "HUSN": "HOUSING_UNIT_NO", "HSN": "HH_NO", "LNA": "MEMBER_NO", "P2": "REL_TO_HEAD",
    "P3": "SEX", "P5": "AGE", "P8": "MARITAL_STATUS", "P9": "RELIGION",
    "P10": "CITIZENSHIP1", "P11": "CITIZENSHIP2", "P12": "CITIZENSHIP3",
    "P13A": "DISABILITY_A", "P13B": "DISABILITY_B", "P13C": "DISABILITY_C",
    "P13D": "DISABILITY_D", "P13E": "DISABILITY_E", "P13F": "DISABILITY_F",
    "P14_PRVMUN": "PREV_LOC", "P14_RECODE": "PREV_LOC_RECODE",
    "P15_PRVMUN": "BIRTH_LOC", "P15_RECODE": "BIRTH_LOC_RECODE",
    "P16": "LITERACY", "P17": "EDUC_LEVEL", "P20": "OFW_STATUS"
}

def remove_special_values(df):
    """Remove rows with any special value in any column."""
    return df[~df.isin(SPECIAL_VALUES).any(axis=1)]

def generate_psgc(row):
    """Generate 10-digit PSGC code from region, province, municipality/city, and barangay."""
    return f"{int(row['REGION_CODE']):02d}{int(row['PROV_CODE']):03d}{int(row['MUN_CODE']):02d}{int(row['BRGY_CODE']):03d}"

def preprocess_cph():
    # Load raw data
    print("Loading CPH data...")
    hh = pd.read_csv(RAW_DIR / "CPH-PUF-2020-QC-HH.CSV")
    hhm = pd.read_csv(RAW_DIR / "CPH-PUF-2020-QC-HHM.CSV")
    print(f"Loaded {len(hh)} households and {len(hhm)} members.")

    # Drop special value rows
    print("Cleaning CPH data...")
    hh = remove_special_values(hh)
    hhm = hhm[hhm["HSN"] != 999999]  # HSN == 999999 is explicitly noted as invalid
    print(f"Cleaned to {len(hh)} households and {len(hhm)} members.")

    # Rename columns
    print("Renaming columns...")
    hh = hh.rename(columns=HH_RENAME)
    hhm = hhm.rename(columns=HHM_RENAME)
    print("Columns renamed.")

    # -- Before merging PSGC to HHM, ensure household keys are unique --
    print("🔍 Checking for duplicate household keys in HH data...")
    dupes = hh.duplicated(subset=["HOUSING_UNIT_NO", "HH_NO"], keep=False)
    if dupes.any():
        print(f"⚠️ Found {dupes.sum()} duplicated household entries on [HOUSING_UNIT_NO + HH_NO]. Deduplicating before merge...")
    else:
        print("✅ Household keys are unique. Proceeding with merge.")

    # Generate PSGC
    print("Assigning Barangay PSGC codes...")
    hh["PSGC"] = hh.apply(generate_psgc, axis=1)
    print("Barangay PSGC codes assigned.")

    # PSGC assignment per household
    print("🔍 Aggregating PSGC per household...")

    # Take the first (or most common) PSGC per [HOUSING_UNIT_NO, HH_NO]
    hh_psgc = (
        hh.groupby(["HOUSING_UNIT_NO", "HH_NO"])["PSGC"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )

    print(f"✅ Aggregated PSGC for {len(hh_psgc)} unique households.")

    # Merge into HHM
    print("📌 Merging PSGC codes into HHM...")
    print("Before merge:", hhm.shape)
    hhm_old = copy.deepcopy(hhm.shape[0])
    hhm = hhm.merge(hh_psgc, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")
    print("After merge:", hhm.shape)
    if hhm.shape[0] != hhm_old:
        print(f"⚠️ Merging resulted in {hhm_old - hhm.shape[0]} duplicate rows.")
        sys.exit(1)

    # Generate sequential Agent IDs
    print(f"Generating agent IDs for {len(hhm)} members...")
    hhm = hhm.sort_values(by=["PSGC", "HOUSING_UNIT_NO", "HH_NO", "MEMBER_NO"]).reset_index(drop=True)
    hhm["AGENT_ID"] = hhm.index + 1
    print("Agent IDs generated.")

    # --- Calculate HH_SIZE for CPH ---
    print("Calculating HH_SIZE for CPH...")
    hh_size_cph = hhm.groupby(["HOUSING_UNIT_NO", "HH_NO"]).size().reset_index(name="HH_SIZE")
    hh = pd.merge(hh, hh_size_cph, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")
    print("✅ HH_SIZE for CPH calculated.")

    # Export cleaned data
    print("Saving cleaned data...")
    hh.to_csv(OUT_DIR / "CPH_HH_cleaned.csv", index=False)
    hhm.to_csv(OUT_DIR / "CPH_HHM_cleaned.csv", index=False)

    print("✅ CPH HH and HHM cleaned and saved to data/processed")

if __name__ == "__main__":
    preprocess_cph()
