import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_CPH_DIR, RAW_MUCEP_DIR, PROCESSED_DATA_DIR #Use the specific CPH raw data directory

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
    "REG": "REGION_CODE",
    "PRV": "PROV_CODE", "MUN": "MUN_CODE", "BGY": "BRGY_CODE", "URB": "URBAN_RURAL", 
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
    try:
        # Load raw data
        print("Loading CPH data...")
        hh = pd.read_csv(RAW_CPH_DIR / "CPH-PUF-2020-QC-HH.CSV")
        hhm = pd.read_csv(RAW_CPH_DIR / "CPH-PUF-2020-QC-HHM.CSV")
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
        print(list(hh.columns.values))
        print(list(hhm.columns.values))

        # Generate PSGC
        print("Assigning Barangay PSGC codes...")
        hh["PSGC"] = hh.apply(generate_psgc, axis=1)
        hhm["PSGC"] = hhm.apply(generate_psgc, axis=1) #Generate PSGC on HHM as well
        print("Barangay PSGC codes assigned.")

                # PSGC assignment per household
        print("🔍 Aggregating PSGC per household...")
        # Take the first (or most common) PSGC per [HOUSING_UNIT_NO, HH_NO]
        hh_psgc = (
            hh.groupby(["PSGC", "HOUSING_UNIT_NO", "HH_NO"])["PSGC"]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
            .reset_index(name = "PSGC_AGG") # change PSGC to PSGC_AGG
        )
        print(f"✅ Aggregated PSGC for {len(hh_psgc)} unique households.")

        # Remove household members from removed households
        print("Removing household members from removed households...")
        removed_hh_ids = hh[["HOUSING_UNIT_NO", "HH_NO", "PSGC"]]
        hhm = pd.merge(hhm, removed_hh_ids, on=["HOUSING_UNIT_NO", "HH_NO", "PSGC"], how="inner")
        print("Household members removed.")

        # -- Ensure household keys are unique --
        print("🔍 Checking for duplicate household keys in HH data...")
        dupes = hh.duplicated(subset=["PSGC", "HOUSING_UNIT_NO", "HH_NO"], keep=False)
        if dupes.any():
            print(f"⚠️ Found {dupes.sum()} duplicated household entries on [PSGC, HOUSING_UNIT_NO + HH_NO]. Deduplicating before merge...")
        else:
            print("✅ Household keys are unique. Proceeding with merge.")
        
            # --- Calculate HH_SIZE for CPH ---
        print("Calculating HH_SIZE for CPH...")
        hh_size_cph = hhm.groupby(["HOUSING_UNIT_NO", "HH_NO"]).size().reset_index(name="HH_SIZE")
        hh = pd.merge(hh, hh_size_cph, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")
        print("✅ HH_SIZE for CPH calculated.")

        # Merge into HHM
        print("📌 Merging PSGC codes into HHM...")
        print("Before merge:", hhm.shape)
        hhm_old = copy.deepcopy(hhm.shape[0])
        hhm = hhm.merge(hh_psgc, left_on = ["PSGC", "HOUSING_UNIT_NO", "HH_NO"], right_on = ["PSGC", "HOUSING_UNIT_NO", "HH_NO"], how="left") #Merge using the new PSGC column
        hhm = hhm.drop(columns = ["PSGC"]) #drop the old PSGC column.
        hhm = hhm.rename(columns = {"PSGC_AGG":"PSGC"}) #rename the new column to PSGC.
        print("After merge:", hhm.shape)
        if hhm.shape[0] != hhm_old:
            print(f"⚠️ Merging resulted in {hhm_old - hhm.shape[0]} duplicate rows.")
            sys.exit(1)

        # Generate sequential Agent IDs
        print(f"Generating agent IDs for {len(hhm)} members...")
        hhm = hhm.sort_values(by=["PSGC", "HOUSING_UNIT_NO", "HH_NO", "MEMBER_NO"]).reset_index(drop=True)
        hhm["AGENT_ID"] = hhm.index + 1
        print("Agent IDs generated.")

        # Export cleaned data
        print("Saving cleaned data...")
        hh.to_csv(PROCESSED_DATA_DIR / "CPH_HH_cleaned.csv", index=False)
        hhm.to_csv(PROCESSED_DATA_DIR / "CPH_HHM_cleaned.csv", index=False)

        print("✅ CPH HH and HHM cleaned and saved to data/processed")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

# Add MUCEP Preprocessing here.
def preprocess_mucep():
    try:
        # Load QC MUCEP codes for filtering
        ref = pd.read_csv(RAW_MUCEP_DIR / "5_BrgyZones_QC.csv")
        qc_codes = ref["MUCEPCode"].dropna().unique().tolist()

        # ========== Form 1: Household Information ==========
        form1 = pd.read_csv(RAW_MUCEP_DIR / "1_HH.csv")
        form1.columns = form1.columns.str.strip().str.lower()
        qc_form1 = form1[form1["2_address_mucep_code"].isin(qc_codes)].copy()
        qc_form1 = qc_form1.drop_duplicates(subset=["household_no"])

        # Extract household numbers from filtered Form 1
        qc_hh_numbers = qc_form1["household_no"].tolist()

        # ========== Form 2: Household Members ==========
        form2 = pd.read_csv(RAW_MUCEP_DIR / "2_HHM.csv")
        form2.columns = form2.columns.str.strip().str.lower()
        qc_form2 = form2[form2["household_no"].isin(qc_hh_numbers)].copy()
        qc_form2 = qc_form2.drop_duplicates(subset=["household_no", "hh_member_no"])

        # SUBSTEP --- Calculate HH_SIZE for MUCEP ---
        print("Calculating HH_SIZE for MUCEP...")
        hh_size_mucep = qc_form2.groupby("household_no").size().reset_index(name="hh_size")
        mucep_form1 = pd.merge(qc_form1, hh_size_mucep, on="household_no", how="left")
        print("✅ HH_SIZE for MUCEP calculated.")

        # 1. Aggregate Education Level
        print("🔍 Aggregating education level per household...")
        household_max_education = qc_form2.groupby(["HOUSING_UNIT_NO", "HH_NO"])["EDUC_LEVEL"].max().reset_index()
        household_max_education.rename(columns={"EDUC_LEVEL": "MAX_EDUC_LEVEL"}, inplace=True)
        hh = pd.merge(hh, household_max_education, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")
        household_avg_education = qc_form2.groupby(["HOUSING_UNIT_NO", "HH_NO"])["EDUC_LEVEL"].mean().reset_index()
        household_avg_education.rename(columns={"EDUC_LEVEL": "AVG_EDUC_LEVEL"}, inplace=True)
        hh = pd.merge(hh, household_avg_education, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")

        # 2. Aggregate Age
        print("🔍 Aggregating age per household...")
        household_avg_age = qc_form2.groupby(["HOUSING_UNIT_NO", "HH_NO"])["AGE"].mean().reset_index()
        household_avg_age.rename(columns={"AGE": "AVG_AGE"}, inplace=True)
        hh = pd.merge(hh, household_avg_age, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")

        household_max_age = qc_form2.groupby(["HOUSING_UNIT_NO", "HH_NO"])["AGE"].max().reset_index()
        household_max_age.rename(columns={"AGE": "MAX_AGE"}, inplace=True)
        hh = pd.merge(hh, household_max_age, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")

        household_min_age = qc_form2.groupby(["HOUSING_UNIT_NO", "HH_NO"])["AGE"].min().reset_index()
        household_min_age.rename(columns={"AGE": "MIN_AGE"}, inplace=True)
        hh = pd.merge(hh, household_min_age, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")

        # 3. Aggregate Sex
        print("🔍 Aggregating sex at birth per household...")
        household_male_proportion = qc_form2.groupby(["HOUSING_UNIT_NO", "HH_NO"])["SEX"].apply(lambda x: (x == 1).mean()).reset_index()
        household_male_proportion.rename(columns={"SEX": "MALE_PROPORTION"}, inplace=True)
        hh = pd.merge(hh, household_male_proportion, on=["HOUSING_UNIT_NO", "HH_NO"], how="left")
        print("✅ Attributes aggregated!")

        print(f"✅ Filtered MUCEP Form 1: {len(qc_form1)} QC households")
        print(f"✅ Filtered MUCEP Form 2: {len(qc_form2)} QC members")

        # ========== Form 3: Trip Information ==========
        dtype_f3 = {
            "6_1": str, "6_2": str, "6_3": str, "6_4": str,
            "6_1_others": str, "6_2_others": str, "6_3_others": str, "6_4_others": str
        }
        form3 = pd.read_csv(RAW_MUCEP_DIR / "3_Trip.csv", low_memory=False, dtype=dtype_f3)
        form3.columns = form3.columns.str.strip().str.lower()
        qc_form3 = form3[form3["household_no"].isin(qc_hh_numbers)].copy()
        qc_form3 = qc_form3.drop_duplicates(subset=["household_no", "hh_member_no", "trip_no"])
        print(f"✅ Filtered MUCEP Form 3: {len(qc_form3)} QC trips")

        # ========== Rename Important Columns for Modeling ==========
        qc_form3.rename(columns={
            "4_1": "origin_type",
            "4_2_origin": "origin_mucep_code",
            "5": "departure_time",
            "6_1": "mode_leg1", "6_1_others": "mode_leg1_other", "6_2": "stop_leg1", "6_3": "stop_1_dep_time",
            "6_4": "mode_leg2", "6_4_others": "mode_leg2_other", "6_5": "stop_leg2", "6_6": "stop_2_dep_time",
            "6_7": "mode_leg3", "6_7_others": "mode_leg3_other", "6_8": "stop_leg3", "6_9": "stop_3_dep_time",
            "6_10": "mode_leg4", "6_10_others": "mode_leg4_other", "6_11": "stop_leg4", "6_12": "stop_4_dep_time",
            "6_13": "mode_leg5", "6_13_others": "mode_leg5_other",
            "7": "arrival_time",
            "8_1": "destination_type",
            "8_2_destination": "destination_mucep_code",
            "9": "trip_purpose", "9_others": "trip_purpose_other",
            "10": "trip_cost",
            "11_1": "parking_type", "11_2": "parking_fee",
            "13": "mode_choice_reason",
            "14_1": "travel_rating_time", "14_2": "travel_rating_comfort", "14_3": "travel_rating_convenience",
            "14_4": "travel_rating_cost", "14_5": "travel_rating_safety", "14_6": "travel_rating_overall"
        }, inplace=True)

        # ========== Save Cleaned Outputs ==========
        qc_form1.to_csv(PROCESSED_DATA_DIR / "mucep_form1_qc.csv", index=False)
        qc_form2.to_csv(PROCESSED_DATA_DIR / "mucep_form2_qc.csv", index=False)
        qc_form3.to_csv(PROCESSED_DATA_DIR / "mucep_form3_qc.csv", index=False)
        print("✅ MUCEP Forms 1–3 cleaned and saved.")

    except FileNotFoundError as e:
        print(f"Error processing MUCEP data: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while processing MUCEP data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    preprocess_cph()
    preprocess_mucep()