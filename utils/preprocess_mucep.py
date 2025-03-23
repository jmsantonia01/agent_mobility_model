import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/qc-mucep")
PROC_DIR = Path("data/processed")
REF_FILE = RAW_DIR / "5_BrgyZones_QC.csv"

# Load QC MUCEP codes for filtering
ref = pd.read_csv(REF_FILE)
qc_codes = ref["MUCEPCode"].dropna().unique().tolist()

# ========== Form 1: Household Information ==========
form1 = pd.read_csv(RAW_DIR / "1_HH.csv")
form1.columns = form1.columns.str.strip().str.lower()
qc_form1 = form1[form1["2_address_mucep_code"].isin(qc_codes)].copy()
qc_form1 = qc_form1.drop_duplicates(subset=["household_no"])
print(f"✅ Filtered Form 1: {len(qc_form1)} QC households")

# ========== Form 2: Household Members ==========
form2 = pd.read_csv(RAW_DIR / "2_HHM.csv")
form2.columns = form2.columns.str.strip().str.lower()
qc_form2 = form2[form2["household_no"].isin(qc_codes)].copy()
qc_form2 = qc_form2.drop_duplicates(subset=["household_no", "hh_member_no"])
print(f"✅ Filtered Form 2: {len(qc_form2)} QC members")

# ========== Form 3: Trip Information ==========
dtype_f3 = {
    "6_1": str, "6_2": str, "6_3": str, "6_4": str,
    "6_1_others": str, "6_2_others": str, "6_3_others": str, "6_4_others": str
}
form3 = pd.read_csv(RAW_DIR / "3_Trip.csv", low_memory=False, dtype=dtype_f3)
form3.columns = form3.columns.str.strip().str.lower()
qc_form3 = form3[form3["household_no"].isin(qc_codes)].copy()
qc_form3 = qc_form3.drop_duplicates(subset=["household_no", "hh_member_no", "trip_no"])
print(f"✅ Filtered Form 3: {len(qc_form3)} QC trips")

# ========== Rename Important Columns for Modeling ==========
qc_form3.rename(columns={
    "5": "departure_time",
    "6_1": "mode_leg1", "6_1_others": "mode_leg1_other",
    "6_2": "mode_leg2", "6_2_others": "mode_leg2_other",
    "6_3": "mode_leg3", "6_3_others": "mode_leg3_other",
    "6_4": "mode_leg4", "6_4_others": "mode_leg4_other",
    "7": "arrival_time",
    "8_1": "destination_type",
    "8_2_destination": "destination_mucep_code",
    "9": "trip_purpose", "9_others": "trip_purpose_other",
    "10": "trip_cost",
    "11_1": "parking_type", "11_2": "parking_fee",
    "13": "mode_choice_reason"
}, inplace=True)

# ========== Save Cleaned Outputs ==========
qc_form1.to_csv(PROC_DIR / "mucep_form1_qc.csv", index=False)
qc_form2.to_csv(PROC_DIR / "mucep_form2_qc.csv", index=False)
qc_form3.to_csv(PROC_DIR / "mucep_form3_qc.csv", index=False)
print("✅ MUCEP Forms 1–3 cleaned and saved.")

