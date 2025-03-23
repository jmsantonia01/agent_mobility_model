import pandas as pd

def prepare_mode_preference_data(trip_csv_path):
    df = pd.read_csv(trip_csv_path)

    df["agent_id"] = df["Household_No"].astype(str) + "_" + df["HH_Member_No"].astype(str)

    # Get primary mode (first leg) only for now
    df["primary_mode"] = df["6_1"]
    
    # Keep only relevant columns
    ratings_cols = ["14_1_Cost", "14_2_TravelTime", "14_3_Comfort", 
                    "14_4_Safety", "14_5_Availability", "14_6_Reliability"]
    used_cols = ["agent_id", "primary_mode", "13"] + ratings_cols

    df = df[used_cols].dropna()

    # Drop entries with missing or invalid mode
    df = df[df["primary_mode"].between(1, 27)]  # valid MUCEP codes

    # Average ratings per agent (in case of multiple trips)
    ratings_df = df.groupby("agent_id")[ratings_cols].mean().reset_index()

    # Get dominant mode per agent
    mode_df = df.groupby(["agent_id", "primary_mode"]).size().reset_index(name="count")
    mode_df = mode_df.sort_values("count", ascending=False).drop_duplicates("agent_id")
    mode_df = mode_df[["agent_id", "primary_mode"]]

    final_df = pd.merge(ratings_df, mode_df, on="agent_id")
    return final_df
