import pandas as pd

# Load the CSV progression log
df = pd.read_csv("outputs/simulation/agent_progression.csv")

# Ensure datetime columns are parsed properly
df["start_time"] = pd.to_datetime(df["start_time"])
df["end_time"] = pd.to_datetime(df["end_time"])

# Optional: Convert categorical data
df["mode"] = df["mode"].astype("category")

# Load agent profiles to get household ID
profiles = pd.read_csv("outputs/agents/agent_profiles.csv")

# Merge household ID into progression data
df = df.merge(profiles[["agent_id", "household_id"]], on="agent_id", how="left")

# Optional: create a unique label per agent
df["agent_label"] = "Agent " + df["agent_id"].astype(str)

import plotly.express as px

# Create a Plotly timeline figure
fig = px.timeline(
    df,
    x_start="start_time",
    x_end="end_time",
    y="agent_label",
    color="mode",
    hover_data=["trip_id", "activity", "mode", "start_time", "end_time", "duration_min"],
    facet_row="household_id",  # One timeline per household
    title="Daily Agent Travel Timeline Grouped by Household"
)

# Reverse the Y-axis so time flows top to bottom
fig.update_yaxes(autorange="reversed")

# Tidy layout
fig.update_layout(
    height=300 + len(df["household_id"].unique()) * 100,
    legend_title_text="Mode of Travel",
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.show()
fig.write_html("data/outputs/agent_timeline.html")