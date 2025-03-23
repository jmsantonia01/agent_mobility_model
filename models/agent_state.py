from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentState:
    agent_id: str
    household_id: str
    current_time: str
    location_mucep: str
    current_activity: str  # e.g., "home", "travel", "work", etc.
    current_mode: Optional[str] = None
    is_traveling: bool = False
    active_trip_id: Optional[int] = None
    segment_index: Optional[int] = None

from collections import defaultdict
import pandas as pd

# Load trip logs and agent profiles
logs = pd.read_csv("outputs/simulation/agent_progression.csv")
profiles = pd.read_csv("outputs/agents/agent_profiles.csv")

# Initialize state per agent at simulation start
agent_states = {}
for _, row in profiles.iterrows():
    agent_states[row["agent_id"]] = AgentState(
        agent_id=row["agent_id"],
        household_id=row["household_id"],
        current_time="05:00",  # Simulation start
        location_mucep=row["home_mucep"],
        current_activity="home"
    )

def update_agent_states(current_time: pd.Timestamp):
    active_agents = []

    for agent_id, state in agent_states.items():
        agent_log = logs[logs["agent_id"] == agent_id]
        
        # Find activity window that matches current time
        match = agent_log[
            (agent_log["start_time"] <= current_time) &
            (agent_log["end_time"] > current_time)
        ]

        if not match.empty:
            row = match.iloc[0]
            state.current_time = current_time.strftime("%H:%M")
            state.current_mode = row["mode"] if row["mode"] != "stay" else None
            state.current_activity = row["activity"]
            state.location_mucep = row["dest_mucep"] if row["mode"] != "stay" else row["origin_mucep"]
            state.is_traveling = row["mode"] != "stay"
            state.active_trip_id = row["trip_id"]
            state.segment_index = row["segment"]
            active_agents.append(agent_id)

    return active_agents
