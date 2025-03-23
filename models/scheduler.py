import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Time interval in minutes (5 minutes = 288 steps per day)
TIME_INTERVAL = 5
TOTAL_STEPS = int(24 * 60 / TIME_INTERVAL)

# Build the time index for the day
time_steps = [timedelta(minutes=i * TIME_INTERVAL) for i in range(TOTAL_STEPS)]

agent_states = {
    agent_id: {
        "status": "idle",
        "location": home_building,
        "current_trip": None,
        "remaining_time": 0
    }
    for agent_id, home_building in agent_home_buildings.items()  # assume you have this
}

# Scheduler dict: timestep -> list of (agent_id, trip_id, trip_info)
scheduler = {i: [] for i in range(TOTAL_STEPS)}

def get_timestep_index(time_str):
    """Convert 'hhmm' string to 5-min interval index."""
    if pd.isna(time_str):
        return None
    try:
        time = datetime.strptime(str(int(time_str)).zfill(4), "%H%M")
        minutes_since_midnight = time.hour * 60 + time.minute
        return minutes_since_midnight // TIME_INTERVAL
    except:
        return None

def populate_scheduler(agent_profiles):
    """
    Assumes agent_profiles is a list/dict where each agent has:
    - agent_id
    - trips: list of dicts with keys: departure_time, route_geom, travel_time, etc.
    """
    for agent in agent_profiles:
        agent_id = agent["agent_id"]
        for i, trip in enumerate(agent["trips"]):
            t_idx = get_timestep_index(trip["departure_time"])
            if t_idx is not None and t_idx < TOTAL_STEPS:
                scheduler[t_idx].append({
                    "agent_id": agent_id,
                    "trip_id": i,
                    "departure_time": trip["departure_time"],
                    "origin": trip["origin_building"],
                    "destination": trip["destination_building"],
                    "mode": trip["mode"],
                    "geometry": trip["route_geom"],
                    "travel_time": trip["estimated_duration"],
                })

    return scheduler

import json
from collections import defaultdict

def simulate_day(scheduler, agent_profiles, agent_states, interval_minutes=5, output_path="outputs/agent_logs.json"):
    TOTAL_STEPS = 288  # 24h * (60 / 5min)
    time_log = []
    
    # Cumulative modal times
    modal_time_tracker = defaultdict(lambda: defaultdict(int))  # agent_id → mode → total_minutes

    for t in range(TOTAL_STEPS):
        active_trips = scheduler.get(t, [])

        # Start new trips
        for trip in active_trips:
            agent_id = trip["agent_id"]
            agent_states[agent_id]["status"] = "in_transit"
            agent_states[agent_id]["current_trip"] = trip
            agent_states[agent_id]["remaining_time"] = int(np.ceil(trip["travel_time"] / interval_minutes))

        # Update ongoing trips
        for agent_id, state in agent_states.items():
            if state["status"] == "in_transit":
                state["remaining_time"] -= 1

                # Add to modal time
                current_mode = state["current_trip"]["mode"]  # e.g., "walk", "bus", "rail"
                modal_time_tracker[agent_id][current_mode] += interval_minutes

                if state["remaining_time"] <= 0:
                    state["status"] = "idle"
                    state["location"] = state["current_trip"]["destination"]
                    state["current_trip"] = None
                    state["remaining_time"] = 0

        # Log snapshot
        snapshot = {
            "timestep": t,
            "agent_states": {
                aid: {
                    "status": s["status"],
                    "location": s["location"],
                    "remaining_time": s["remaining_time"]
                }
                for aid, s in agent_states.items()
            }
        }
        time_log.append(snapshot)

    # Save logs
    with open(output_path, "w") as f:
        json.dump({
            "agent_progression": time_log,
            "modal_time_tracker": modal_time_tracker
        }, f, indent=2)

    print(f"Saved simulation log to {output_path}")
    return time_log

import csv

def save_agent_progression_csv(agent_progression, output_path="outputs/agent_progression.csv"):
    rows = []
    for snapshot in agent_progression:
        timestep = snapshot["timestep"]
        for agent_id, state in snapshot["agent_states"].items():
            rows.append({
                "timestep": timestep,
                "agent_id": agent_id,
                "status": state["status"],
                "location": state["location"],
                "remaining_time": state["remaining_time"]
            })
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved agent progression CSV to {output_path}")

def save_modal_time_csv(modal_time_tracker, output_path="outputs/agent_modal_time.csv"):
    rows = []
    for agent_id, modes in modal_time_tracker.items():
        row = {"agent_id": agent_id}
        row.update(modes)
        rows.append(row)
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved modal time CSV to {output_path}")

