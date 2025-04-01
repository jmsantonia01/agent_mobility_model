import pandas as pd
from collections import defaultdict

metric_snapshots = []

def collect_metrics(agent_states, current_time):
    snapshot = {
        "time": current_time.strftime("%H:%M"),
        "total_agents": len(agent_states),
        "num_traveling": 0,
    }

    # Mode share
    mode_counts = defaultdict(int)
    activity_counts = defaultdict(int)
    zone_occupancy = defaultdict(int)
    trip_starts = 0
    trip_ends = 0

    for state in agent_states.values():
        if state.is_traveling:
            snapshot["num_traveling"] += 1
            if state.current_mode:
                mode_counts[state.current_mode] += 1
        else:
            activity_counts[state.current_activity] += 1

        zone_occupancy[state.location_mucep] += 1

    # Flatten into snapshot dict
    for mode, count in mode_counts.items():
        snapshot[f"mode_{mode}"] = count
    for act, count in activity_counts.items():
        snapshot[f"act_{act}"] = count
    for zone, count in zone_occupancy.items():
        snapshot[f"zone_{zone}"] = count

    metric_snapshots.append(snapshot)
