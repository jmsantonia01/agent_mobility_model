# simulation_engine.py

import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString
from shapely.ops import substring
import os
from utils.config import TIME_STEP,OUTPUT_DIR
from datetime import datetime

# Simulation tick size in seconds (5 minutes)
TICK_SIZE = TIME_STEP
MAX_TIME = 86100  # 11:55 PM

class SimulationEngine:
    def __init__(self, agents_df, output_dir=OUTPUT_DIR):
        self.agents = agents_df.copy()
        self.time = 0
        self.tick = 0
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logs = []
        self.snapshots = []

    def run(self):
        while self.time <= MAX_TIME:
            print(f"[Tick {self.tick}] Time: {self.time // 3600:02}:{(self.time % 3600) // 60:02}")
            self.tick_agents()
            self.save_snapshot()
            self.time += TICK_SIZE
            self.tick += 1
        self.save_logs()

    def tick_agents(self):
        for idx, agent in self.agents.iterrows():
            agent_id = agent['agent_id']
            state = agent['state']
            schedule = agent['schedule']
            current_trip = agent.get('current_trip', 0)

            # Check if new trip starts
            if current_trip < len(schedule):
                trip_info = schedule[current_trip]
                if self.time >= trip_info['start_time'] and state != 'traveling':
                    # Start new trip
                    self.agents.at[idx, 'state'] = 'traveling'
                    self.agents.at[idx, 'trip_start_time'] = self.time
                    self.agents.at[idx, 'route_pos'] = 0.0
                    self.log_event(agent_id, 'DEPART', trip_info)

            # Process traveling agents
            if state == 'traveling':
                trip_info = schedule[current_trip]
                route = trip_info['route']
                travel_time = trip_info['travel_time']
                segment_speed = route.length / travel_time  # units per sec
                time_elapsed = self.time - agent.get('trip_start_time', self.time)
                distance_traveled = min(segment_speed * time_elapsed, route.length)

                # Update position
                new_pos = substring(route, 0, distance_traveled)
                self.agents.at[idx, 'geometry'] = new_pos.interpolate(1.0, normalized=True)

                # Check for arrival
                if distance_traveled >= route.length:
                    self.agents.at[idx, 'state'] = 'at_activity'
                    self.agents.at[idx, 'geometry'] = trip_info['destination_geom']
                    self.agents.at[idx, 'current_trip'] = current_trip + 1
                    self.log_event(agent_id, 'ARRIVE', trip_info)

    def log_event(self, agent_id, event_type, trip_info):
        self.logs.append({
            'agent_id': agent_id,
            'event': event_type,
            'trip_purpose': trip_info['purpose'],
            'mode': trip_info['mode'],
            'start_time': trip_info['start_time'],
            'duration': trip_info['travel_time'],
            'timestamp': self.time
        })

    def save_snapshot(self):
        gdf = self.agents[['agent_id', 'state', 'geometry']].copy()
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:32651')
        out_path = os.path.join(self.output_dir, f"snapshot_{self.tick:04}.geojson")
        gdf.to_file(out_path, driver='GeoJSON')

    def save_logs(self):
        log_df = pd.DataFrame(self.logs)
        log_path = os.path.join(self.output_dir, "agent_travel_logs.csv")
        log_df.to_csv(log_path, index=False)


# Example usage (if run as script)
if __name__ == "__main__":
    agents = gpd.read_file("data/processed/agent_profiles.geojson")  # Preloaded with schedule, routes, etc.
    sim = SimulationEngine(agents)
    sim.run()
