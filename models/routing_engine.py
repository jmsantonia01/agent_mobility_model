import geopandas as gpd

all_routes = []
updated_agents = []

for agent in agents_with_trips:
    for trip in agent["trips"]:
        agent_trip = {
            "agent_id": agent["agent_id"],
            "trip_id": trip["trip_id"],
            "origin_building_id": trip["origin_building_id"],
            "dest_building_id": trip["dest_building_id"],
            "predicted_mode": trip["predicted_mode"]
        }

        route, total_time, total_cost, used_mode = route_with_fallback(agent_trip, G_combined, buildings_df)

        if route:
            all_routes.extend(route)
            trip.update({
                "used_mode": used_mode,
                "total_travel_time": total_time,
                "total_travel_cost": total_cost
            })
        else:
            trip.update({
                "used_mode": None,
                "total_travel_time": None,
                "total_travel_cost": None
            })

    updated_agents.append(agent)

route_gdf = gpd.GeoDataFrame(all_routes, geometry="geometry", crs="EPSG:32651")  # or your preferred CRS
route_gdf.to_file("data/processed/agent_routes.geojson", driver="GeoJSON")

import json

with open("data/processed/agents_with_routes.json", "w") as f:
    json.dump(updated_agents, f, indent=2)
