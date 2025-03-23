def train_mode_preference_model(trip_csv_path):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    df = pd.read_csv(trip_csv_path)

    df["agent_id"] = df["Household_No"].astype(str) + "_" + df["HH_Member_No"].astype(str)
    df["primary_mode"] = df["6_1"]

    ratings_cols = ["14_1_Cost", "14_2_TravelTime", "14_3_Comfort", 
                    "14_4_Safety", "14_5_Availability", "14_6_Reliability"]
    
    df = df[["agent_id", "primary_mode"] + ratings_cols].dropna()
    df = df[df["primary_mode"].between(1, 27)]

    # Average ratings per agent
    avg_ratings = df.groupby("agent_id")[ratings_cols].mean().reset_index()

    # Dominant mode
    top_modes = (
        df.groupby(["agent_id", "primary_mode"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates("agent_id")[["agent_id", "primary_mode"]]
    )

    # Merge
    final_df = pd.merge(avg_ratings, top_modes, on="agent_id")

    X = final_df[ratings_cols]
    y = final_df["primary_mode"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X_train, y_train)

    print("Mode prediction accuracy report:\n")
    print(classification_report(y_test, clf.predict(X_test)))

    return clf, final_df

def predict_agent_mode_preferences(agent_ids, mode_model, default_rating=2):
    import pandas as pd

    X_pred = pd.DataFrame({
        "14_1_Cost": [default_rating] * len(agent_ids),
        "14_2_TravelTime": [default_rating] * len(agent_ids),
        "14_3_Comfort": [default_rating] * len(agent_ids),
        "14_4_Safety": [default_rating] * len(agent_ids),
        "14_5_Availability": [default_rating] * len(agent_ids),
        "14_6_Reliability": [default_rating] * len(agent_ids),
    }, index=agent_ids)

    y_pred = mode_model.predict(X_pred)
    return pd.DataFrame({
        "agent_id": agent_ids,
        "predicted_mode": y_pred
    })

def apply_mode_predictions_to_agents(agent_profiles_df, predicted_modes_df):
    # Merge predicted mode into agent profiles
    merged_df = agent_profiles_df.merge(
        predicted_modes_df, on="agent_id", how="left"
    )

    # If there are agents with no prediction, assign fallback (e.g., walk = 1)
    merged_df["predicted_mode"] = merged_df["predicted_mode"].fillna(1).astype(int)
    return merged_df

def assign_edge_weights_based_on_mode(G, mode_code):
    """
    Assigns weights to edges in the network graph based on agent's preferred mode.
    """
    G_mod = G.copy()
    for u, v, k, data in G_mod.edges(keys=True, data=True):
        mode = data.get("mode", "walk")
        base_time = data.get("travel_time", 1)

        # Sample penalty schema (you can fine-tune)
        if mode_code == 1:  # Walk
            weight = base_time if mode == "walk" else base_time * 3
        elif mode_code == 2:  # Jeepney
            weight = base_time if mode == "jeep" else base_time * 2
        elif mode_code == 3:  # Bus
            weight = base_time if mode == "bus" else base_time * 2.5
        elif mode_code == 4:  # Rail
            weight = base_time if mode == "rail" else base_time * 2
        else:  # Default: less strict
            weight = base_time * 1.2 if mode != "walk" else base_time

        data["weight"] = weight
    return G_mod

def route_with_mode_preference(agent, G_combined, buildings_df):
    import networkx as nx

    origin_building = buildings_df.loc[agent["origin_building_id"]]
    dest_building = buildings_df.loc[agent["dest_building_id"]]
    mode_pref = agent["predicted_mode"]

    # Adjust edge weights based on modal preference
    G_weighted = assign_edge_weights_based_on_mode(G_combined, mode_pref)

    # Get nearest nodes
    orig_node = get_nearest_node(origin_building.geometry, G_weighted)
    dest_node = get_nearest_node(dest_building.geometry, G_weighted)

    # Find shortest path
    try:
        path = nx.shortest_path(G_weighted, source=orig_node, target=dest_node, weight="weight")
        path_edges = list(zip(path[:-1], path[1:]))
    except nx.NetworkXNoPath:
        return None

    # Reconstruct the route with attributes (travel time, mode, geometry)
    route_info = []
    total_time = 0
    for u, v in path_edges:
        edge_data = G_weighted.get_edge_data(u, v, 0)
        route_info.append({
            "from": u,
            "to": v,
            "mode": edge_data["mode"],
            "travel_time": edge_data["travel_time"],
            "geometry": edge_data["geometry"]
        })
        total_time += edge_data["travel_time"]

    return {
        "agent_id": agent["agent_id"],
        "route": route_info,
        "total_travel_time": total_time
    }

def route_with_fallback(agent, G_combined, buildings_df, max_fallbacks=2):
    import networkx as nx
    
    preferred_mode = agent["predicted_mode"]
    fallback_modes = [m for m in [1, 2, 3, 4, 5] if m != preferred_mode][:max_fallbacks]

    all_modes = [preferred_mode] + fallback_modes
    for mode in all_modes:
        G_weighted = assign_edge_weights_based_on_mode(G_combined, mode)

        origin_geom = buildings_df.loc[agent["origin_building_id"]].geometry
        dest_geom = buildings_df.loc[agent["dest_building_id"]].geometry

        orig_node = get_nearest_node(origin_geom, G_weighted)
        dest_node = get_nearest_node(dest_geom, G_weighted)

        try:
            path = nx.shortest_path(G_weighted, source=orig_node, target=dest_node, weight="weight")
            path_edges = list(zip(path[:-1], path[1:]))
            route_segments = []
            total_time = 0
            total_cost = 0

            for u, v in path_edges:
                edge_data = G_weighted.get_edge_data(u, v, 0)
                seg_time = edge_data.get("travel_time", 1)
                seg_cost = edge_data.get("travel_cost", 0)
                route_segments.append({
                    "agent_id": agent["agent_id"],
                    "trip_id": agent["trip_id"],
                    "from_node": u,
                    "to_node": v,
                    "mode": edge_data["mode"],
                    "travel_time": seg_time,
                    "travel_cost": seg_cost,
                    "geometry": edge_data["geometry"]
                })
                total_time += seg_time
                total_cost += seg_cost

            return route_segments, total_time, total_cost, mode  # success

        except nx.NetworkXNoPath:
            continue

    return None, None, None, None  # fallback failed

trip_csv = "data/raw/qc-mucep/3_Trip.csv"

# Train model
mode_model, training_data = train_mode_preference_model(trip_csv)

# Predict for all agents
all_agents = list(agent_home_building_df["agent_id"].unique())
predicted_modes = predict_agent_mode_preferences(all_agents, mode_model)
agent_profiles_with_modes = apply_mode_predictions_to_agents(agent_home_building_df, predicted_modes)
