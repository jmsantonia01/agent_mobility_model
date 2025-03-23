# **Agent Mobility Model Script**
This is a compilation of Python scripts (minus the data) for modeling population mobility via public transport systems in Quezon City for my undergraduate research.

## Project Structure
Currently, the project is structured as such:
```
agent_mobility_model (root folder)
- data (main data folder)
-- raw (Raw, unporcessed data)
--- gtfs (Transportation GTFS data)
--- qc-cph (Initial population data)
--- qc-gdf (Geographic data files)
--- qc-img (Image files)
--- qc-mucep (Trip surey data)
-- processed (Preprocessed data)
-- output (Script final outputs)
-- clean (Intermediately processed data)

- models (Modeling scripts)
- results (Model results)
- scripts (Intermediate level script folder, for testing)
- utils (General utilities folder, for single-use scripts not part of simulation)
```

## Further developments
-[] Use ML algorithms for synthetic population generation (XGBoost, LightGBM, RF)
-[] Adjust synthetic population generator (not based on income)
-[] Adjust travel preferences modeling scripts (travel_mode_model, mode_choice, travel_prefs)
-[] Ensure consistency across scheduler, agent_state, building_assignment, net_anim, timeline_visuals
-[] Integrate models, task hooks, routing_engine into simulation_engine
-[] Adjust main model script for actual mobility simulation
-[] Benchmark performance at 10 households OR 100 agents
-[] Adjust main script for CUDA, parallel processing capabilities
-[] Benchmark new performance with extended processing power
-[] Implement across 10% sample
