'@author: John Carlo Santonia'
'@date: 2024-01-15'

# Include all imports here
from models.population_generator import generate_synthetic_population
from models.routing_engine import compute_paths
from models.simulation_engine import run_simulation
from utils.config import *

# Main function
def main():
    print("[STEP 1] Generating population...")
    agents = generate_synthetic_population(POPULATION_FRACTION)

    print("[STEP 2] Computing routes...")
    compute_paths(agents, network_data=None)  # placeholder

    print("[STEP 3] Running simulation...")
    run_simulation(agents, SIM_DURATION, TIME_STEP)

if __name__ == "__main__":
    main()
