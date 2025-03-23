import os
from pathlib import Path

def find_project_root(marker="README.md"):
    """Finds the project root by searching upwards for a marker file."""
    current_path = Path(__file__).resolve()
    while current_path.parent != current_path:  # Avoid infinite loop at root
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Could not find project root with marker: {marker}")

PROJECT_ROOT = find_project_root()

RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "data/output"
RAW_CPH_DIR = RAW_DATA_DIR / "qc-cph"
RAW_MUCEP_DIR = RAW_DATA_DIR / "qc-mucep"
RAW_GDF_DIR = RAW_DATA_DIR / "qc-gdf"
RAW_IMG_DIR = RAW_DATA_DIR / "qc-img"
RAW_GTFS_DIR = RAW_DATA_DIR / "gtfs"

# Simulation
TIME_STEP = 300  # 5 minutes
SIM_DURATION = 86400  # 24 hours

POPULATION_FRACTION = 0.10

# Mode utility weights (modifiable later)
MODE_WEIGHTS = {
    "walk": {"time": -1.0, "cost": 0, "access": 1.0},
    "jeep": {"time": -0.8, "cost": -0.5, "access": -0.2},
    "bus": {"time": -0.9, "cost": -0.4, "access": -0.3},
    "train": {"time": -1.2, "cost": -0.6, "access": -0.1},
    "tricycle": {"time": -0.5, "cost": -0.2, "access": -0.5},
    "uv_express": {"time": -1.0, "cost": -0.6, "access": -0.3}
}
