# Simulate low-voltage demand on a synthetic grid and estimate congestion risk.

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid_model import generate_lv_grid, plot_lv_grid
from src.simulate import generate_hourly_load_profiles_realistic, save_load_profiles
from src.visualize import compute_total_load, plot_total_load, plot_all_houses

# Parameters
N_HOUSES = 50
HOURS = 24
TRANSFORMER_CAPACITY = 110  # kW
TRANSFORMER_ID = "TR_001"
DATA_DIR = "data/generated"
PLOTS_DIR = "data/plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
# Generate synthetic grid
G = generate_lv_grid(N_HOUSES)


# Plot synthetic grid
plot_lv_grid(G, plot_name=f"{PLOTS_DIR}/1_simple_simulation_grid_model.png")


# Generate synthetic hourly load profiles
house_ids = list(G.nodes())
load_profiles = generate_hourly_load_profiles_realistic(house_ids)


# Save profiles with transformer ID
saved_path = save_load_profiles(
    load_profiles, 
    output_path=DATA_DIR,
    prefix='house',
    start_id=1,
    transformer_id=TRANSFORMER_ID
)

# Load the saved profiles
loaded_profiles = pd.read_csv(saved_path, index_col='house_id')
print(f"Loaded profiles shape: {loaded_profiles.shape}")

viz_profiles = loaded_profiles.loc[~loaded_profiles.index.str.startswith('transformer'),:]
viz_profiles = viz_profiles.drop(columns=['transformer_id'])
print(f"Visualization profiles shape: {viz_profiles.shape}")

# Simple visualiation in Joy Division style:
# https://www.radiox.co.uk/artists/joy-division/cover-joy-division-unknown-pleasures-meaning/
plot_all_houses(viz_profiles, plot_name=f"{PLOTS_DIR}/2_simple_simulation_all_houses.png")

total = compute_total_load(viz_profiles)
plot_total_load(total, transformer_capacity=TRANSFORMER_CAPACITY, plot_name=f"{PLOTS_DIR}/3_simple_simulation_total_load.png")
