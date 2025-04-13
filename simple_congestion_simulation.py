# %% [markdown]
# # Grid Congestion Risk Simulation
# 
# Simulate low-voltage demand on a synthetic grid and estimate congestion risk.

# %%
# Imports
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')  # Add parent directory to path

# Import from src
from src.grid_model import generate_lv_grid, plot_lv_grid
from src.simulate import generate_hourly_load_profiles_realistic, save_load_profiles
from src.visualize import compute_total_load, plot_total_load, plot_all_houses

# Parameters
N_HOUSES = 50
HOURS = 24
TRANSFORMER_CAPACITY = 110  # kW
TRANSFORMER_ID = "TR_001"
DATA_DIR = "data/generated"

# %%
# Generate synthetic grid
G = generate_lv_grid(N_HOUSES)

# %%
# Plot synthetic grid
#plot_lv_grid(G)

# %%
# Generate synthetic hourly load profiles
house_ids = list(G.nodes())
load_profiles = generate_hourly_load_profiles_realistic(house_ids)

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

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

# Remove transformer column for visualization
if 'transformer_id' in loaded_profiles.columns:
    viz_profiles = loaded_profiles.drop(columns=['transformer_id'])
else:
    viz_profiles = loaded_profiles

plot_all_houses(viz_profiles)

# %%
total = compute_total_load(viz_profiles)
plot_total_load(total, transformer_capacity=TRANSFORMER_CAPACITY)
