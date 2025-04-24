import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from src
from src.grid_model import generate_lv_grid
from src.simulate import generate_hourly_load_profiles_realistic
from src.visualize import compute_total_load, plot_all_houses

# Constants
HOURS = 24
TRANSFORMER_CAPACITY = 125  # kW
DATA_DIR = "data/generated"
PLOTS_DIR = "data/plots"
RESULTS_FILE = f"{DATA_DIR}/monte_carlo_results.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# EV and Solar PV parameters
EV_LOAD = 2.5  # kW
EV_HOURS = range(18, 23)  # 18:00-22:00
PV_MAX_OUTPUT = 2.0  # kW
PV_HOURS = range(10, 17)  # 10:00-16:00

colors = {'EVPV': '#C8CFA0', 'EV': '#78ABA8', 'PV': '#FCDC94', 'Standard': '0.7'}

def add_technologies(base_profiles, n_ev, n_pv, ev_load, pv_output):
    """Add EVs and solar panels to random houses"""
    profiles = base_profiles.copy()
    profiles = profiles.loc[~profiles.index.str.startswith("transformer"), :]

    # Randomly select houses for each technology
    all_houses = profiles.index.tolist()
    ev_houses = np.random.choice(all_houses, size=min(n_ev, len(all_houses)), replace=False)
    pv_houses = np.random.choice(all_houses, size=min(n_pv, len(all_houses)), replace=False)

    # Add EV load
    for house in ev_houses:
        for hour in EV_HOURS:
            profiles.loc[house, f"hour_{hour}"] += ev_load

    # Subtract PV generation
    for house in pv_houses:
        for hour in PV_HOURS:
            hour_factor = 1.0 - 0.3 * abs(hour - 12) / 3
            pv_output_hour = pv_output * hour_factor
            current_load = profiles.loc[house, f"hour_{hour}"]
            profiles.loc[house, f"hour_{hour}"] = max(0.1, current_load - pv_output_hour)

    return profiles, ev_houses, pv_houses

def run_simulation(house_ids, n_ev, n_pv, transformer_capacity, ev_load, pv_output, n_simulations=100):
    """Run simulations with specified EV and PV counts"""
    congestion_count = 0
    peak_loads = []
    load_profiles = []
    house_profiles = None
    ev_houses = None
    pv_houses = None

    for sim in range(n_simulations):
        base_load_profiles = generate_hourly_load_profiles_realistic(house_ids)
        modified_profiles, sim_ev_houses, sim_pv_houses = add_technologies(base_load_profiles, n_ev, n_pv, ev_load, pv_output)
        total_load = compute_total_load(modified_profiles)
        
        # Store for visualization
        load_profiles.append(total_load)
            
        # Store one set of house profiles for visualization
        if sim == 0:
            house_profiles = modified_profiles
            ev_houses = sim_ev_houses
            pv_houses = sim_pv_houses
            
        peak_load = total_load.max()
        peak_loads.append(peak_load)
        
        if peak_load > transformer_capacity:
            congestion_count += 1

    congestion_probability = congestion_count / n_simulations
    return {
        "congestion_probability": congestion_probability,
        "avg_peak_load": np.mean(peak_loads),
        "max_peak_load": np.max(peak_loads),
        "min_peak_load": np.min(peak_loads),
        "load_profiles": load_profiles,
        "house_profiles": house_profiles,
        "ev_houses": ev_houses,
        "pv_houses": pv_houses
    }

# Streamlit app
st.title("Grid Congestion Analysis")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
n_houses = st.sidebar.slider("Number of Houses", 5, 50, 25)
n_ev = st.sidebar.slider("Number of EVs", 0, 10, 5)
n_pv = st.sidebar.slider("Number of Solar PV", 0, 10, 5)
transformer_capacity = st.sidebar.slider("Transformer Capacity (kW)", 50, 200, 70)
n_simulations = st.sidebar.slider("Number of Simulations", 10, 100, 50)

st.sidebar.header("Technology Parameters")
ev_load = st.sidebar.slider("EV Charging Power (kW)", 1.0, 10.0, 2.5, 0.5)
pv_output = st.sidebar.slider("PV Maximum Output (kW)", 0.5, 5.0, 2.0, 0.5)

# Update the global constants
TRANSFORMER_CAPACITY = transformer_capacity
EV_LOAD = ev_load
PV_MAX_OUTPUT = pv_output

# Generate grid when parameters change
if 'G' not in st.session_state or st.session_state.n_houses != n_houses:
    st.session_state.G = generate_lv_grid(n_houses)
    st.session_state.n_houses = n_houses
    st.session_state.house_ids = list(st.session_state.G.nodes())

# Run simulation button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulations..."):
        results = run_simulation(
            st.session_state.house_ids, 
            n_ev, 
            n_pv,
            transformer_capacity,
            ev_load,
            pv_output,
            n_simulations
        )
        st.session_state.results = results

# Display results
if 'results' in st.session_state:
    results = st.session_state.results
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Congestion Probability", f"{results['congestion_probability']:.2f}")
    col2.metric("Average Peak Load", f"{results['avg_peak_load']:.2f} kW")
    col3.metric("Maximum Peak Load", f"{results['max_peak_load']:.2f} kW")
    
    # Load profile visualization
    st.subheader("Load Profiles")
    
    # Create two columns for side-by-side plots
    left_col, right_col = st.columns(2)
    
    # Aggregate load plot in left column
    with left_col:
        st.write("### Aggregate Load")
        fig1, ax1 = plt.subplots(figsize=(6, 8))
        
        # Create extended hours array to match second plot
        hours = range(HOURS)
        hours_extended = list(range(HOURS + 1))
        
        # Plot all simulations with the same styling
        for i, profile in enumerate(results['load_profiles']):
            # Add hour 24 data (same as hour 0) to match second plot
            profile_values = profile.values  # Convert to numpy array
            profile_extended = np.append(profile_values, profile_values[0])
            
            # All simulations in same color and style
            ax1.plot(hours_extended, profile_extended, color='0.2', alpha=0.9, 
                    linewidth=0.5)
        
        # Add a single label for all simulations
        # This is a dummy line just for the legend
        ax1.plot([], [], color='0.2', alpha=0.9, linewidth=2, label=f"Simulations ({len(results['load_profiles'])})")
        
        ax1.axhline(transformer_capacity, color='r', linestyle='--', label="Transformer Capacity")
        
        # Highlight EV and PV hours
        if n_ev > 0:
            ax1.axvspan(min(EV_HOURS), max(EV_HOURS) + 1, alpha=0.7, color=colors['EV'])
        if n_pv > 0:
            ax1.axvspan(min(PV_HOURS), max(PV_HOURS) + 1, alpha=0.7, color=colors['PV'])
        
        ax1.set_xlabel("Hour of Day", fontsize=11)
        ax1.set_ylabel("Load (kW)", fontsize=11)
        ax1.set_xticks(hours_extended[::4])
        ax1.set_xticklabels([f"{h}:00" for h in hours_extended[::4]], fontsize=10)
        ax1.set_xlim(0, 24)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.grid(True)
        
        # Create custom legend with patches for EV/PV hours
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], color='0.2', lw=2, label=f'Simulations ({len(results["load_profiles"])})'),
            Line2D([0], [0], color='r', linestyle='--', label='Transformer Capacity')
        ]
        
        if n_ev > 0:
            legend_elements.append(Patch(facecolor=colors['EV'], alpha=0.7, label='EV Hours'))
        if n_pv > 0:
            legend_elements.append(Patch(facecolor=colors['PV'], alpha=0.7, label='PV Hours'))
        
        ax1.legend(handles=legend_elements, fontsize=9)
        ax1.set_title(f"Load with {n_ev} EVs, {n_pv} PVs", fontsize=12)
        
        st.pyplot(fig1)
    
    # Individual houses plot in right column
    with right_col:
        st.write("### Individual House Profiles")
        if results['house_profiles'] is not None:
            # Close any existing figures
            plt.close()
            
            # Create a single plot with more bottom margin for legend
            fig2 = plt.figure(figsize=(6, 7.5))
            # Add subplot with adjusted bottom margin for legend
            ax2 = fig2.add_subplot(111)
            fig2.subplots_adjust(bottom=0.15)  # Make room for legend
            
            # Sample houses if there are too many
            sample_size = min(30, len(results['house_profiles']))
            
            # Create a custom version of plot_all_houses for Streamlit
            hours = range(24)
            plot_df = results['house_profiles']
            
            # Add hour 24 data (same as hour 0)
            plot_df_extended = plot_df.copy()
            plot_df_extended.insert(24, "hour_24", plot_df_extended.iloc[:,0])
            hours_extended = list(range(25))
            
            # Plot each house
            offset_cur = 0
            offset = 1
            zorder = 100
            
            # Color houses based on technology
            ev_houses = results['ev_houses']
            pv_houses = results['pv_houses']
            
            
            for house in plot_df_extended.index:
                if house in ev_houses and house in pv_houses:
                    color = colors['EVPV']  # Both EV and PV
                    label = "EV+PV"
                elif house in ev_houses:
                    color = colors['EV']  # EV only
                    label = "EV"
                elif house in pv_houses:
                    color = colors['PV']  # PV only
                    label = "PV"
                else:
                    color = colors['Standard']  # Neither
                    label = "Standard"
                
                ax2.plot(hours_extended, plot_df_extended.loc[house] + offset_cur, color=color, lw=2, zorder=zorder)
                ax2.fill_between(hours_extended, offset_cur, plot_df_extended.loc[house] + offset_cur, color=color, zorder=zorder, alpha=0.8)
                offset_cur += offset
                zorder -= 1
                
            ax2.set_xticks(hours_extended[::4], [f'{h}:00' for h in hours_extended[::4]], fontsize=10)
            ax2.set_xlim(0, 24)
            ax2.set_yticks([])
            ax2.set_ylabel("Individual houses", fontsize=11)
            ax2.set_title(f'Profiles for {len(plot_df)} Houses in one simulation', fontsize=12)
            
            # Add a legend for house types
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors['EV'], alpha=0.7, label='EV House'),
                Patch(facecolor=colors['PV'], alpha=0.7, label='PV House'),
                Patch(facecolor=colors['EVPV'], alpha=0.7, label='EV+PV House'),
                Patch(facecolor=colors['Standard'], alpha=0.7, label='Standard House')
            ]
            
            # Place legend outside the plot at the bottom center
            ax2.legend(handles=legend_elements, 
                          loc='lower center', 
                          ncol=4, 
                          bbox_to_anchor=(0.5, -0.05),
                          fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig2)
    
    # Explanation
    st.subheader("Analysis")
    st.write(f"""
    - **Grid Configuration**: {n_houses} houses connected to a {transformer_capacity} kW transformer
    - **Technology Adoption**: {n_ev} electric vehicles ({ev_load} kW) and {n_pv} solar panels ({pv_output} kW)
    - **Congestion Analysis**: The grid experiences congestion in {results['congestion_probability']*100:.1f}% of simulations
    - **Load Statistics**:
      - Average peak load: {results['avg_peak_load']:.1f} kW ({(results['avg_peak_load']/transformer_capacity*100):.1f}% of capacity)
      - Maximum peak load: {results['max_peak_load']:.1f} kW ({(results['max_peak_load']/transformer_capacity*100):.1f}% of capacity)
      - Minimum peak load: {results['min_peak_load']:.1f} kW ({(results['min_peak_load']/transformer_capacity*100):.1f}% of capacity)
    - **Time Patterns**:
      - EV charging occurs between {min(EV_HOURS)}:00-{max(EV_HOURS)}:00
      - Solar generation peaks between {min(PV_HOURS)}:00-{max(PV_HOURS)}:00
    """)

    # Add recommendations based on results
    if results['congestion_probability'] > 0.5:
        st.warning(f"""
        **High Congestion Risk**: This configuration has a high probability of grid congestion.
        
        Possible solutions:
        - Increase transformer capacity
        - Implement smart charging for EVs
        - Encourage load shifting to off-peak hours
        """)
    elif results['congestion_probability'] > 0.1:
        st.info(f"""
        **Moderate Congestion Risk**: This configuration has some risk of grid congestion.
        
        Considerations:
        - Monitor peak loads during EV charging hours
        - Consider local energy storage to buffer peak demands
        """)
    else:
        st.success(f"""
        **Low Congestion Risk**: This configuration has minimal risk of grid congestion.
        
        The grid can likely support additional EV adoption.
        """)

else:
    st.info("Click 'Run Simulation' to see results")

# Instructions
with st.expander("How to use this app"):
    st.write("""
    1. Adjust parameters using the sliders
    2. Click 'Run Simulation' to analyze grid congestion
    3. View the results including congestion probability and load profiles
    4. Try different combinations to see how technology adoption affects grid congestion
    """) 