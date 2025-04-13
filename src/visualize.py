import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_total_load(load_df):
    """
    Sum all household loads for each hour.
    Returns a Series with total load per hour.
    """
    return load_df.sum(axis=0)

def plot_total_load(total_load, transformer_capacity=150):
    """
    Plot total load against transformer capacity.
    """
    plt.figure(figsize=(10, 4))
    total_load.plot(label='Total Load', marker='o')
    plt.axhline(transformer_capacity, color='r', linestyle='--', label='Capacity')
    plt.title('Total Grid Load vs Transformer Capacity')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_house_load(load_df, house_id):
    """
    Plot load profile for a specific house.
    
    Args:
        load_df: DataFrame with load profiles
        house_id: ID of house to plot
    """
    if house_id not in load_df.index:
        raise ValueError(f"House ID {house_id} not found in data")
        
    plt.figure(figsize=(10, 4))
    load_df.loc[house_id].plot(marker='o')
    plt.title(f'Load Profile for {house_id}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (kW)')
    plt.grid(True)
    plt.xticks(range(24), [f'{h}:00' for h in range(24)])
    plt.tight_layout()
    plt.show()

def plot_all_houses(load_df, sample_size=None, alpha=0.3, offset=0.7):
    """
    Plot load profiles for all houses on the same graph.
    
    Args:
        load_df: DataFrame with load profiles
        sample_size: Optional number of houses to sample (None for all)
        alpha: Transparency level for individual house lines
    """
    plt.figure(figsize=(6, 12))
    hours = range(24)
    
    # Sample houses if requested
    if sample_size and sample_size < len(load_df):
        houses = np.random.choice(load_df.index, size=sample_size, replace=False)
        plot_df = load_df.loc[houses]
    else:
        plot_df = load_df
    
    plt.subplot2grid((10,1), (1,0), rowspan=9)
    # Plot each house
    offset_cur = 0
    for house in plot_df.index:
        plt.plot(hours, plot_df.loc[house] + offset_cur, color='w')
        plt.fill_between(hours, offset_cur,plot_df.loc[house] + offset_cur, color='0.5')
        offset_cur += offset
        
    plt.xticks(hours[::2], [f'{h}:00' for h in hours[::2]])
    
    plt.subplot2grid((10,1), (0,0))
    # Plot average
    avg_load = plot_df.mean(axis=0)
    plt.plot(hours, avg_load + offset_cur, 'r-', linewidth=2, label='Average')
    
    plt.xticks(hours[::2], [f'{h}:00' for h in hours[::2]])
    plt.title(f'Load Profiles for {len(plot_df)} Houses')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (kW)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Dummy test data: 30 houses, each with constant 1.5 kW for 24 hours
    df = pd.DataFrame({f'house_{i}': [1.5]*24 for i in range(1, 31)}).T
    df.columns = [f'hour_{i}' for i in range(24)]
    total = compute_total_load(df)
    plot_total_load(total, transformer_capacity=150)
