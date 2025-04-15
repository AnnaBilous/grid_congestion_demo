import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_total_load(load_df):
    """
    Calculate the total load across all households for each hour.
    
    Sums the electricity consumption of all houses at each time point
    to determine the aggregate load on the transformer.
    
    Parameters
    ----------
    load_df : pandas.DataFrame
        DataFrame with houses as rows and hourly loads as columns
        
    Returns
    -------
    pandas.Series
        Series containing the total load for each hour
    """
    return load_df.sum(axis=0)

def plot_total_load(total_load, transformer_capacity=150, plot_name=None):
    """
    Plot total grid load against transformer capacity.
    
    Creates a visualization showing the aggregate load profile and
    the transformer capacity threshold to identify potential congestion.
    
    Parameters
    ----------
    total_load : pandas.Series
        Series containing total load for each hour
    transformer_capacity : float, default=150
        Transformer capacity in kW
    plot_name : str, optional
        If provided, save the plot to this path instead of displaying
        
    Returns
    -------
    None
        Either displays the plot or saves it to file
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
    if plot_name:
        plt.savefig(plot_name)
    else:
        plt.show()



def plot_all_houses(load_df, sample_size=None, offset=0.7, plot_name=None):
    """
    Plot load profiles for all houses in a stacked visualization.
    
    Creates a stacked area chart showing individual house load profiles
    and the average profile to visualize consumption patterns. The visualization
    style is inspired by Joy Division's "Unknown Pleasures" album cover.
    
    Parameters
    ----------
    load_df : pandas.DataFrame
        DataFrame with houses as rows and hourly loads as columns
    sample_size : int, optional
        Number of houses to randomly sample (None for all houses)
    offset : float, default=0.7
        Vertical spacing between house profiles
    plot_name : str, optional
        If provided, save the plot to this path instead of displaying
        
    Returns
    -------
    None
        Either displays the plot or saves it to file
    """
    plt.figure(figsize=(6, 12))
    hours = range(24)
    
    # Sample houses if requested
    if sample_size and sample_size < len(load_df):
        houses = np.random.choice(load_df.index, size=sample_size, replace=False)
        plot_df = load_df.loc[houses]
    else:
        plot_df = load_df
    
    # Add hour 24 data (same as hour 0)
    plot_df_extended = plot_df.copy()
    plot_df_extended.insert(24, "hour_24", plot_df_extended.iloc[:,0])
    hours_extended = list(range(25))
    
    plt.subplot2grid((10,1), (1,0), rowspan=9)
    # Plot each house
    offset_cur = 0
    zorder = 100
    for house in plot_df_extended.index:
        plt.plot(hours_extended, plot_df_extended.loc[house] + offset_cur, color='w', lw=5, zorder=zorder)
        plt.fill_between(hours_extended, offset_cur, plot_df_extended.loc[house] + offset_cur, color='0.2', zorder=zorder)
        offset_cur += offset
        zorder -= 1
        
    plt.xticks(hours_extended[::2], [f'{h}:00' for h in hours_extended[::2]])
    plt.xlim(0, 24)
    plt.yticks([])
    plt.ylabel("Individual houses")
    plt.subplot2grid((10,1), (0,0))
    # Plot average
    avg_load = plot_df_extended.mean(axis=0)
    plt.plot(hours_extended, avg_load, 'r-', linewidth=2, label='Average')
    
    plt.xticks(hours_extended[::2], [f'{h}:00' for h in hours_extended[::2]])
    plt.xlim(0, 24)
    plt.title(f'Load Profiles for {len(plot_df)} Houses')
    plt.ylabel('Load (kW)')
    plt.legend()
    
    plt.tight_layout()
    if plot_name:
        plt.savefig(plot_name)
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Dummy test data: 30 houses, each with constant 1.5 kW for 24 hours
    df = pd.DataFrame({f'house_{i}': [1.5]*24 for i in range(1, 31)}).T
    df.columns = [f'hour_{i}' for i in range(24)]
    total = compute_total_load(df)
    plot_total_load(total, transformer_capacity=150)
