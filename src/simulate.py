import numpy as np
import pandas as pd


def model_load_profiles_realistic(hours=24, idle_consumption=0.3, 
                                 morning_peak_height=1.0, morning_peak_width=1.5, 
                                 midday_peak_height=0.7, midday_peak_width=2.0,
                                 evening_peak_height=1.5, evening_peak_width=1.8, 
                                 evening_decay_time=3.0):
    """
    Generate realistic load profile with four components:
    A) Constant idle consumption
    B) Morning peak (Gaussian around 7am)
    C) Midday peak (Gaussian around 12pm)
    D) Evening peak (Gaussian multiplied by exponential decay, wrapping around midnight)
    
    Returns hourly load profile for 24 hours
    """
    # Time points (hours)
    t = np.arange(hours)
    
    # A) Constant idle consumption
    idle = np.ones(hours) * idle_consumption
    
    # B) Morning peak (Gaussian around 7am)
    morning_center = 7
    morning = morning_peak_height * np.exp(-((t - morning_center) ** 2) / (2 * morning_peak_width ** 2))
    
    # C) Midday peak (Gaussian around 12pm)
    midday_center = 12
    midday = midday_peak_height * np.exp(-((t - midday_center) ** 2) / (2 * midday_peak_width ** 2))
    
    # D) Evening peak with fast rise and exponential decay that wraps around midnight
    evening_center = 18
    # Extended time array to handle wrap-around (18 hours before to 18 hours after evening_center)
    t_extended = np.arange(evening_center - 18, evening_center + 18)
    # Gaussian part on extended array
    evening_gaussian_ext = np.exp(-((t_extended - evening_center) ** 2) / (2 * evening_peak_width ** 2))
    # Apply asymmetric decay
    evening_asymmetric_ext = np.zeros_like(t_extended, dtype=float)
    for i, hour in enumerate(t_extended):
        if hour <= evening_center:
            evening_asymmetric_ext[i] = evening_gaussian_ext[i]
        else:
            # Exponential decay after peak
            evening_asymmetric_ext[i] = evening_gaussian_ext[i] * np.exp(-(hour - evening_center) / evening_decay_time)
    
    # Map extended hours back to 0-23 range
    evening_asymmetric = np.zeros(hours)
    for i in range(len(t_extended)):
        hour_mod = t_extended[i] % hours
        evening_asymmetric[hour_mod] += evening_asymmetric_ext[i]
    
    evening = evening_peak_height * evening_asymmetric
    
    # Combine all components
    load_profile = idle + morning + midday + evening
    
    return load_profile

def generate_hourly_load_profiles_realistic(house_ids, hours=24, seed=None):
    """
    Generate synthetic hourly load profiles for each house.
    Returns a pandas DataFrame indexed by house_id.
    """
    if seed is not None:
        np.random.seed(seed)
    
    load_profiles = {}
    for house in house_ids:
        # Get base profile with random variations in parameters
        idle = np.random.uniform(0.2, 0.4)
        morning_h = np.random.uniform(0.2, 2.0)
        morning_w = np.random.uniform(1.0, 3.0)
        midday_h = np.random.uniform(0.0, 1.5)
        midday_w = np.random.uniform(1.8, 2.2)
        evening_h = np.random.uniform(1.0, 2.8)
        evening_w = np.random.uniform(0.5, 4.0)
        evening_d = np.random.uniform(2.0, 10.0)
        
        profile = model_load_profiles_realistic(
            hours, idle, morning_h, morning_w, midday_h, midday_w, 
            evening_h, evening_w, evening_d
        )
        
        # Add random noise
        noise = np.random.normal(0, 0.1, hours)
        profile = profile + noise
        
        # Clip to realistic values
        load_profiles[house] = np.clip(profile, 0.2, 3.5)
    
    df = pd.DataFrame(load_profiles).T
    df.columns = [f'hour_{i}' for i in range(hours)]
    df.index.name = 'house_id'
    
    return df

def save_load_profiles(load_df, output_path, prefix='house', start_id=1, transformer_id=None):
    """
    Save load profiles to CSV with customizable labeling.
    
    Args:
        load_df: DataFrame with load profiles
        output_path: Path to save CSV file
        prefix: Prefix for house IDs (default: 'house')
        start_id: Starting ID number (default: 1)
        transformer_id: Optional transformer ID to include in filename
    """
    # Create a copy to avoid modifying the original
    df_save = load_df.copy()
    
    # Rename index if needed
    if df_save.index.name != f'{prefix}_id':
        # Create new index labels
        new_index = [f'{prefix}_{i}' for i in range(start_id, start_id + len(df_save))]
        df_save.index = new_index
        df_save.index.name = f'{prefix}_id'
    
    # Add transformer column if specified
    if transformer_id is not None:
        df_save['transformer_id'] = transformer_id
    
    # Save to CSV
    filename = f"load_profiles_{transformer_id}.csv" if transformer_id else "load_profiles.csv"
    full_path = f"{output_path}/{filename}"
    df_save.to_csv(full_path)
    print(f"Saved load profiles to {full_path}")
    
    return full_path

# Example usage
if __name__ == "__main__":
    house_ids = [f'house_{i}' for i in range(1, 31)]
    df = generate_hourly_load_profiles_realistic(house_ids)
    print(df.head())
