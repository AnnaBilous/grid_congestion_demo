import numpy as np
import pandas as pd


def model_load_profiles_realistic(hours=24, idle_consumption=0.3, 
                                 morning_peak_height=1.0, morning_peak_width=1.5, 
                                 midday_peak_height=0.7, midday_peak_width=2.0,
                                 evening_peak_height=1.5, evening_peak_width=1.8, 
                                 evening_decay_time=3.0):
    """
    Generate a realistic residential electricity load profile.
    
    Creates a 24-hour load profile by combining four components:
    - Constant base/idle load
    - Morning peak (typically breakfast time)
    - Midday peak (lunch time)
    - Evening peak with asymmetric decay (dinner and evening activities)
    
    Parameters
    ----------
    hours : int, default=24
        Number of hours in the profile
    idle_consumption : float, default=0.3
        Base load in kW that's constant throughout the day
    morning_peak_height : float, default=1.0
        Height of morning peak in kW
    morning_peak_width : float, default=1.5
        Width (standard deviation) of morning peak in hours
    midday_peak_height : float, default=0.7
        Height of midday peak in kW
    midday_peak_width : float, default=2.0
        Width (standard deviation) of midday peak in hours
    evening_peak_height : float, default=1.5
        Height of evening peak in kW
    evening_peak_width : float, default=1.8
        Width (standard deviation) of evening peak in hours
    evening_decay_time : float, default=3.0
        Decay time constant for evening peak in hours
        
    Returns
    -------
    numpy.ndarray
        Hourly load profile values in kW
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
    Generate synthetic hourly load profiles for multiple houses.
    
    Creates realistic load profiles with random variations for each house,
    simulating different household behaviors and consumption patterns.
    
    Parameters
    ----------
    house_ids : list
        List of house identifiers
    hours : int, default=24
        Number of hours in each profile
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with houses as rows and hourly loads as columns.
        Index is house_id and columns are hour_0 through hour_23.
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
    
    Exports the load profiles DataFrame to a CSV file with options to
    customize the naming and add transformer information.
    
    Parameters
    ----------
    load_df : pandas.DataFrame
        DataFrame containing load profiles
    output_path : str
        Directory path where the CSV file will be saved
    prefix : str, default='house'
        Prefix for house IDs in the index
    start_id : int, default=1
        Starting ID number for houses
    transformer_id : str, optional
        Transformer ID to include in filename and as a column
        
    Returns
    -------
    str
        Full path to the saved CSV file
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
