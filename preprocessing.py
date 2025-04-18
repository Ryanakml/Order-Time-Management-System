import pandas as pd
import numpy as np

def preprocess_data(data, agg_level='hourly'):
    """
    Preprocess time series data for modeling
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with at least 'timestamp' and 'jumlah_order' columns
    agg_level : str
        Aggregation level: 'hourly', 'daily', or 'weekly'
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame in Prophet format (ds, y)
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Aggregate data based on specified level
    if agg_level == 'hourly':
        # Already hourly, just rename columns for Prophet
        result = df[['timestamp', 'jumlah_order']].rename(
            columns={'timestamp': 'ds', 'jumlah_order': 'y'}
        )
    
    elif agg_level == 'daily':
        # Aggregate to daily level
        result = df.groupby(df['timestamp'].dt.date).agg(
            {'jumlah_order': 'sum'}
        ).reset_index()
        result.rename(columns={'timestamp': 'ds', 'jumlah_order': 'y'}, inplace=True)
        result['ds'] = pd.to_datetime(result['ds'])
    
    elif agg_level == 'weekly':
        # Create week start date
        df['week_start'] = df['timestamp'] - pd.to_timedelta(df['timestamp'].dt.dayofweek, unit='D')
        
        # Aggregate to weekly level
        result = df.groupby('week_start').agg(
            {'jumlah_order': 'sum'}
        ).reset_index()
        result.rename(columns={'week_start': 'ds', 'jumlah_order': 'y'}, inplace=True)
    
    else:
        raise ValueError(f"Unsupported aggregation level: {agg_level}")
    
    return result

# Function to prepare data for LSTM (if needed)
def prepare_for_lstm(data, n_steps=24):
    """
    Prepare time series data for LSTM model by creating sequences
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with 'ds' and 'y' columns (output from preprocess_data)
    n_steps : int
        Number of time steps to use for each sequence
        
    Returns:
    --------
    tuple
        (X, y) where X is the input sequences and y is the target values
    """
    # Extract the target variable
    values = data['y'].values
    
    # Create sequences
    X, y = [], []
    for i in range(len(values) - n_steps):
        X.append(values[i:i+n_steps])
        y.append(values[i+n_steps])
    
    return np.array(X), np.array(y)