import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_data(num_days=30):
    """
    Generate dummy order data for the specified number of days
    
    Parameters:
    -----------
    num_days : int
        Number of days to generate data for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: timestamp, jumlah_order, lokasi
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # End date (today)
    end_date = datetime.now().replace(hour=23, minute=0, second=0, microsecond=0)
    
    # Start date (num_days ago)
    start_date = end_date - timedelta(days=num_days)
    
    # Generate hourly timestamps
    hours = num_days * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Generate order counts with time-based patterns
    base_orders = 50  # Base number of orders
    
    # Create patterns
    hourly_pattern = np.array([0.5, 0.3, 0.2, 0.1, 0.1, 0.2,  # 0-5 AM (low)
                              0.7, 1.5, 2.0, 1.5, 1.2, 1.5,  # 6-11 AM (morning peak)
                              1.8, 1.7, 1.3, 1.2, 1.5, 2.0,  # 12-5 PM (afternoon)
                              2.5, 2.2, 1.8, 1.5, 1.0, 0.7])  # 6-11 PM (evening peak)
    
    # Day of week pattern (Monday=0, Sunday=6)
    dow_pattern = np.array([1.0, 0.9, 0.9, 1.0, 1.1, 1.5, 1.3])  # Weekends higher
    
    # Generate order counts
    orders = []
    for ts in timestamps:
        hour_factor = hourly_pattern[ts.hour]
        dow_factor = dow_pattern[ts.weekday()]
        
        # Add some randomness
        random_factor = np.random.normal(1, 0.2)  # Mean=1, SD=0.2
        
        # Calculate orders with some randomness
        order_count = int(base_orders * hour_factor * dow_factor * random_factor)
        
        # Ensure non-negative
        order_count = max(0, order_count)
        
        orders.append(order_count)
    
    # Generate random locations
    locations = ['Jakarta Pusat', 'Jakarta Barat', 'Jakarta Selatan', 
                'Jakarta Timur', 'Jakarta Utara', 'Depok', 'Tangerang', 'Bekasi']
    
    # Assign locations with different weights
    location_weights = [0.2, 0.15, 0.25, 0.15, 0.1, 0.05, 0.05, 0.05]
    random_locations = np.random.choice(locations, size=len(timestamps), p=location_weights)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'jumlah_order': orders,
        'lokasi': random_locations
    })
    
    return df

# For testing
if __name__ == "__main__":
    data = generate_dummy_data()
    print(f"Generated {len(data)} records")
    print(data.head())