# Define the content for each file

import pandas as pd

def prepare_data(input_file, output_file):
    print("Step 1: Preparing Data...")
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Select core columns
    core_cols = ['DATE', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']
    df_clean = df[core_cols].copy()
    
    # Fill missing values (Forward Fill)
    df_clean = df_clean.ffill()
    
    # Feature Engineering
    # 1. Rolling Mean (30 days)
    df_clean['rolling_30_TMAX'] = df_clean['TMAX'].rolling(window=30).mean()
    
    # 2. Monthly and Daily Averages
    df_clean['month'] = df_clean['DATE'].dt.month
    df_clean['day_of_year'] = df_clean['DATE'].dt.dayofyear
    
    # Note: Using transform on the whole dataset for simplicity. 
    # For strict time-series forecasting, consider expanding windows to avoid leakage.
    monthly_avg = df_clean.groupby('month')['TMAX'].transform('mean')
    df_clean['monthly_avg_TMAX'] = monthly_avg
    
    daily_avg = df_clean.groupby('day_of_year')['TMAX'].transform('mean')
    df_clean['daily_avg_TMAX'] = daily_avg
    
    # 3. Create Target (Next day's TMAX)
    df_clean['target'] = df_clean['TMAX'].shift(-1)
    
    # Drop rows with NaNs created by rolling/shifting
    df_clean = df_clean.dropna()
    
    # Save processed data
    df_clean.to_csv(output_file, index=False)
    print(f"Data prepared and saved to {output_file}")

if __name__ == "__main__":
    prepare_data("JFK Airport Weather Data.csv", "processed_weather_data.csv")
