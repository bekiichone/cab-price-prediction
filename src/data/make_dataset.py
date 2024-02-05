# src/data/make_dataset.py
import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load the raw data from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_cab_rides(df):
    """
    Preprocess the cab rides data.
    """
    cab_rides_data = df.copy()

    # Drop rows with missing values in the 'price' column
    cab_rides_data = cab_rides_data.dropna(subset=['price'])

    # Convert 'time_stamp' to datetime
    cab_rides_data['time_stamp'] = pd.to_datetime(cab_rides_data['time_stamp'], unit='ms')

    # Dropping 'id' column since its irrelevant
    cab_rides_data.drop('id', axis=1, inplace=True)

    return cab_rides_data

def preprocess_weather(df):
    """
    Preprocess the weather data.
    """
    weather_data = df.copy()
    # Convert 'time_stamp' to datetime
    weather_data['time_stamp'] = pd.to_datetime(weather_data['time_stamp'], unit='s')
    
    # Round time_stamp to nearest lowest hour
    weather_data['hourly_time_stamp'] = weather_data['time_stamp'].dt.ceil('H')
    weather_data.drop('time_stamp', axis=1, inplace=True)

    # Fill rain na data with 0 (no precipitation)
    weather_data['rain'] = weather_data['rain'].fillna(0)

    # grouping via location and time_stamp
    weather_data = weather_data.groupby(['location', 'hourly_time_stamp']).mean().reset_index()

    return weather_data

def merge_data(cab_rides_data, weather_data):
    """
    Merge the cab rides and weather data on the 'time_stamp' column.
    """
    # Assuming you have 'source' and 'destination' columns in cab_rides
    # and 'time_stamp' column in both cab_rides and weather
    cab_rides = cab_rides_data.copy()
    weather = weather_data.copy()

    # Truncate 'time_stamp' to the nearest hour in both cab_rides and weather
    cab_rides['hourly_time_stamp'] = cab_rides['time_stamp'].dt.floor('H')

    # Merge for source location
    cab_rides_merged_source = pd.merge(cab_rides, weather.add_suffix('_source'), left_on=['source', 'hourly_time_stamp'], right_on=['location_source', 'hourly_time_stamp_source'], how='left')
    cab_rides_merged_source.drop(['location_source', 'hourly_time_stamp_source'], axis=1, inplace=True)

    # Merge for destination location
    cab_rides_merged_dest = pd.merge(cab_rides_merged_source, weather.add_suffix('_destination'), left_on=['destination', 'hourly_time_stamp'], right_on=['location_destination', 'hourly_time_stamp_destination'], how='left')
    cab_rides_merged_dest.drop(['hourly_time_stamp', 'location_destination', 'hourly_time_stamp_destination'], axis=1, inplace=True)

    return cab_rides_merged_dest

def save_processed_data(data, output_path):
    """
    Save the processed data to a CSV file.
    """
    data.to_csv(output_path, index=False)

def main():

    # Load raw data
    cab_rides_data = load_data('../../data/raw/cab_rides.csv')
    weather_data = load_data('../../data/raw/weather.csv')

    # Preprocess data
    cab_rides_data = preprocess_cab_rides(cab_rides_data)
    weather_data = preprocess_weather(weather_data)

    # Merge data
    merged_data = merge_data(cab_rides_data, weather_data)

    # Save processed data
    save_processed_data(merged_data, '../../data/interim_data.csv')

if __name__ == "__main__":
    main()
