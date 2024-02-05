# tests/test_make_dataset.py
import sys
sys.path.append('..')

import pandas as pd
import pytest
from something_analysis.src.data.make_dataset import preprocess_cab_rides, preprocess_weather, merge_data

@pytest.fixture 
def sample_cab_rides_data():
    data = {
        'distance': [10.0, 15.0, 20.0],
        'cab_type': ['A', 'B', 'A'],
        'time_stamp': [1544952607890, 1543284023677, 1543366822198],
        'price': [25.0, 30.0, 40.0],
        'source': ['X', 'Y', 'Z'],
        'destination': ['Y', 'Z', 'X'],
        'id': [1, 2, 3]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_weather_data():
    data = {
        'location': ['X', 'Y', 'Z'],
        'time_stamp': [1544952607, 1543284023, 1543366822],
        'rain': [0.1, 0.0, 0.2]
    }
    return pd.DataFrame(data)

def test_preprocess_cab_rides(sample_cab_rides_data):
    # Test the preprocess_cab_rides function

    # Given data with missing prices
    processed_data = preprocess_cab_rides(sample_cab_rides_data)

    # Assertions
    assert len(processed_data) == 3  # Check if rows with missing price are dropped
    assert 'time_stamp' in processed_data.columns  # Check if 'time_stamp' column exists
    assert 'id' not in processed_data.columns  # Check if 'id' column is dropped
    assert pd.to_datetime(processed_data['time_stamp']).dt.year.min() > 1970  # Check if 'time_stamp' is converted to datetime

    # Check if non-null prices are preserved
    assert processed_data['price'].notna().all()

    # Check if 'price' is of float type
    assert processed_data['price'].dtype == float

def test_preprocess_weather(sample_weather_data):
    # Test preprocess_weather function
    processed_data = preprocess_weather(sample_weather_data)
    
    # Add assertions based on the expected behavior of preprocess_weather
    assert 'hourly_time_stamp' in processed_data.columns  # Ensure 'hourly_time_stamp' is created
    assert processed_data['rain'].notna().all()  # Ensure 'rain' has no missing values
    assert processed_data['rain'].eq([0.1, 0.0, 0.2]).all()  # Ensure 'rain' values are unchanged


def test_merge_data(sample_cab_rides_data, sample_weather_data):
    # Test merge_data function
    cab_rides_data = preprocess_cab_rides(sample_cab_rides_data)
    weather_data = preprocess_weather(sample_weather_data)
    merged_data = merge_data(cab_rides_data, weather_data)
    
    # Add assertions based on the expected behavior of merge_data
    assert 'rain_source' in merged_data.columns  # Ensure the merged columns are present
    assert 'rain_destination' in merged_data.columns


