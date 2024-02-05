# tests/test_build_features.py
import sys
sys.path.append('..')

import pandas as pd
import pytest
from something_analysis.src.features.build_features import encode_categorical, create_time_features, create_distance_related_features, fill_missing_values, feature_engineering

@pytest.fixture
def sample_processed_data():
    # Create a sample DataFrame for processed data
    # Modify this based on your actual data schema
    data = {
        'distance': [10.0, 15.0, 20.0],
        'cab_type': ['A', 'B', 'A'],
        'time_stamp': ['2022-01-01 10:00:00', '2022-01-01 11:00:00', '2022-01-01 12:00:00'],
        'price': [25.0, 30.0, 40.0],
        'rain_source': [None, 0.12, 0.09],
        'rain_destination': [0.25, None, 0.50]
        # Add other columns as needed
    }
    data['time_stamp'] = pd.to_datetime(data['time_stamp'])
    return pd.DataFrame(data)

def test_encode_categorical(sample_processed_data):
    # Test encode_categorical function
    processed_data = encode_categorical(sample_processed_data, ['cab_type'])
    
    # Add assertions based on the expected behavior of encode_categorical
    assert 'cab_type' in processed_data.columns  # Ensure 'cab_type' column is encoded
    assert pd.api.types.is_numeric_dtype(processed_data['cab_type'])  # Ensure encoded column is numerical

def test_create_time_features(sample_processed_data):
    # Test create_time_features function
    processed_data = create_time_features(sample_processed_data, 'time_stamp')
    
    # Add assertions based on the expected behavior of create_time_features
    assert 'hour_of_day' in processed_data.columns  # Ensure 'hour_of_day' is created
    assert 'day_of_week' in processed_data.columns  # Ensure 'day_of_week' is created


def test_create_distance_related_features(sample_processed_data):
    # Test create_distance_related_features function
    processed_data = create_distance_related_features(sample_processed_data)
    
    # Add assertions based on the expected behavior of create_distance_related_features
    assert 'distance_squared' in processed_data.columns  # Ensure 'distance_squared' is created
    assert 'inverse_distance' in processed_data.columns  # Ensure 'inverse_distance' is created

def test_fill_missing_values(sample_processed_data):
    # Test fill_missing_values function
    processed_data = fill_missing_values(sample_processed_data)
    
    # Add assertions based on the expected behavior of fill_missing_values
    assert processed_data['rain_source'].notna().all()  # Ensure 'rain_source' has no missing values

