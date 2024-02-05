# src/features/build_features.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df, columns):
    """
    Encode categorical columns using Label Encoding.
    """
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

def create_time_features(df, time_column):
    """
    Create sophisticated time-related features.
    """
    df['hour_of_day'] = df[time_column].dt.hour
    df['day_of_week'] = df[time_column].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour_of_day'].between(7, 9) | df['hour_of_day'].between(16, 18)
    df['time_of_day'] = pd.cut(df['hour_of_day'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False)
    df.drop(time_column, axis=1, inplace=True)
    return df

def create_distance_related_features(df):
    """
    Create advanced features related to distance.
    """
    df['distance_squared'] = df['distance'] ** 2
    df['inverse_distance'] = 1 / (df['distance'] + 1)  # Adding 1 to avoid division by zero
    return df

def fill_missing_values(df):
    """
    Fill missing values with advanced strategies.
    """
    # Impute missing values based on statistical measures
    df = df.sort_values('time_stamp').reset_index(drop=True)
    df['rain_source'].fillna(0, inplace=True)
    df['rain_destination'].fillna(0, inplace=True)
    df.interpolate(inplace=True, limit_direction='forward')
    
    return df

def feature_engineering(df):
    """
    Perform sophisticated feature engineering on the given DataFrame.
    """
    # Encode categorical columns
    categorical_columns = ['cab_type', 'destination', 'source', 'product_id', 'name']
    df = encode_categorical(df, categorical_columns)

    # Fill missing values using advanced strategies
    df = fill_missing_values(df)

    # Create sophisticated time-related features
    df = create_time_features(df, 'time_stamp')

    # Create advanced distance-related features
    df = create_distance_related_features(df)

    # dropping NA again because weather data was not available for first 77 rows
    df = df.dropna().reset_index(drop=True)

    return df

def main():
    # Load processed data
    interim_data = pd.read_csv('../../data/interim_data.csv')

    # Perform sophisticated feature engineering
    processed_data = feature_engineering(interim_data)

    # Save the sophisticated engineered data
    processed_data.to_csv('../../data/processed_data.csv', index=False)

if __name__ == "__main__":
    main()
