# src/models/predict_model.py
import pandas as pd
from src.features.build_features import feature_engineering
from src.utils.utils import load_model

def predict_model(data, model_name, output_path='../../data/predictions/predictions.csv'):

    # Load the trained model
    model_path = f'{model_name}.pkl'
    trained_model = load_model(model_path)

    # Make predictions
    predictions = trained_model.predict(data)

    # Add predictions to the new data
    data['predicted_price'] = predictions

    # Save the predictions to a CSV file
    data.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Make predictions using a trained model.')
    parser.add_argument('--new_data_path', type=str, help='Path to the new data CSV file.')
    parser.add_argument('--model_name', type=str, help='Name of the trained model (without the .pkl extension).')

    args = parser.parse_args()

    # Make predictions using the specified model
    predict_model(args.new_data_path, args.model_name)
