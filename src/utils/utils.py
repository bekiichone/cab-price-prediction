# src/utils/utils.py
import numpy as np
import pickle  # Import pickle for model loading and saving

def save_model(model, model_path):
    """
    Save a machine learning model using pickle.
    """
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(model_path):
    """
    Load a machine learning model using pickle.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Add any other utility functions you may need
