# src/models/train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
import logging
import click

from src.utils.utils import save_model
from src.data.make_dataset import load_data

logging.basicConfig(level=logging.INFO)

def create_time_series_split(data, n_splits=5):
    """
    Create TimeSeriesSplit object for time series cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(data)

def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def evaluate_model(model, X, y):
    """
    Evaluate the model using SMAPE.
    """
    y_pred = model.predict(X)
    smape = calculate_smape(y, y_pred)
    return smape

def train_model(data, model_name):
    """
    Train a specific machine learning model with normalization,
    evaluate, and save the best model using RandomizedSearchCV.
    """

    # Split the data into features (X) and target variable (y)
    X = data.drop('price', axis=1)
    y = data['price']

    # Create TimeSeriesSplit for time series cross-validation
    tscv = create_time_series_split(data)

    # Define the model and hyperparameter distributions for the specified model
    model_configs = {
        'RandomForest': (RandomForestRegressor(), {'model__regressor__n_estimators': [50, 100, 200], 
                                                   'model__regressor__max_depth': [None, 10, 20]}),
        'LightGBM': (LGBMRegressor(), {'model__regressor__n_estimators': [50, 100, 200], 'model__regressor__max_depth': [5, 10, 15],
                                       'model__regressor__learning_rate': [0.01, 0.05, 0.1], 'model__regressor__num_leaves': [31, 50, 100],
                                       'model__regressor__min_child_samples': [20, 50, 100]}),
        'NeuralNetwork': (MLPRegressor(), {'model__regressor__hidden_layer_sizes': [(50,), (100,)], 
                                           'model__regressor__alpha': [0.0001, 0.001, 0.01]})
    }

    model, param_dist = model_configs[model_name]

    logging.info(f'Tuning hyperparameters for {model_name} using RandomizedSearchCV...')

    # Create a pipeline with normalization using StandardScaler
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1))
    ])

    # RandomizedSearchCV for hyperparameter tuning
    randomized_search = RandomizedSearchCV(
        model_pipeline,
        param_dist,
        n_iter=10,  # Adjust the number of iterations as needed
        scoring=make_scorer(calculate_smape, greater_is_better=False),
        cv=tscv,
        n_jobs=-1
    )

    # Perform randomized search
    randomized_search.fit(X, y)

    # Get the best model from randomized search
    best_model_from_search = randomized_search.best_estimator_

    # Get the best hyperparameters
    best_params = randomized_search.best_params_

    logging.info(f'Best hyperparameters for {model_name}: {best_params}')

    # Get the best candidate index

    best_index = randomized_search.best_index_  

    # Get test scores for each split
    smape_scores = []
    for i in range(randomized_search.n_splits_):
        smape_scores.append(-randomized_search.cv_results_[f'split{i}_test_score'][best_index])
    smape_scores = np.array(smape_scores)

    # Calculate mean SMAPE across folds
    mean_smape = smape_scores.mean()
    std_smape = smape_scores.std()

    logging.info(f'Mean SMAPE for {model_name}: {mean_smape}')
    logging.info(f'Std SMAPE for {model_name}: {std_smape}')

    # Log metrics
    logging.info(f'Metrics for {model_name}: {smape_scores}')

    # Save the best model
    save_model(best_model_from_search, f'{model_name}.pkl')
    logging.info(f'Best {model_name} model saved.')

@click.command()
@click.option('--model_name', type=click.Choice(['RandomForest', 'XGBoost', 'LinearRegression', 'NeuralNetwork']),
              help='Specify the model to train and evaluate.')
def main(model_name):
    # Load the processed data
    processed_data = load_data('../../data/processed_data.csv')

    # Train the specified machine learning model with normalization,
    # evaluate, and save the best model using RandomizedSearchCV
    train_model(processed_data, model_name)

if __name__ == "__main__":
    main()
