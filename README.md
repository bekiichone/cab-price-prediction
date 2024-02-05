# Uber Price Prediction Project

## Project Overview

This project aims to predict Uber ride prices based on various features such as distance, time, weather conditions, and more. The goal is to understand the factors influencing ride prices and build a predictive model for accurate fare estimates.

## Data Source

The dataset used in this project is sourced from the [Monash Time Series Forecasting Repository](https://zenodo.org/records/5122114), which sourced from [Uber & Lyft Cab prices](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices/data). It simulates ride data with real prices, allowing for the exploration of factors affecting ride costs.

Context:
> Uber and Lyft's ride prices are not constant like public transport. They are greatly affected by the demand and supply of rides at a given time. So what exactly drives this demand? The first guess would be the time of the day; times around 9 am and 5 pm should see the highest surges on account of people commuting to work/home. Another guess would be the weather; rain/snow should cause more people to take rides.

## Project Structure

The project follows the [Cookiecutter Data Science directory structure](https://drivendata.github.io/cookiecutter-data-science/):

```plaintext
├── data
│   └── raw             # Raw data (Monash Time Series Forecasting Repository)
├── models              # Trained machine learning models
├── notebooks           # Jupyter notebooks for analysis and visualization
    └── exploratory.ipynb    # Here I run the experiments 
├── reports             # Project reports, documentation, and results
├── src
│   ├── data            # Data processing scripts
        └── make_dataset.py    # Build dataset for training
│   ├── features        # Feature engineering scripts
        └── build_features.py    # Preprocess and feature engineering of dataset
│   ├── models          # Model training and evaluation scripts
        └── train_model.py    # Model training pipeline
        └── predict_model.py    # Model inferenct pipeline
│   └── utils           # Utility functions
└── tests               # Unit tests for project functions
```
## Usage

1. **Install Dependencies**: Run `pip install -r requirements.txt` to install project dependencies.
2. **Run Notebooks**: Explore Jupyter notebook in the `notebooks` directory to explore EDA, model training and etc.
3. **Train Models**: Execute `train_model.py` in the `src/models` directory to train and save machine learning models.
4. **Predictions**: Utilize `predict_model.py` for making predictions based on the trained models.
