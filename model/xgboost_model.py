import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
import json
import os
import numpy as np
from tqdm import tqdm

model_path = os.path.join(os.path.dirname(__file__),
                          '..', 'best_model.json')


def train_xgboost_model(x_train, y_train):
    """
    Train the XGBoost model with hyperparameter tuning.
    """
    # Define hyperparameter grid for tuning
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    # Instantiate the model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit the model
    grid_search.fit(x_train, y_train)

    # Print the best parameters found
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Return the best model found by GridSearchCV
    save_best_model(grid_search.best_estimator_, grid_search.best_score_)
    return grid_search.best_estimator_, grid_search.best_score_


def save_best_model(model, mae):
    """
    Save the best model in JSON format if the current model has a lower MAE.
    """
    # Check if the model already exists
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            saved_model = json.load(f)
        saved_mae = saved_model.get('mae', float('inf'))
    else:
        saved_mae = float('inf')

    # If the current model has a lower MAE, save it
    print(f"Current model MAE: {mae}")
    if mae < saved_mae:  # Lower MAE is better, so we save if it's lower
        print(f"New best model found (MAE: {mae}), saving...")

        # Save the model
        model.get_booster().save_model(f"best_model_{mae:.2f}.json")

        # Save the model's performance metric in a separate JSON file
        saved_model = {
            'mae': mae,
            'model_filename': f"best_model_{mae:.2f}.json"
        }

        with open(model_path, 'w') as f:
            json.dump(saved_model, f)

        print(f"Model saved as best_model_{mae:.2f}.json")
    else:
        print("No improvement in model, skipping save.")
