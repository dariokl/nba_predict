import xgboost as xgb
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import train_test_split
import os


def train_xgboost_model(x_train, y_train):
    """
    Train the XGBoost model with hyperparameter tuning.
    """
    # Define hyperparameter grid for tuning
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)
    # Instantiate the model
    xgb_model = xgb.XGBRegressor(
        objective='reg:pseudohubererror')

    rmse_scorer = make_scorer(root_mean_squared_error)

    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(x_train, y_train, eval_set=[
                    (x_train, y_train), (x_test, y_test)], verbose=100)

    # Print the best parameters found
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Return the best model found by GridSearchCV
    save_model(grid_search.best_estimator_, grid_search.best_score_)


def save_model(model, rmse):
    """
    Save the best model in JSON format if the current model has a lower MAE.
    """

    model_path = os.path.join(os.path.dirname(__file__),
                              '../..', f'model_{rmse}.json')

    # Save the model
    model.get_booster().save_model(model_path)

    # Save the model's performance metric in a separate JSON file

    print(f"Model saved as model_{rmse}.json")
