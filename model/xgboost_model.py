import xgboost as xgb
import os
import json
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.json')


def train_xgboost_model(X_train, y_train):
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
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Print the best parameters found
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Return the best model found by GridSearchCV
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using MAE.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae


def save_best_model(model, mae):
    """
    Save the best model in JSON format if the current model has a lower MAE.
    """
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            saved_model = json.load(f)
        saved_mae = saved_model.get('mae', float('inf'))
    else:
        saved_mae = float('inf')

    print(f"Current model MAE: {mae}")

    if mae < saved_mae:
        print("New best model found, saving...")

        # Save model as JSON and update with MAE
        booster = model.get_booster()
        booster.save_model(model_path)
        model_json = booster.save_raw().decode('utf-8')
        model_data = {"model": model_json, "mae": mae}

        with open(model_path, 'w') as f:
            json.dump(model_data, f)

        print(f"Model saved with MAE: {mae}")
    else:
        print("No improvement in model, skipping save.")
