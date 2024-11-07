import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
import json
import os

model_path = os.path.join(os.path.dirname(__file__),
                          '..', 'best_model.json')


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
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, early_stopping_rounds=50)

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
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    return mae


def save_best_model(model, mae):
    """
    Save the best model in JSON format if the current model has a lower MAE.
    """
    # Load the previous best model's MAE if it exists
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            saved_model = json.load(f)
        saved_mae = saved_model.get('mae', float('inf'))
        print(saved_mae, '---att savedd')
    else:
        saved_mae = float('inf')

    # If current model's MAE is better (lower), save the model and MAE
    print(f"Current model MAE: {mae}")

    if mae < saved_mae:
        print("New best model found, saving...")

        # Save model in JSON format and store MAE in the same file
        model.get_booster().save_model('temp_model.json')
        with open('temp_model.json', 'r') as model_file:
            model_json = json.load(model_file)

        # Add MAE to the JSON data
        model_json['mae'] = mae

        # Save updated JSON with model and MAE
        with open(model_path, 'w') as f:
            json.dump(model_json, f)

        print(f"Model saved with MAE: {mae}")
        os.remove('temp_model.json')  # Clean up temporary file
    else:
        print("No improvement in model, skipping save.")
