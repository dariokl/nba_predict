import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


def train_gradient_boosting_model(x_train, y_train):
    """
    Train the GradientBoostingRegressor model with hyperparameter tuning,
    and save the model, hyperparameters, and performance data.
    """
    # Define hyperparameter grid for tuning
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'subsample': [0.7, 0.8],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'max_features': ['auto', 'sqrt'],
    }

    # Ensure data is sorted by time before splitting
    x_train = x_train.sort_index()
    y_train = y_train.loc[x_train.index]  # Align labels with features

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, shuffle=False, random_state=42
    )

    # Define time series cross-validation strategy
    tscv = TimeSeriesSplit(n_splits=5)

    # Instantiate the GradientBoostingRegressor model
    gb_model = GradientBoostingRegressor()

    # Create GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=gb_model,
        param_grid=param_grid,
        cv=tscv,
        # You can change scoring depending on your needs
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model using GridSearchCV
    grid_search.fit(x_train, y_train)

    # Print best parameters
    print("Best Hyperparameters found: ", grid_search.best_params_)

    # Evaluate on the test set
    y_pred = grid_search.best_estimator_.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test Set MAE: {mae:.4f}")

    # Save the model and configuration
    save_gradient_boosting_model(
        grid_search.best_estimator_, grid_search.best_params_, mae)


def save_gradient_boosting_model(model, mae):
    """
    Save the best model, hyperparameters, and MAE into the specified directory.
    """

    # Save the model
    model_path = os.path.join(os.path.dirname(
        __file__),
        '../..', f"gradient_boosting_model_{mae:.4f}.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved at {model_path}")
