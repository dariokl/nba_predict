import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import os
import joblib


def train_xgboost_model(x_train, y_train):
    """
    Train the XGBoost model with hyperparameter tuning.
    """

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [50, 100],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 5],
    }

    x_train = x_train.sort_index()
    y_train = y_train.loc[x_train.index]

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, shuffle=False, random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)

    xgb_model = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        verbosity=1
    )

    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=1
    )

    save_model(grid_search.best_estimator_, grid_search.best_score_)


def save_model(model, score):
    """
    Save the best model in JSON format if the current model has a lower MAE.
    """
    model_path = os.path.join(os.path.dirname(
        __file__), '../..', f'model_{score:.4f}.json')

    model.get_booster().save_model(model_path)

    print(f"Model saved as model_{score:.4f}.json")
