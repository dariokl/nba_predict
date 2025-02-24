import os
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from app.data_processing.feature_engineering import prepare_features_with_rolling_averages
from app.utils.labels import rolling_average_labels

MODEL_PATH = os.path.join(os.path.dirname(
    __file__), '../../gradient_boosting_model_0.3511.pkl')


def load_model():
    """Load the Gradient Boosting model from the specified path."""
    model = joblib.load(MODEL_PATH)
    return model


def preprocess_data(player_id):
    """Prepares and scales features for a given player."""
    games_df = prepare_features_with_rolling_averages(player_id=player_id)
    if len(games_df) > 5:
        games_df = games_df.tail(5)
    x_player = games_df[rolling_average_labels]
    return x_player


def dynamic_alpha(data):
    """Adjust alpha dynamically based on player performance variability."""
    std_dev = np.std(data)
    return min(0.8, max(0.2, 1 - std_dev / np.mean(data)))


def exponential_moving_average(data, span=3):
    """Smooth trend with an exponential moving average."""
    return pd.Series(data).ewm(span=span, adjust=False).mean().iloc[-1]


def compute_confidence(deviation, predicted_points):
    """Compute confidence dynamically with improved scaling."""

    max_deviation = max(
        np.max(np.abs(predicted_points - np.mean(predicted_points))), 1)

    normalized_deviation = abs(deviation) / max_deviation
    confidence = 100 * (1 - normalized_deviation)

    # Clip the confidence value to ensure it remains within [0, 100]
    confidence = np.clip(confidence, 0, 100)

    return confidence


def predict_for_player_trend(player_id, betline):
    """Predict whether a player will score above a certain betline using trend analysis."""
    model = load_model()
    x_player = preprocess_data(player_id)
    predicted_points = model.predict(x_player)
    trend_predicted_points = exponential_moving_average(predicted_points)
    deviation = trend_predicted_points - betline
    confidence = compute_confidence(deviation, predicted_points)
    return trend_predicted_points > betline, trend_predicted_points, confidence


def backtest_trend_predict(games_df, betline):
    """Backtest the trend-based prediction for a given games dataframe."""
    model = load_model()
    if len(games_df) > 5:
        games_df = games_df.tail(5)
    x_player = games_df[rolling_average_labels]
    predicted_points = model.predict(x_player)
    trend_predicted_points = exponential_moving_average(predicted_points)
    deviation = trend_predicted_points - betline
    confidence = compute_confidence(deviation, predicted_points)
    return trend_predicted_points > betline, trend_predicted_points, confidence
