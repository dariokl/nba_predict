import xgboost as xgb
import os
import numpy as np

from app.data.feature_engineering import prepare_features_with_rolling_averages
from app.utils.labels import rolling_average_labels

model = os.path.join(os.path.dirname(__file__),
                     '../..', 'best_model_5.671712954134959.json')


def predict_for_player_mean(player_id, threshold):
    """
    Predict whether a player will score above a certain threshold.
    """
    best_model = xgb.XGBRegressor()
    best_model.load_model(model)

    games_df = prepare_features_with_rolling_averages(
        player_id=player_id, rolling_window=5
    )

    if len(games_df) > 5:
        games_df = games_df.tail(5)

    X_player = games_df[rolling_average_labels]
    predicted_points = best_model.predict(X_player)

    # Calculate mean prediction across last N games
    mean_predicted_points = np.mean(predicted_points)

    # Calculate deviation from threshold for mean prediction
    deviation = mean_predicted_points - threshold
    confidence = max(0, 100 - abs(deviation) * 10)

    # Decide if the player will score above the threshold

    will_score_above = mean_predicted_points > threshold
    print(
        f"Will player {player_id} score above {threshold} points? {will_score_above}")
    print(f"Mean predicted points: {mean_predicted_points}")
    print(f"Confidence: {confidence:.2f}%")

    return will_score_above, mean_predicted_points, confidence


def predict_for_player_trend(player_id, threshold):
    """
    Predict whether a player will score above a certain threshold.
    """
    best_model = xgb.XGBRegressor()
    best_model.load_model(model)

    games_df = prepare_features_with_rolling_averages(
        player_id=player_id, rolling_window=5
    )

    if len(games_df) > 5:
        games_df = games_df.tail(5)

    X_player = games_df[rolling_average_labels]

    predicted_points = best_model.predict(X_player)

    alpha = 0.5
    weights = (1 - alpha) ** np.arange(len(predicted_points))[::-1]
    trend_predicted_points = np.dot(weights, predicted_points) / weights.sum()
    deviation = trend_predicted_points - threshold
    confidence = max(0, 100 - abs(deviation) * 10)

    will_score_above = trend_predicted_points > threshold
    print(
        f"Will player {player_id} score above {threshold} points? {will_score_above}")
    print(f"Confidence: {confidence:.2f}%")

    return will_score_above, trend_predicted_points, confidence
