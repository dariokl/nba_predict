import xgboost as xgb
import os
import json

from data.feature_engineering import prepare_features_with_rolling_averages
from utils.labels import rolling_average_labels

model = os.path.join(os.path.dirname(__file__),
                     '..', 'best_model_1.00.json')


def predict_for_player(player_id, threshold):
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

    will_score_above = predicted_points[-1] > threshold

    # Calculate the deviation from the threshold
    deviation = predicted_points[-1] - threshold
    confidence = max(0, 100 - abs(deviation) * 10)

    print(
        f"Will player {player_id} score above {threshold} points? {will_score_above}")
    print(f"Predicted points: {predicted_points[-1]}")
    print(f"Confidence: {confidence:.2f}%")

    return will_score_above, predicted_points[-1], confidence
