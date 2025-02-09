import xgboost as xgb
import os
import numpy as np
import joblib

from app.data.feature_engineering import prepare_features_with_rolling_averages
from app.utils.labels import rolling_average_labels

model = os.path.join(os.path.dirname(__file__),
                     '../../..', 'model_-0.0011.json')

scaler_file = os.path.join(os.path.dirname(__file__),
                           '../../..', 'scaler.pkl')
scaler = joblib.load(scaler_file)


def predict_for_player_mean(player_id, betline):
    """
    Predict whether a player will score above a certain betline.
    """
    best_model = xgb.XGBRegressor()
    best_model.load_model(model)

    games_df = prepare_features_with_rolling_averages(
        player_id=player_id,
    )

    if len(games_df) > 5:
        games_df = games_df.tail(5)

    X_player = games_df[rolling_average_labels]
    X_player = scaler.transform(X_player)

    predicted_points = best_model.predict(X_player)

    mean_predicted_points = np.mean(predicted_points)

    deviation = mean_predicted_points - betline
    confidence = max(0, 100 - abs(deviation) * 10)

    will_score_above = mean_predicted_points > betline

    return will_score_above, mean_predicted_points, confidence


def predict_for_player_trend(player_id, betline):
    """
    Predict whether a player will score above a certain betline.
    """
    best_model = xgb.XGBRegressor()
    best_model.load_model(model)

    games_df = prepare_features_with_rolling_averages(
        player_id=player_id
    )

    if len(games_df) > 5:
        games_df = games_df.tail(5)

    X_player = games_df[rolling_average_labels]
    X_player = scaler.transform(X_player)
    predicted_points = best_model.predict(X_player)
    alpha = 0.5
    weights = (1 - alpha) ** np.arange(len(predicted_points))[::-1]
    trend_predicted_points = np.dot(weights, predicted_points) / weights.sum()

    deviation = trend_predicted_points - betline
    confidence = 1 - min(1, abs(deviation) / max(10, abs(betline)))

    will_score_above = trend_predicted_points > betline
    return will_score_above, trend_predicted_points, confidence * 100


def predict_for_player_ema(player_id, betline):

    best_model = xgb.XGBRegressor()
    best_model.load_model(model)

    games_df = prepare_features_with_rolling_averages(
        player_id=player_id
    )

    if len(games_df) > 5:
        games_df = games_df.tail(5)

    X_player = games_df[rolling_average_labels]

    print(games_df['PTS'], games_df['DAYS_SINCE_LAST_GAME'])

    predicted_points = best_model.predict(X_player)

    alpha = 0.3
    ema_predicted_points = np.zeros_like(predicted_points)
    ema_predicted_points[0] = predicted_points[0]
    for i in range(1, len(predicted_points)):
        ema_predicted_points[i] = alpha * predicted_points[i] + \
            (1 - alpha) * ema_predicted_points[i - 1]
    final_ema_prediction = ema_predicted_points[-1]

    # Confidence metric
    deviation = final_ema_prediction - betline
    confidence = 1 - min(1, abs(deviation) / max(10, abs(betline)))

    # Final prediction
    will_score_above = final_ema_prediction > betline
    return will_score_above, final_ema_prediction, confidence * 100


def backtest_trend_predict(games_df, betline):

    best_model = xgb.XGBRegressor()
    best_model.load_model(model)

    if len(games_df) > 5:
        games_df = games_df.tail(5)

    X_player = games_df[rolling_average_labels]
    X_player = scaler.transform(X_player)
    predicted_points = best_model.predict(X_player)
    alpha = 0.5
    weights = (1 - alpha) ** np.arange(len(predicted_points))[::-1]
    trend_predicted_points = np.dot(weights, predicted_points) / weights.sum()

    deviation = trend_predicted_points - betline
    confidence = 1 - min(1, abs(deviation) / max(10, abs(betline)))

    will_score_above = trend_predicted_points > betline
    return will_score_above, trend_predicted_points, confidence * 100
