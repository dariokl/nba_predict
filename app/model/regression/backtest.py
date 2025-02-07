import os
import sqlite3 as sq
from datetime import datetime
import pandas as pd

from nba_api.stats.static import players

from app.data.feature_engineering import prepare_features_with_rolling_averages
from app.model.regression.predict_regression_model import backtest_trend_predict

db_path = os.path.join(os.path.dirname(__file__),
                       '../../../nba_predict.sqlite')

# Needs more testing


def backtest():
    predictions = get_predictions()

    correct_predictions = 0
    total_predictions = 0

    for prediction in predictions[0: 500]:
        name, predicted_points, betline, scored_points, date = prediction

        # Convert prediction date to datetime format
        prediction_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

        # Get player info
        player = players.find_players_by_full_name(name)
        games_df = prepare_features_with_rolling_averages(player[0]['id'])

        # Ensure GAME_DATE is in datetime format
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

        # Filter for games that happened BEFORE or ON the prediction date
        past_games = games_df[games_df['GAME_DATE'] <= prediction_date]

        # Get the 5 most recent games before or on prediction_date
        last_5_games = past_games.sort_values(
            by='GAME_DATE', ascending=False).head(5)

        over_under, predicted_points, tree = backtest_trend_predict(
            last_5_games, betline)

        print(f"Prediction: {predicted_points}, Betline: {
              betline}, Scored Points: {scored_points}")

        if predicted_points > betline:
            prediction_result = "Over"
        else:
            prediction_result = "Under"

        if float(scored_points) > betline:
            actual_result = "Over"
        else:
            actual_result = "Under"

        if prediction_result == actual_result:
            print(f"Backtest Passed: {name} - Prediction was correct. ({
                  prediction_result} {predicted_points} vs {actual_result} {scored_points})")
            correct_predictions += 1  # Increment correct predictions
        else:
            print(f"Backtest Failed: {name} - Prediction was incorrect. ({
                  prediction_result} {predicted_points} vs {actual_result} {scored_points})")

        total_predictions += 1

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Model Accuracy: {accuracy:.2f}%")
    else:
        print("No predictions to evaluate.")


def get_predictions():
    conn = sq.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT player_name, predicted_points, betline, scored_points, date
        FROM predictions
        where type = 'trend' and scored_points is not NULL
    """

    cursor.execute(query)
    results = cursor.fetchall()  # Fetch all results

    # Close the connection
    conn.close()

    return results
