import os
import sqlite3 as sq
from datetime import datetime
import pandas as pd

from nba_api.stats.static import players
from app.data_processing.feature_engineering import prepare_features_with_rolling_averages
from app.models.regression_prediction import backtest_trend_predict

db_path = os.path.join(os.path.dirname(__file__), '../../nba_predict.sqlite')


def get_predictions():
    """Fetch stored predictions from the database."""
    query = """
        SELECT player_name, predicted_points, betline, scored_points, date
        FROM predictions
        WHERE type = 'trend' AND scored_points IS NOT NULL
    """

    with sq.connect(db_path) as conn:
        return conn.execute(query).fetchall()


def get_recent_games(player_name, prediction_date):
    """Retrieve last 5 games before or on the prediction date."""
    player = players.find_players_by_full_name(player_name)
    if not player:
        print(f"Warning: Player {player_name} not found.")
        return None

    games_df = prepare_features_with_rolling_averages(player[0]['id'])
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

    return (
        games_df[games_df['GAME_DATE'] < prediction_date]
        .sort_values(by='GAME_DATE', ascending=False)
        .head(5)
    )


def evaluate_prediction(predicted_points, betline, scored_points):
    """Determine the correctness of a prediction."""
    prediction_result = "Over" if predicted_points > betline else "Under"
    actual_result = "Over" if float(scored_points) > betline else "Under"

    return prediction_result == actual_result, prediction_result, actual_result


def backtest():
    predictions = get_predictions()
    correct_predictions = 0

    # Confidence Buckets
    confidence_buckets = {
        "0-10": {"correct": 0, "total": 0},
        "10-20": {"correct": 0, "total": 0},
        "20-30": {"correct": 0, "total": 0},
        "30-40": {"correct": 0, "total": 0},
        "40-50": {"correct": 0, "total": 0},
        "50-60": {"correct": 0, "total": 0},
        "60-70": {"correct": 0, "total": 0},
        "70-80": {"correct": 0, "total": 0},
        "80-90": {"correct": 0, "total": 0},
        "90-100": {"correct": 0, "total": 0},
    }

    if not predictions:
        print("No predictions to evaluate.")
        return

    for name, _, betline, scored_points, date in predictions:
        prediction_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

        recent_games = get_recent_games(name, prediction_date)
        if recent_games is None or recent_games.empty:
            print(f"Skipping {name}: Not enough game data available.")
            continue

        _, predicted_points, confidence = backtest_trend_predict(
            recent_games, betline)
        predicted_points = round(predicted_points, 0)

        correct, prediction_result, actual_result = evaluate_prediction(
            predicted_points, betline, scored_points
        )

        print(
            f"Player: {name} | Prediction: {predicted_points} | Betline: {betline} | "
            f"Scored: {scored_points} | Prediction: {prediction_result} | Actual: {actual_result} | "
            f"Confidence: {confidence:.2f}%"
        )

        if correct:
            print(f"Backtest Passed: {name} - Prediction was correct.")
            correct_predictions += 1
        else:
            print(f"Backtest Failed: {name} - Prediction was incorrect.")

        confidence_rounded = int(confidence // 10) * 10
        confidence_category = f"{confidence_rounded}-{confidence_rounded+10}"
        if confidence_category in confidence_buckets:
            confidence_buckets[confidence_category]["total"] += 1
            if correct:
                confidence_buckets[confidence_category]["correct"] += 1

    accuracy = (correct_predictions / len(predictions)) * 100

    print(f'Overrall Accuracy {accuracy}')
    print("Confidence Range Performance")
    for category, data in confidence_buckets.items():
        if data["total"] > 0:
            accuracy = (data["correct"] / data["total"]) * 100
            print(
                f"Confidence {category}% → Win Rate: {accuracy:.2f}% ({data['correct']}/{data['total']})")
        else:
            print(f"Confidence {category}% → No data available.")


if __name__ == "__main__":
    backtest()
