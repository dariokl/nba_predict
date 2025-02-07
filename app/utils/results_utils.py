from datetime import datetime
from datetime import timedelta
import os
import pandas as pd
import sqlite3 as sq

from app.data.preprocessing_players import get_player_recent_performance

db_path = os.path.join(os.path.dirname(__file__),
                       '../../nba_predict.sqlite')


def fill_win_column():
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')

    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return

    connection = sq.connect(db_path)
    cursor = connection.cursor()

    query = f"""
        SELECT player_name, betline, over_under, type
        FROM predictions
        WHERE win is NULL and DATE(date) = ?
        """
    rows = cursor.execute(query, (yesterday,)).fetchall()

    if not rows:
        print(f"No predictions to process in table")

    for row in rows:
        player_name, betline, over_under, type_ = row
        try:
            performance = get_player_recent_performance(player_name)
        except:
            continue

        if performance.empty:
            continue

        actual_points = performance['PTS'].iloc[0] if isinstance(
            performance, pd.DataFrame) else performance['PTS']

        if (over_under == 'Over' and actual_points > betline) or (over_under == 'Under' and actual_points <= betline):
            win = 1
        else:
            win = 0

        update_query = f"""
        UPDATE predictions
        SET scored_points = ?, win = ?
        WHERE win IS NULL and DATE(date) = ? and player_name = ? and type = ?
        """
        cursor.execute(update_query, (float(actual_points),
                       win, yesterday, player_name, type_))

    connection.commit()
    connection.close()
    print('Predictions are updated')


def predictions_stats():
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return

    today = datetime.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')

    connection = sq.connect(db_path)
    cursor = connection.cursor()

    # List of predictions to process
    types = ['trend', 'mean']

    for prediciton_type in types:
        # Count wins (1) and total valid predictions (excluding 'DNP')
        query = """
        SELECT 
            COUNT(*) AS total_predictions, 
            SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) AS total_wins
        FROM (
            SELECT win 
            FROM predictions
            WHERE win IS NOT NULL AND type = ?
            LIMIT 500
        ) subquery;
        """
        result = cursor.execute(query, (prediciton_type,)).fetchone()

        print(result)
        total_predictions = result[0]  # Total valid predictions
        total_wins = result[1]

        if total_predictions > 0:
            winning_percentage = (total_wins / total_predictions) * 100
            print(f"Winning Percentage for {
                  prediciton_type}: {winning_percentage:.2f}%")
        else:
            print(f"No valid predictions for type {prediciton_type}.")

    connection.close()
