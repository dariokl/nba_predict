import os
import sqlite3 as sq
from datetime import datetime, timedelta
import pandas as pd
from app.data_processing.player_preprocessing import get_player_recent_performance

db_path = os.path.join(os.path.dirname(__file__), "../../nba_predict.sqlite")

today = datetime.today()
yesterday = today - timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")


def fill_win_column():
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return

    try:
        with sq.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT player_name, betline, over_under, type 
                FROM predictions 
                WHERE win IS NULL AND DATE(date) = ?
                """,
                (yesterday_str,),
            )
            rows = cursor.fetchall()

            if not rows:
                print("No predictions to process.")
                return

            updates = []

            for player_name, betline, over_under, type_ in rows:
                try:
                    performance = get_player_recent_performance(player_name)

                    if performance.empty:
                        continue

                    actual_points = (
                        performance["PTS"].iloc[0]
                        if isinstance(performance, pd.DataFrame)
                        else performance["PTS"]
                    )

                    win = 1 if (over_under == "Over" and actual_points > betline) or (
                        over_under == "Under" and actual_points <= betline
                    ) else 0

                    updates.append((float(actual_points), win,
                                   yesterday_str, player_name, type_))

                except Exception as e:
                    print(f"Error processing {player_name}: {e}")
                    continue

            if updates:
                cursor.executemany(
                    """
                    UPDATE predictions 
                    SET scored_points = ?, win = ? 
                    WHERE win IS NULL AND DATE(date) = ? AND player_name = ? AND type = ?
                    """,
                    updates,
                )
                conn.commit()
                print(f"{len(updates)} predictions updated.")

    except sq.Error as e:
        print(f"Database error: {e}")


def predictions_stats():
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return

    try:
        with sq.connect(db_path) as conn:
            cursor = conn.cursor()
            prediction_types = ["trend", "mean"]

            for prediction_type in prediction_types:
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) AS total_predictions, 
                        SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) AS total_wins
                    FROM predictions 
                    WHERE win IS NOT NULL AND type = ?
                    """,
                    (prediction_type,),
                )
                total_predictions, total_wins = cursor.fetchone()

                if total_predictions > 0:
                    winning_percentage = (total_wins / total_predictions) * 100
                    print(
                        f"Winning Percentage for {prediction_type}: {winning_percentage:.2f}%")
                else:
                    print(f"No valid predictions for type {prediction_type}.")

    except sq.Error as e:
        print(f"Database error: {e}")
