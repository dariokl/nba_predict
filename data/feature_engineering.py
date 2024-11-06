import pandas as pd
from utils.labels import rolling_average_labels
from .preprocessing import fetch_player_game_logs
import time


def prepare_features_with_rolling_averages(player_id, rolling_window=5):
    # Fetch the player's game data
    games_df = fetch_player_game_logs(player_id)

    # Extract key columns: points, minutes, field goals made, etc.
    games_df = games_df[[
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "REB",
        "AST",
        "STL",
        "TOV",
        "PF",
        "PTS",
        "PLUS_MINUS",
    ]]

    # Convert 'MIN' (minutes played) to a numeric format
    games_df['MIN'] = pd.to_numeric(games_df['MIN'], errors='coerce')

    # Calculate rolling averages
    games_df['points_rolling_avg'] = games_df['PTS'].rolling(
        window=rolling_window).mean()
    games_df['minutes_rolling_avg'] = games_df['MIN'].rolling(
        window=rolling_window).mean()
    games_df['fg_pct_rolling_avg'] = games_df['FG_PCT'].rolling(
        window=rolling_window).mean()
    games_df['reb_rolling_avg'] = games_df['REB'].rolling(
        window=rolling_window).mean()
    games_df['ast_rolling_avg'] = games_df['AST'].rolling(
        window=rolling_window).mean()

    # Shift rolling averages down by one row so that the average is for previous games only
    games_df[rolling_average_labels] = games_df[rolling_average_labels].shift(
        1)

    # Drop the rows with NaN values in rolling averages (due to initial empty rolling window)
    games_df = games_df.dropna().reset_index(drop=True)

    return games_df
