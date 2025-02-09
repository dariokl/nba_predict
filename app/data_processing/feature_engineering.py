import pandas as pd
import numpy as np
from time import sleep

from .player_preprocessing import get_player_game_logs
from .team_preprocessing import get_team_game_logs


def prepare_features_with_rolling_averages(player_id, rolling_window=5):
    """
    Prepare features for a player using rolling averages and advanced statistics.
    """
    games_df = get_player_game_logs(player_id)
    if games_df.empty:
        return None

    opponent_df = get_opponent_stats(games_df)
    games_df = preprocess_games_data(games_df)
    games_df = calculate_rolling_averages(games_df, rolling_window)
    games_df = calculate_advanced_metrics(games_df)
    games_df = add_opponent_metrics(games_df, opponent_df)
    games_df = clean_data(games_df)

    return games_df


def get_opponent_stats(games_df):
    """
    Get and process opponent statistics based on game IDs and results.
    """
    return get_team_game_logs(
        games_df['Game_ID'], games_df['WL'].apply(
            lambda x: 'L' if x == 'W' else 'W').to_list()
    )


def preprocess_games_data(games_df):
    """
    Preprocess game data, including date parsing, feature encoding, and lag calculation.
    """
    games_df['GAME_DATE'] = pd.to_datetime(
        games_df['GAME_DATE'], format='%b %d, %Y')
    today = pd.Timestamp('today').normalize()
    games_df['DAYS_SINCE_LAST_GAME'] = (today - games_df['GAME_DATE']).dt.days
    games_df['HOME'] = games_df['MATCHUP'].apply(
        lambda x: 1 if 'vs.' in x else 0)
    games_df['WL'] = games_df['WL'].apply(lambda x: 1 if x == 'W' else 0)

    return games_df.drop(columns=['MATCHUP', 'VIDEO_AVAILABLE'])


def calculate_rolling_averages(games_df, rolling_window):
    """
    Calculate rolling averages and standard deviations for key features.
    """
    rolling_features = ['PTS', 'MIN', 'FG_PCT', 'FGM', 'FGA', 'FG3M', 'FG3A',
                        'FG3_PCT', 'FTM', 'FT_PCT', 'REB', 'AST']

    for feature in rolling_features:
        games_df[f'{feature}_ROLL_AVG'] = games_df[feature].rolling(
            window=rolling_window).mean()
        if feature == 'PTS':  # Example: adding standard deviation for PTS
            games_df[f'{feature}_STD_AVG'] = games_df[feature].rolling(
                window=rolling_window).std()

    return games_df


def calculate_advanced_metrics(games_df):
    """
    Add advanced metrics, including true shooting percentage and player efficiency rating.
    """
    games_df['TRUE_SHOOTING_PCT'] = games_df['PTS'] / \
        (2 * (games_df['FGA'] + (0.44 * games_df['FTA'])))
    games_df['BACK_TO_BACK'] = games_df['DAYS_SINCE_LAST_GAME'].apply(
        lambda x: 1 if x <= 1 else 0)

    # Player Efficiency Rating (PER)
    games_df['PER'] = (
        games_df['PTS'] + (0.4 * games_df['FGM']) - (0.7 * games_df['FGA']) -
        (0.4 * (games_df['FTA'] - games_df['FTM'])) + (0.7 * games_df['REB']) +
        (0.3 * games_df['AST']) + (0.1 * games_df['STL']) + (0.1 * games_df['BLK']) -
        (0.1 * games_df['TOV']) - (0.2 * games_df['PF'])
    ) / games_df['MIN']
    games_df['PER'] = games_df['PER'].replace([np.inf, -np.inf], np.nan)
    games_df['ROLLING_PER'] = games_df['PER'].rolling(window=5).mean()

    games_df['FGM_FGA_RATIO'] = games_df['FGM'] / (games_df['FGA'] + 1e-6)
    games_df['3PM_3PA_RATIO'] = games_df['FG3M'] / (games_df['FG3A'] + 1e-6)
    games_df['REST_IMPACT'] = games_df['DAYS_SINCE_LAST_GAME'] * \
        games_df['PTS_ROLL_AVG']

    return games_df


def add_opponent_metrics(games_df, opponent_df):
    """
    Add opponent-related metrics to the game dataset.
    """
    opponent_metrics = ['W_PCT_RANK', 'STL_RANK', 'PF_RANK']

    for metric in opponent_metrics:
        games_df[f'OPP_{metric}'] = opponent_df[metric]

    games_df['TOTAL_POINTS'] = opponent_df['TOTAL_POINTS']

    return games_df


def clean_data(games_df):
    """
    Scale numerical features and clean up the dataset by removing NaNs.
    """
    games_df = games_df.apply(pd.to_numeric, errors='coerce')
    games_df = games_df.dropna().reset_index(drop=True)

    # Sort by days since the last game
    return games_df.sort_values(by='DAYS_SINCE_LAST_GAME', ascending=False)
