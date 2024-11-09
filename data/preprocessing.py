import os
import pandas as pd
from datetime import datetime, timedelta

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


def fetch_all_active_players():
    return players._get_active_players()


def get_player_recent_performance(name, games=1):
    player = find_players_by_full_name(name)
    game_log = playergamelog.PlayerGameLog(
        player_id=player, season='2024-25')
    game_data = game_log.get_data_frames()[0]
    return game_data


def find_players_by_full_name(name):
    player = players.find_players_by_full_name(name)
    return player[0]['id']


def fetch_player_game_logs(player_id):
    csv_file = os.path.join(os.path.dirname(__file__),
                            '..', 'player_performance.csv')

    all_games_df = pd.read_csv(csv_file)

    all_games_df['GAME_DATE'] = pd.to_datetime(
        all_games_df['GAME_DATE'], format='%b %d, %Y')

    latest_game_date = all_games_df['GAME_DATE'].max()
    all_games_df['days_since_last_game'] = (
        latest_game_date - all_games_df['GAME_DATE']).dt.days

    all_games_df = all_games_df.drop(columns=['GAME_DATE'])
    all_games_df['WL'] = all_games_df['WL'].map({'W': 1, 'L': 0})
    player_games_df = all_games_df[all_games_df['Player_ID'] == player_id]
    player_games_df = player_games_df.sort_values(
        by='days_since_last_game', ascending=False)

    return player_games_df
