import os
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


def fetch_all_active_players():
    return players._get_active_players()


def find_players_by_full_name(name):
    player = players.find_players_by_full_name(name)

    return player[0]['id']


def fetch_player_game_logs(player_id):
    csv_file = os.path.join(os.path.dirname(__file__),
                            '..', 'nba_player_performance_last_3_seasons.csv')

    all_games_df = pd.read_csv(csv_file)

    player_games_df = all_games_df[all_games_df['Player_ID'] == player_id]

    player_games_df = player_games_df.sort_values(by='GAME_DATE')

    return player_games_df
