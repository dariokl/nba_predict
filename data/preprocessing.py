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
    player_games_df = all_games_df[all_games_df['Player_ID'] == player_id]
    player_games_df = player_games_df.sort_values(by='GAME_DATE')

    return player_games_df


def fill_win_column():
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    csv_file = os.path.join(os.path.dirname(__file__),
                            '..', f'predictions_{yesterday}.csv')

    if not os.path.exists(csv_file):
        print(f"{csv_file} does not exist.")
        return

    predictions_df = pd.read_csv(csv_file)

    for index, row in predictions_df.iterrows():
        player_name = row['player_name']
        threshold = row['threshold']
        over_under = row['over_under']

        performance = get_player_recent_performance(player_name)

        actual_points = performance['PTS'].iloc[0] if isinstance(
            performance, pd.DataFrame) else performance['PTS']

        predictions_df.at[index, 'scored_points'] = actual_points

        if (over_under == 'Over' and actual_points > threshold) or (over_under == 'Under' and actual_points <= threshold):
            predictions_df.at[index, 'win'] = 'W'
        else:
            predictions_df.at[index, 'win'] = 'L'

    predictions_df.to_csv(csv_file, index=False)
    print(f"Win column and scored_points updated and saved to {csv_file}")
