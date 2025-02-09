import os
import pandas as pd
import sqlite3 as sq
from time import sleep
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


def get_all_active_players():
    return players._get_active_players()


def get_player_recent_performance(name):
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime("%m/%d/%Y")

    player = find_players_by_full_name(name)
    game_log = playergamelog.PlayerGameLog(
        player_id=player, date_from_nullable=yesterday)
    game_data = game_log.get_data_frames()[0]
    sleep(1)
    return game_data


def find_players_by_full_name(name):
    player = players.find_players_by_full_name(name)
    return player[0]['id']


def get_player_game_logs(player_id):
    db_path = os.path.join(os.path.dirname(__file__),
                           '../..', 'nba_predict.sqlite')

    connection = sq.connect(db_path)

    query = "SELECT * FROM players_data WHERE Player_ID = ?"
    player_games_df = pd.read_sql_query(query, connection, params=(player_id,))

    connection.close()

    player_games_df['GAME_DATE'] = pd.to_datetime(
        player_games_df['GAME_DATE'], format='%b %d, %Y')
    player_games_df = player_games_df.sort_values(
        by='GAME_DATE', ascending=True)

    return player_games_df
