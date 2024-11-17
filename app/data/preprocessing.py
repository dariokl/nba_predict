import os
import pandas as pd
import sqlite3 as sq
from time import sleep

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


def fetch_all_active_players():
    return players._get_active_players()


def get_player_recent_performance(name, games=1):
    player = find_players_by_full_name(name)
    game_log = playergamelog.PlayerGameLog(
        player_id=player, season='2024-25')
    game_data = game_log.get_data_frames()[0]
    sleep(1)
    return game_data


def find_players_by_full_name(name):
    player = players.find_players_by_full_name(name)
    return player[0]['id']


def get_team_game_logs(game_ids, wls):
    db_path = os.path.join(os.path.dirname(__file__),
                           '../..', 'nba_predict.sqlite')
    connection = sq.connect(db_path)

    # Aggregate results
    results = []
    for game_id, wl in zip(game_ids, wls):

        wl = str(wl)
        query = f"""
        SELECT * 
        FROM teams_data 
        WHERE GAME_ID = ? and WL = ?
        """
        team_df = pd.read_sql_query(
            query, connection, params=(f'00{game_id}', wl))
        results.append(team_df)

    connection.close()

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def get_player_game_logs(player_id):
    db_path = os.path.join(os.path.dirname(__file__),
                           '../..', 'nba_predict.sqlite')

    connection = sq.connect(db_path)

    query = "SELECT * FROM players_data WHERE Player_ID = ?"
    player_games_df = pd.read_sql_query(query, connection, params=(player_id,))

    # Close the database connection
    connection.close()

    player_games_df['GAME_DATE'] = pd.to_datetime(player_games_df['GAME_DATE'])
    player_games_df = player_games_df.sort_values(
        by='GAME_DATE', ascending=True)

    return player_games_df
