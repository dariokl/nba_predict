import pandas as pd
import sqlite3 as sq
import os
from datetime import datetime, timedelta
from .scrape_utils import fill_players_data, fill_teams_data, fill_shot_data

db_path = os.path.join(os.path.dirname(__file__),
                       '../..', 'nba_predict.sqlite')


def fill_data_to_db():
    players_data = fill_players_data()
    teams_data = fill_teams_data()
    shots_data = fill_shot_data()

    if not players_data.empty:
        connection = sq.connect(db_path.format('players_data'))
        players_data.to_sql('players_data', connection,
                            if_exists='append', index=False)
        connection.close()
        print('Saved most recent players data')

    if not teams_data.empty:
        connection = sq.connect(
            db_path.format('teams_data'))
        teams_data.to_sql('teams_data', connection,
                          if_exists='append', index=False)
        connection.close()
        print('Saved most recent teams data')

    if not shots_data.empty:
        connection = sq.connect(
            db_path.format('shot_data'))
        shots_data.to_sql('shot_data', connection,
                          if_exists='append', index=False)
        connection.close()
        print('Saved most recent shots data')
