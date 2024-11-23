import pandas as pd
import sqlite3 as sq
import os
from datetime import datetime, timedelta
from .scrape_utils import fill_players_data, fill_teams_data

db_path = os.path.join(os.path.dirname(__file__),
                       '../..', 'nba_predict.sqlite')


def fill_data_to_db():
    players_data = fill_players_data()

    if not players_data.empty:
        connection = sq.connect(db_path.format('players_data'))
        players_data.to_sql('players_data', connection,
                            if_exists='append', index=False)
        connection.close()
        print('Saved most recent players data')
