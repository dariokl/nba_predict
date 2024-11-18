import pandas as pd
import sqlite3 as sq
import os
from datetime import datetime, timedelta
from .scrape_utils import fill_players_data, fill_teams_data


def fill_data_to_db():
    teams_data = fill_teams_data()
    players_data = fill_players_data()

    if not players_data.empty:
        connection = sq.connect('nba_predict.sqlite'.format('players_data'))
        players_data.to_sql('players_data', connection,
                            if_exists='append', index=False)
        print('Saved most recent players data')

    if not teams_data.empty:
        connection = sq.connect(
            'nba_predict.sqlite'.format('teams_data'))
        teams_data.to_sql('teams_data', connection,
                          if_exists='append', index=False)
        print('Saved most recent teams data')


def predictions_to_db():
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    csv_file = os.path.join(os.path.dirname(__file__),
                            '../..', f'predictions_{yesterday}_mean.csv')

    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime((datetime.now() - timedelta(days=1)).date())

    connection = sq.connect('nba_predict.sqlite'.format('predictions'))

    df.to_sql('predictions', connection, if_exists='append', index=False)
