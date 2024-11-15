import pandas as pd
import sqlite3 as sq
import os

csv_file = os.path.join(os.path.dirname(__file__),
                        '../..', 'player_performance.csv')


df = pd.read_csv(csv_file)

total_games = len(df)

connection = sq.connect('nba_predict.sqlite'.format('players_data'))

df.to_sql('players_data', connection, if_exists='replace', index=False)
