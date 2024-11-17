import pandas as pd
import sqlite3 as sq
import os
from datetime import datetime, timedelta

csv_file = os.path.join(os.path.dirname(__file__),
                        '../..', 'predictions_2024-11-17_trend_new.csv')


df = pd.read_csv(csv_file)
df['date'] = pd.to_datetime((datetime.now() - timedelta(days=1)).date())

total_games = len(df)
connection = sq.connect('nba_predict.sqlite'.format('predictions'))

df.to_sql('predictions', connection, if_exists='replace', index=False)
