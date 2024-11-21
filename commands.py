import pandas as pd
import json
import os
import argparse
from datetime import datetime, timedelta
from fuzzywuzzy import process
import sqlite3 as sq
import uuid

from app.data.preprocessing_players import find_players_by_full_name, get_all_active_players, get_player_recent_performance
from app.utils.results_utils import fill_win_column, predictions_stats
from app.utils.scrape_utils import scrape_season, scrape_team_seasons, scrape_seasons
from app.model.train_helper import train_model_and_save_model
from app.model.regression.predict_regression_model import predict_for_player_mean, predict_for_player_trend
from app.utils.database_utils import fill_data_to_db

db_path = os.path.join(os.path.dirname(__file__),
                       './', 'nba_predict.sqlite')


def find_player_by_name(partial_name):
    nba_players = get_all_active_players()
    full_names = [player['full_name'] for player in nba_players]
    partial_name = partial_name.replace(".", " ")

    match, score = process.extractOne(partial_name, full_names)

    if score > 80:
        return match
    else:
        return None


def predict_from_json(type):
    with open('data.json') as f:
        predictions = []
        player_data = json.load(f)
        today_date = datetime.today().strftime('%Y-%m-%d')
        filename = f'predictions_{today_date}_{type}.csv'

    for player in player_data:
        print(f"Processing player: {player['name']}")
        name = find_player_by_name(player['name'])
        betline = float(player['points'])

        player_id = find_players_by_full_name(name)

        if player_id is None:
            print(f"Player {name} not found.")
            continue

        if (type == 'mean'):
            over_under, predicted_points, confidence = predict_for_player_mean(
                player_id, betline=betline)
        else:
            over_under, predicted_points, confidence = predict_for_player_trend(
                player_id, betline=betline)

        predictions.append({
            'player_name': name,
            'betline': betline,
            'over_under': 'Over' if over_under else 'Under',
            'confidence': round(confidence, 2),
            'predicted_points': round(predicted_points, 0),
            'odds': 1.90,
            'win': None,
            'scored_points': None,
            'type': type
        })

    predictions_df = pd.DataFrame(predictions)

    predictions_df['date'] = pd.to_datetime(
        (datetime.now() - timedelta(days=0)).date(), format='%b %d, %Y')

    print(predictions_df)

    connection = sq.connect(db_path.format('predictions_test'))
    predictions_df.to_sql('predictions_test', connection,
                          if_exists='append', index=False)

    print('Saved predictions')


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict using the XGBoost model.")
    parser.add_argument(
        'action', choices=['train', 'predict-mean', 'predict-trend', 'scrape', 'fill-predictions', 'predictions-stats', 'fill-to-db']
    )
    args = parser.parse_args()

    match args.action:
        case 'train':
            train_model_and_save_model()
        case 'predict-mean':
            predict_from_json('mean')
        case 'predict-trend':
            predict_from_json('trend')
        case 'scrape':
            scrape_team_seasons()
        case 'fill-predictions':
            fill_win_column()
            predictions_stats()
        case 'predictions-stats':
            predictions_stats()
        case 'fill-to-db':
            fill_data_to_db()


if __name__ == "__main__":
    main()
