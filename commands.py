import pandas as pd
import json
import os
import argparse
from datetime import datetime
from fuzzywuzzy import process

from app.data.preprocessing import find_players_by_full_name, fetch_all_active_players, get_player_recent_performance
from app.utils.results_utils import fill_win_column, predictions_stats
from app.utils.scrape_utils import scrape_season, scrape_team_seasons, scrape_seasons
from app.model.train_helper import train_model_and_save_model
from app.model.predict import predict_for_player_mean, predict_for_player_trend
from app.utils.database_utils import predictions_to_db, fill_data_to_db


def find_player_by_name(partial_name):
    nba_players = fetch_all_active_players()
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
        threshold = float(player['points'])

        player_id = find_players_by_full_name(name)

        if player_id is None:
            print(f"Player {name} not found.")
            continue

        if (type == 'mean'):
            over_under, predicted_points, confidence = predict_for_player_mean(
                player_id, threshold=threshold)
        else:
            over_under, predicted_points, confidence = predict_for_player_trend(
                player_id, threshold=threshold)

        predictions.append({
            'player_name': name,
            'threshold': threshold,
            'over_under': 'Over' if over_under else 'Under',
            'confidence': round(confidence, 2),
            'predicted_points': round(predicted_points, 0),
            'odds': 1.90,
            'win': '',
        })

    predictions_df = pd.DataFrame(predictions)

    predictions_df.to_csv(filename, mode='a', index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict using the XGBoost model.")
    parser.add_argument(
        'action', choices=['train', 'predict-mean', 'predict-trend', 'scrape', 'fill-predictions', 'predictions-stats', 'predictions-to-db', 'fill-to-db']
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
        case 'predictions-to-db':
            predictions_to_db()
        case 'fill-to-db':
            fill_data_to_db()


if __name__ == "__main__":
    main()
