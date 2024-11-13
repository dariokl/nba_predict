import pandas as pd
import json
import os
import argparse
from datetime import datetime
from fuzzywuzzy import process

from app.data.preprocessing import find_players_by_full_name, get_player_recent_performance, fetch_all_active_players
from app.data.results import fill_win_column, predictions_stats
from app.data.scrape import scrape_season, scrape_seasons
from app.model.train import train_model_and_save_model
from app.model.predict import predict_for_player_mean, predict_for_player_trend


def find_player_by_name(partial_name):
    nba_players = fetch_all_active_players()
    full_names = [player['full_name'] for player in nba_players]
    partial_name = partial_name.replace(".", " ")
    # Use fuzzy matching to find the closest NBA player name
    match, score = process.extractOne(partial_name, full_names)
    # Return the match if the score is above a reasonable threshold (e.g., 80)
    if score > 80:
        return match
    else:
        return None


def predict_from_json(type):
    with open('data.json') as f:
        predictions = []
        player_data = json.load(f)
        today_date = datetime.today().strftime('%Y-%m-%d')
        filename = f'predictions_{today_date}_{type}_2.csv'

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
    file_exists = os.path.exists(filename)

    print(file_exists)
    print(predictions_df)

    predictions_df.to_csv(filename, mode='a', index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict using the XGBoost model.")
    parser.add_argument(
        'action', choices=['train', 'predict-mean', 'predict-trend', 'scrape', 'fill-predictions', 'predictions-stats']
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
            scrape_season('2024-25')
        case 'fill-predictions':
            fill_win_column()
        case 'predictions-stats':
            predictions_stats()


if __name__ == "__main__":
    main()
