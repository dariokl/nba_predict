import pandas as pd
import json
import os
import argparse
from datetime import datetime

from data.preprocessing import find_players_by_full_name, get_player_recent_performance, fill_win_column
from model.train import train_model_and_save_model
from model.predict import predict_for_player


def predict_from_json():
    with open('data.json') as f:
        predictions = []
        player_data = json.load(f)
        today_date = datetime.today().strftime('%Y-%m-%d')
        filename = f'predictions_{today_date}.csv'

    for player in player_data:
        print(f"Processing player: {player['name']}")
        name = player['name']
        threshold = player['points']

        player_id = find_players_by_full_name(name)

        if player_id is None:
            print(f"Player {name} not found.")
            continue

        over_under, predicted_points, confidence = predict_for_player(
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
        'action', choices=['train', 'predict', 'scrape', 'fill-predictions']
    )
    args = parser.parse_args()

    match args.action:
        case 'train':
            train_model_and_save_model()
        case 'predict':
            predict_from_json()
        case 'scrape':
            get_player_recent_performance('LaMelo Ball')
        case 'fill-predictions':
            fill_win_column()


if __name__ == "__main__":
    main()
