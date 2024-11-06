import pandas as pd
import json
import os
import argparse

from sklearn.model_selection import train_test_split
import xgboost as xgb

from data.feature_engineering import prepare_features_with_rolling_averages
from data.preprocessing import fetch_all_active_players, find_players_by_full_name
from model.xgboost_model import train_xgboost_model, evaluate_model, save_best_model
from utils.labels import rolling_average_labels
from scrape import scrape_seasons


def train_model():
    """
    Train the XGBoost model with hyperparameter tuning and save the best model.
    """
    all_players = fetch_all_active_players()

    all_player_data = []
    all_labels = []

    for player in all_players:
        print(f"Preparing data for player {player['full_name']}...")

        # Prepare features with rolling averages for each player
        player_data = prepare_features_with_rolling_averages(
            player_id=player['id'], rolling_window=5
        )

        # Convert the numpy array of features to a DataFrame
        player_features = pd.DataFrame(
            player_data[rolling_average_labels].values, columns=rolling_average_labels)

        # Append the DataFrame to the all_player_data list
        all_player_data.append(player_features)

        # Append the labels (target variable) to the all_labels list
        player_labels = pd.Series(player_data['PTS'].values)
        all_labels.append(player_labels)

    all_labels = [
        label for label in all_labels if label is not None and not label.empty]

    X_all = pd.concat(all_player_data, axis=0, ignore_index=True)
    y_all = pd.concat(all_labels, axis=0, ignore_index=True)

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42)

    # Train the XGBoost model with hyperparameter tuning
    best_model = train_xgboost_model(X_train, y_train)

    # Evaluate the model on the test set
    mae = evaluate_model(best_model, X_test, y_test)
    print(f"Test MAE: {mae}")

    # Save the best model if it's the best so far
    save_best_model(best_model, mae)


def predict_for_player(player_id, threshold):
    """
    Predict whether a player will score above a certain threshold.
    """

    best_model = xgb.XGBRegressor()
    best_model.load_model('best_model.json')

    games_df = prepare_features_with_rolling_averages(
        player_id=player_id, rolling_window=5
    )

    X_player = games_df[rolling_average_labels]

    predicted_points = best_model.predict(X_player)

    will_score_above = predicted_points[-1] > threshold

    percentage_deviation = (
        (predicted_points[-1] - threshold) / threshold) * 100

    print(
        f"Will player {player_id} score above {threshold} points? {will_score_above}")
    print(f"Predicted points: {predicted_points[-1]}")

    return will_score_above, predicted_points[-1], percentage_deviation


def main():
    # Setup argument parsing
    parser = argparse.ArgumentParser(
        description="Train or predict using the XGBoost model.")
    parser.add_argument(
        'action', choices=['train', 'predict', 'scrape'], help="Specify whether to 'train' or 'predict'."
    )
    args = parser.parse_args()

    if args.action == 'train':
        print("Training the model...")
        train_model()
        print("Model trained and saved.")
    elif args.action == 'scrape':
        scrape_seasons()
    elif args.action == 'predict':
        # Perform predictions
        print("Performing predictions...")

        with open('data.json') as f:
            predictions = []
            player_data = json.load(f)

        for player in player_data:
            print(f"Processing player: {player['name']}")
            name = player['name']
            stats = player['stats']

            player_id = find_players_by_full_name(name)

            if player_id is None:
                print(f"Player {name} not found.")
                continue

            for stat in stats:
                threshold = stat['points']

                will_score_above, predicted_points, percentage_deviation = predict_for_player(
                    player_id, threshold=threshold)

                predictions.append({
                    'player_name': name,
                    'threshold': threshold,
                    'will_score_above': will_score_above,
                    'predicted_points': predicted_points,
                    'percentage_deviation': percentage_deviation
                })

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('predictions.csv', mode='a', header=not os.path.exists(
            'predictions.csv'), index=False)
        print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()
