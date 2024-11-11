import pandas as pd

from data.feature_engineering import prepare_features_with_rolling_averages
from data.preprocessing import fetch_all_active_players
from utils.labels import rolling_average_labels

from .xgboost_model import train_xgboost_model


def train_model_and_save_model():
    """
    Train the XGBoost model and save the best model.
    """
    all_players = fetch_all_active_players()

    all_player_data = []
    all_labels = []

    for player in all_players:
        print(f"Preparing data for player {player['full_name']}...")

        player_data = prepare_features_with_rolling_averages(
            player_id=player['id']
        )

        player_features = pd.DataFrame(
            player_data[rolling_average_labels].values, columns=rolling_average_labels)

        all_player_data.append(player_features)

        # Append the labels (target variable) to the all_labels list
        player_labels = pd.Series(player_data['PTS'].values)
        all_labels.append(player_labels)

    x_all = pd.concat(all_player_data, axis=0, ignore_index=True)
    y_all = pd.concat(all_labels, axis=0, ignore_index=True)

    train_xgboost_model(x_all, y_all)
