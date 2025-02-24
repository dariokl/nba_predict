import pandas as pd

from app.data_processing.feature_engineering import prepare_features_with_rolling_averages
from app.data_processing.player_preprocessing import get_all_active_players
from app.utils.labels import rolling_average_labels

from .xgboost_model import train_xgboost_model
from .gradient_model import train_gradient_boosting_model


def train_model_and_save_model_xgboost():
    """
    Train the XGBoost model and save it.
    """
    all_players = get_all_active_players()

    all_player_data = []
    all_labels = []

    for player in all_players:
        print(f"Preparing data for player {player['full_name']}...")

        player_data = prepare_features_with_rolling_averages(
            player_id=player['id']
        )

        if not player_data is None:
            player_features = pd.DataFrame(
                player_data[rolling_average_labels].values, columns=rolling_average_labels)

            all_player_data.append(player_features)

            player_labels = pd.Series(player_data['PTS'].values)
            all_labels.append(player_labels)
        else:
            print('No data found for player')

    x_all = pd.concat(all_player_data, axis=0, ignore_index=True)
    y_all = pd.concat(all_labels, axis=0, ignore_index=True)

    train_xgboost_model(x_all, y_all)


def train_model_and_save_model_gradient():
    """
    Train the GradientBoostingRegressor model and save it.
    """
    all_players = get_all_active_players()

    all_player_data = []
    all_labels = []

    for player in all_players:
        print(f"Preparing data for player {player['full_name']}...")

        player_data = prepare_features_with_rolling_averages(
            player_id=player['id']
        )

        if player_data is not None:
            player_features = pd.DataFrame(
                player_data[rolling_average_labels].values, columns=rolling_average_labels)

            all_player_data.append(player_features)

            player_labels = pd.Series(player_data['PTS'].values)
            all_labels.append(player_labels)
        else:
            print('No data found for player')

    # Combine all player data and labels
    x_all = pd.concat(all_player_data, axis=0, ignore_index=True)
    y_all = pd.concat(all_labels, axis=0, ignore_index=True)

    # Train and save GradientBoosting model
    train_gradient_boosting_model(x_all, y_all)
