import pandas as pd
from sklearn.model_selection import train_test_split

from data.feature_engineering import prepare_features_with_rolling_averages
from data.preprocessing import fetch_all_active_players
from utils.labels import rolling_average_labels

from .xgboost_model import train_xgboost_model, evaluate_model, save_best_model


def train_model_and_save_model():
    """
    Train the XGBoost model and save the best model.
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
