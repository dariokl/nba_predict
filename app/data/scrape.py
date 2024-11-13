import csv
import pandas as pd
from time import sleep
import os

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


csv_file = os.path.join(os.path.dirname(__file__),
                        '..', 'player_performance.csv')


def scrape_seasons():
    seasons = ["2020-2021", "2021-2022", "2022-23", "2023-24", "2024-25"]

    all_players_data = pd.DataFrame()

    all_players = players.get_active_players()

    for player in all_players:
        player_id = player["id"]
        print(f"Fetching data for player ID: {player_id}")

        for season in seasons:
            print(f"  Season: {season}")
            try:
                game_log = playergamelog.PlayerGameLog(
                    player_id=player_id, season=season)
                game_data = game_log.get_data_frames()[0]

                all_players_data = pd.concat(
                    [all_players_data, game_data], ignore_index=True)

                sleep(1)
            except Exception as e:
                print(
                    f"Error fetching data for player ID {player_id} in season {season}: {e}")
                continue

    all_players_data.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")


def scrape_season(season):
    """
    Scrape the latest season data and update the CSV file.
    """
    all_players_data = pd.DataFrame()

    try:
        all_players_data = pd.read_csv(
            csv_file)
        print("Existing data loaded.")
    except FileNotFoundError:
        all_players_data = pd.DataFrame()
        print("No existing data found; creating a new file.")

    all_players = players.get_active_players()

    for player in all_players:
        player_id = player["id"]

        player_data = playergamelog.PlayerGameLog(player_id, season=season)
        game_data = player_data.get_data_frames()[0]

        all_players_data = pd.concat(
            [all_players_data, game_data], ignore_index=True)
        sleep(1)

    # Drop duplicates based on all columns (or select specific columns if needed)
    all_players_data.drop_duplicates(inplace=True)

    all_players_data.to_csv(csv_file, index=False)
