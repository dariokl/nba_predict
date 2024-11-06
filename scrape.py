import csv
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from time import sleep
import pandas as pd


def scrape_seasons():
    seasons = ["2022-23", "2023-24", "2024-25"]

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

                # NBA API has a rate limit, so it's best to add a small delay
                sleep(1)  # Sleep 1 second to avoid rate limits
            except Exception as e:
                print(
                    f"Error fetching data for player ID {player_id} in season {season}: {e}")
                continue

    output_file = "nba_player_performance_last_3_seasons.csv"
    all_players_data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
