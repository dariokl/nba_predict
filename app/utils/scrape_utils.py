import csv
import pandas as pd
from time import sleep
import os
import sqlite3 as sq
from datetime import datetime, timedelta

from nba_api.stats.endpoints import playergamelog, teamgamelogs
from nba_api.stats.static import players, teams


csv_file = os.path.join(os.path.dirname(__file__),
                        '../..', 'player_performance.csv')


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

    all_players_data.drop_duplicates(inplace=True)

    all_players_data.to_csv(csv_file, index=False)


def scrape_team_seasons():
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

    all_teams_data = pd.DataFrame()

    all_teams = teams.get_teams()

    for team in all_teams:
        team_id = team["id"]
        team_name = team["full_name"]
        print(f"Fetching data for team: {team_name} (ID: {team_id})")

        for season in seasons:
            print(f"  Season: {season}")
            try:
                game_log = teamgamelogs.TeamGameLogs(
                    team_id_nullable=team_id, season_nullable=season)
                game_data = game_log.get_data_frames()[0]

                all_teams_data = pd.concat(
                    [all_teams_data, game_data], ignore_index=True)

                sleep(1)
            except Exception as e:
                print(
                    f"Error fetching data for team {team_name} (ID: {team_id}) in season {season}: {e}")
                continue

    connection = sq.connect('nba_predict.sqlite'.format('teams_data'))

    all_teams_data.to_sql('teams_data', connection,
                          if_exists='replace', index=False)

    print(f"Data saved to database")


def fill_players_data():
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime("%m/%d/%Y")

    all_players_data = pd.DataFrame()
    all_players = players.get_active_players()
    print(yesterday)
    for player in all_players:
        player_id = player["id"]
        print(f"Fetching data for player ID: {player_id}")

        try:
            game_log = playergamelog.PlayerGameLog(
                player_id=player_id, date_from_nullable=yesterday, season='2024-25')
            game_data = game_log.get_data_frames()[0]

            all_players_data = pd.concat(
                [all_players_data, game_data], ignore_index=True)

            sleep(1)
        except Exception as e:
            print(e)
            print(
                f"Error fetching data for player ID {player_id} : {e}")
            continue

    return all_players_data


def fill_teams_data():
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime("%m/%d/%Y")

    all_teams_data = pd.DataFrame()

    all_teams = teams.get_teams()
    for team in all_teams:
        team_id = team["id"]
        team_name = team["full_name"]
        print(f"Fetching data for team: {team_name} (ID: {team_id})")

        try:
            game_log = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id, date_from_nullable=yesterday, season_nullable='2024-25')
            game_data = game_log.get_data_frames()[0]

            print(game_data)
            all_teams_data = pd.concat(
                [all_teams_data, game_data], ignore_index=True)

            sleep(1)
        except Exception as e:
            print(
                f"Error fetching data for team {team_name} (ID: {team_id}): {e}")
            continue

    return all_teams_data
