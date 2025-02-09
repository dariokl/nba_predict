import os
import pandas as pd
import sqlite3 as sq


def get_team_game_logs(game_ids, wls):
    db_path = os.path.join(os.path.dirname(__file__),
                           '../..', 'nba_predict.sqlite')
    connection = sq.connect(db_path)

    results = []
    for game_id in game_ids:
        query = f"""
        SELECT *
        FROM teams_data
        WHERE GAME_ID = ?
        """
        game_df = pd.read_sql_query(
            query, connection, params=(f'00{game_id}',))
        results.append(game_df)

    if not results:
        connection.close()
        return pd.DataFrame()

    all_games_df = pd.concat(results, ignore_index=True)

    total_points = (
        all_games_df.groupby("GAME_ID")["PTS"].sum().reset_index()
        .rename(columns={"PTS": "TOTAL_POINTS"})
    )

    all_games_df = pd.merge(all_games_df, total_points,
                            on="GAME_ID", how="left")

    filtered_results = []
    for game_id, wl in zip(game_ids, wls):
        wl = str(wl)
        filtered_df = all_games_df[
            (all_games_df["GAME_ID"] == f"00{game_id}") & (
                all_games_df["WL"] == wl)
        ]
        filtered_results.append(filtered_df)

    connection.close()

    return pd.concat(filtered_results, ignore_index=True) if filtered_results else pd.DataFrame()
