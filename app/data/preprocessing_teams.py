import os
import pandas as pd
import sqlite3 as sq


def get_team_game_logs(game_ids, wls):
    db_path = os.path.join(os.path.dirname(__file__),
                           '../..', 'nba_predict.sqlite')
    connection = sq.connect(db_path)

    results = []
    for game_id, wl in zip(game_ids, wls):
        wl = str(wl)
        query = f"""
        SELECT *
        FROM teams_data
        WHERE GAME_ID = ? and WL = ?
        """
        team_df = pd.read_sql_query(
            query, connection, params=(f'00{game_id}', wl))
        results.append(team_df)

    connection.close()

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
