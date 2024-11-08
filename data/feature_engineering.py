import pandas as pd
from .preprocessing import fetch_player_game_logs
from utils.labels import rolling_average_labels


def prepare_features_with_rolling_averages(player_id, rolling_window=7):
    games_df = fetch_player_game_logs(player_id)

    games_df = games_df[[
        "GAME_DATE", "WL", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "REB", "AST", "STL", "TOV", "PTS", "PLUS_MINUS"
    ]]

    games_df['MIN'] = pd.to_numeric(games_df['MIN'], errors='coerce')
    games_df['GAME_DATE'] = pd.to_datetime(
        games_df['GAME_DATE'], format='%b %d, %Y')

    # Add rolling averages and standard deviations
    games_df['points_rolling_avg'] = games_df['PTS'].rolling(
        window=rolling_window).mean()
    games_df['points_std_rolling'] = games_df['PTS'].rolling(
        window=rolling_window).std()
    games_df['minutes_rolling_avg'] = games_df['MIN'].rolling(
        window=rolling_window).mean()
    games_df['reb_rolling_avg'] = games_df['REB'].rolling(
        window=rolling_window).mean()
    games_df['ast_rolling_avg'] = games_df['AST'].rolling(
        window=rolling_window).mean()
    games_df['fgm_rolling_avg'] = games_df['FGM'].rolling(
        window=rolling_window).mean()
    games_df['fga_rolling_avg'] = games_df['FGA'].rolling(
        window=rolling_window).mean()
    games_df['fg_pct_rolling_avg'] = games_df['FG_PCT'].rolling(
        window=rolling_window).mean()
    games_df['fg3m_rolling_avg'] = games_df['FG3M'].rolling(
        window=rolling_window).mean()
    games_df['fg3a_rolling_avg'] = games_df['FG3A'].rolling(
        window=rolling_window).mean()
    games_df['fg3_pct_rolling_avg'] = games_df['FG3_PCT'].rolling(
        window=rolling_window).mean()
    games_df['ftm_pct_rolling_avg'] = games_df['FTM'].rolling(
        window=rolling_window).mean()
    games_df['fta_pct_rolling_avg'] = games_df['FTA'].rolling(
        window=rolling_window).mean()
    games_df['ft_pct_rolling_avg'] = games_df['FT_PCT'].rolling(
        window=rolling_window).mean()

    # Add immediate past game features (lags)
    games_df['pts_lag_1'] = games_df['PTS'].shift(1)
    games_df['pts_lag_2'] = games_df['PTS'].shift(2)

    # Calculate days since last game
    games_df['days_since_last_game'] = games_df['GAME_DATE'].diff().dt.days

    games_df[rolling_average_labels] = games_df[rolling_average_labels].shift(
        1)

    # Drop rows with NaN values in rolling averages
    games_df = games_df.dropna().reset_index(drop=True)

    return games_df
