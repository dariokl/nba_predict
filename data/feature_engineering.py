import pandas as pd
from .preprocessing import fetch_player_game_logs
from utils.labels import rolling_average_labels


def prepare_features_with_rolling_averages(player_id, rolling_window=5):

    games_df = fetch_player_game_logs(player_id)

    games_df['GAME_DATE'] = pd.to_datetime(
        games_df['GAME_DATE'], format='%b %d, %Y')

    latest_game_date = games_df['GAME_DATE'].max()
    games_df['DAYS_SINCE_LAST_GAME'] = (
        latest_game_date - games_df['GAME_DATE']).dt.days

    games_df = games_df.drop(
        columns=['GAME_DATE', 'MATCHUP', 'VIDEO_AVAILABLE'])

    games_df['WL'] = games_df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    games_df = games_df.apply(pd.to_numeric, errors='coerce')

    games_df['PTS_ROLL_AVG'] = games_df['PTS'].rolling(
        window=rolling_window).mean()
    games_df['PTS_STD_AVG'] = games_df['PTS'].rolling(
        window=rolling_window).std()
    games_df['MIN_ROLL_AVG'] = games_df['MIN'].rolling(
        window=rolling_window).mean()
    games_df['FG_PCT_ROLL_AVG'] = games_df['FG_PCT'].rolling(
        window=rolling_window).mean()
    games_df['FGM_ROLL_AVG'] = games_df['FGM'].rolling(
        window=rolling_window).mean()
    games_df['FGA_ROLL_AVG'] = games_df['FGA'].rolling(
        window=rolling_window).mean()
    games_df['FG3M_ROLL_AVG'] = games_df['FG3M'].rolling(
        window=rolling_window).mean()
    games_df['FG3GA_ROLL_AVG'] = games_df['FG3A'].rolling(
        window=rolling_window).mean()
    games_df['FG3_PCT_ROLL_AVG'] = games_df['FG3_PCT'].rolling(
        window=rolling_window).mean()
    games_df['FTM_PCT_ROLL_AVG'] = games_df['FTM'].rolling(
        window=rolling_window).mean()
    games_df['FTA_PCT_ROLL_AVG'] = games_df['FTA'].rolling(
        window=rolling_window).mean()
    games_df['FT_PCT_ROLL_AVG'] = games_df['FT_PCT'].rolling(
        window=rolling_window).mean()
    games_df['REB_ROLL_AVG'] = games_df['REB'].rolling(
        window=rolling_window).mean()
    games_df['AST_ROLL_AVG'] = games_df['AST'].rolling(
        window=rolling_window).mean()

    games_df['PTS_LAG_1'] = games_df['PTS'].shift(1)
    games_df['PTS_LAG_2'] = games_df['PTS'].shift(2)

    games_df[rolling_average_labels] = games_df[rolling_average_labels].shift(
        1)

    games_df = games_df.sort_values(
        by='DAYS_SINCE_LAST_GAME', ascending=False)

    games_df = games_df.dropna().reset_index(drop=True)

    return games_df
