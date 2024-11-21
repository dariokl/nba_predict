import pandas as pd
import numpy as np
from time import sleep

from .preprocessing_players import get_player_game_logs
from .preprocessing_teams import get_team_game_logs
from app.utils.labels import rolling_average_labels


def prepare_features_with_rolling_averages(player_id, rolling_window=5):

    games_df = get_player_game_logs(player_id)

    if games_df.empty:
        return None

    opponent_df = get_team_game_logs(
        games_df['Game_ID'], games_df['WL'].apply(lambda x: 'L' if x == 'W' else 'W').to_list())

    games_df['GAME_DATE'] = pd.to_datetime(
        games_df['GAME_DATE'], format='%b %d, %Y')

    today = pd.Timestamp('today').normalize()  # Normalize to remove time part

    games_df['DAYS_SINCE_LAST_GAME'] = (today - games_df['GAME_DATE']).dt.days

    games_df = games_df.drop(
        columns=['GAME_DATE', 'MATCHUP', 'VIDEO_AVAILABLE'])

    games_df['WL'] = games_df['WL'].apply(lambda x: 1 if x == 'W' else 0)

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

    games_df['TRUE_SHOOTING_PCT'] = games_df['PTS'] / \
        (2 * (games_df['FGA'] + (0.44 * games_df['FTA']))
         )

    games_df['BACK_TO_BACK'] = games_df['DAYS_SINCE_LAST_GAME'].apply(
        lambda x: 1 if x <= 1 else 0)

    games_df['PER'] = (games_df['PTS'] + (0.4 * games_df['FGM']) - (0.7 * games_df['FGA']) - (0.4 * (games_df['FTA'] - games_df['FTM'])) +
                       (0.7 * games_df['REB']) + (0.3 * games_df['AST']) + (0.1 * games_df['STL']) + (0.1 * games_df['BLK']) -
                       (0.1 * games_df['TOV']) - (0.2 * games_df['PF'])) / games_df['MIN']

    games_df.loc[:, 'PER'] = games_df['PER'].replace([np.inf, -np.inf], np.nan)

    games_df['ROLLING_PER'] = games_df['PER'].rolling(
        window=rolling_window).mean()

    games_df['PTS_X_MIN'] = games_df['PTS'] * games_df['MIN']
    games_df['FGM_FGA_RATIO'] = games_df['FGM'] / (games_df['FGA'] + 1e-6)
    games_df['3PM_3PA_RATIO'] = games_df['FG3M'] / (games_df['FG3A'] + 1e-6)
    games_df['REST_IMPACT'] = games_df['DAYS_SINCE_LAST_GAME'] * \
        games_df['PTS_ROLL_AVG']

    games_df['PTS_LAG_1'] = games_df['PTS'].shift(1)
    games_df['PTS_LAG_2'] = games_df['PTS'].shift(2)
    games_df['PTS_LAG_3'] = games_df['PTS'].shift(3)
    games_df['PTS_LAG_4'] = games_df['PTS'].shift(4)
    games_df['PTS_LAG_5'] = games_df['PTS'].shift(5)

    games_df['OPP_GP_RANK'] = opponent_df['GP_RANK']
    games_df['OPP_W_RANK'] = opponent_df['W_RANK']
    games_df['OPP_L_RANK'] = opponent_df['L_RANK']
    games_df['OPP_W_PCT_RANK'] = opponent_df['W_PCT_RANK']
    games_df['OPP_MIN_RANK'] = opponent_df['MIN_RANK']
    games_df['OPP_FGM_RANK'] = opponent_df['FGM_RANK']
    games_df['OPP_FGA_RANK'] = opponent_df['FGA_RANK']
    games_df['OPP_FG_PCT_RANK'] = opponent_df['FG_PCT_RANK']
    games_df['OPP_FG3M_RANK'] = opponent_df['FG3M_RANK']
    games_df['OPP_FG3A_RANK'] = opponent_df['FG3A_RANK']
    games_df['OPP_FG3_PCT_RANK'] = opponent_df['FG3_PCT_RANK']
    games_df['OPP_FTM_RANK'] = opponent_df['FTM_RANK']
    games_df['OPP_FTA_RANK'] = opponent_df['FTA_RANK']
    games_df['OPP_FT_PCT_RANK'] = opponent_df['FT_PCT_RANK']
    games_df['OPP_OREB_RANK'] = opponent_df['OREB_RANK']
    games_df['OPP_DREB_RANK'] = opponent_df['DREB_RANK']
    games_df['OPP_REB_RANK'] = opponent_df['REB_RANK']
    games_df['OPP_AST_RANK'] = opponent_df['AST_RANK']
    games_df['OPP_TOV_RANK'] = opponent_df['TOV_RANK']
    games_df['OPP_STL_RANK'] = opponent_df['STL_RANK']
    games_df['OPP_BLK_RANK'] = opponent_df['BLK_RANK']
    games_df['OPP_BLKA_RANK'] = opponent_df['BLKA_RANK']
    games_df['OPP_PF_RANK'] = opponent_df['PF_RANK']
    games_df['OPP_PFD_RANK'] = opponent_df['PFD_RANK']
    games_df['OPP_PTS_RANK'] = opponent_df['PTS_RANK']
    games_df['OPP_PLUS_MINUS_RANK'] = opponent_df['PLUS_MINUS_RANK']

    games_df = games_df.apply(pd.to_numeric, errors='coerce')

    games_df[rolling_average_labels] = games_df[rolling_average_labels].shift(
        1)

    games_df = games_df.dropna().reset_index(drop=True)

    games_df = games_df.sort_values(
        by='DAYS_SINCE_LAST_GAME', ascending=False)

    return games_df
