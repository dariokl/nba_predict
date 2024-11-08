from datetime import datetime
from datetime import timedelta
import os
import pandas as pd

from .preprocessing import get_player_recent_performance


def fill_win_column():
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    csv_file = os.path.join(os.path.dirname(__file__),
                            '..', f'predictions_{yesterday}.csv')

    if not os.path.exists(csv_file):
        print(f"{csv_file} does not exist.")
        return

    predictions_df = pd.read_csv(csv_file)

    for index, row in predictions_df.iterrows():
        player_name = row['player_name']
        threshold = row['threshold']
        over_under = row['over_under']

        performance = get_player_recent_performance(player_name)

        actual_points = performance['PTS'].iloc[0] if isinstance(
            performance, pd.DataFrame) else performance['PTS']

        predictions_df.at[index, 'scored_points'] = actual_points

        if (over_under == 'Over' and actual_points > threshold) or (over_under == 'Under' and actual_points <= threshold):
            predictions_df.at[index, 'win'] = 'W'
        else:
            predictions_df.at[index, 'win'] = 'L'

    predictions_df.to_csv(csv_file, index=False)
    print(f"Win column and scored_points updated and saved to {csv_file}")


def predictions_stats():
    csv_file = os.path.join(os.path.dirname(__file__),
                            '..', 'all_results.csv')

    df = pd.read_csv(csv_file)

    total_games = len(df)

    wins = df['win'].value_counts().get('W', 0)

    win_percentage = (wins / total_games) * 100 if total_games > 0 else 0

    print(f"Total games: {total_games}")
    print(f"Wins: {wins}")
    print(f"Win percentage: {win_percentage:.2f}%")
