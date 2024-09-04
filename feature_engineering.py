import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectKBest

def create_team_stats(df):
    # Assuming 'team_id' and 'opponent_id' columns exist
    team_stats = df.groupby('team_id').agg({
        'goals_scored': 'mean',
        'goals_conceded': 'mean',
        'shots_on_target': 'mean',
        'possession': 'mean',
        'win': 'mean',
        'draw': 'mean',
        'loss': 'mean'
    })
    return team_stats

def create_head_to_head_features(df):
    # Assuming 'team_id', 'opponent_id', and 'result' columns exist
    h2h = df.groupby(['team_id', 'opponent_id']).agg({
        'result': ['count', lambda x: (x == 'win').mean(), lambda x: (x == 'draw').mean()]
    })
    h2h.columns = ['h2h_matches', 'h2h_win_rate', 'h2h_draw_rate']
    return h2h

def create_form_features(df, n_matches=5):
    # Assuming 'team_id' and 'result' columns exist
    df = df.sort_values('date')
    df['points'] = df['result'].map({'win': 3, 'draw': 1, 'loss': 0})
    df['form'] = df.groupby('team_id')['points'].rolling(window=n_matches, min_periods=1).mean().reset_index(0, drop=True)
    return df['form']

def create_odds_features(df):
    # Assuming 'home_odds', 'draw_odds', and 'away_odds' columns exist
    df['implied_home_prob'] = 1 / df['home_odds']
    df['implied_draw_prob'] = 1 / df['draw_odds']
    df['implied_away_prob'] = 1 / df['away_odds']
    df['odds_sum'] = df['implied_home_prob'] + df['implied_draw_prob'] + df['implied_away_prob']
    df['bookmaker_margin'] = df['odds_sum'] - 1
    return df

def select_best_features(X, y, k=20):
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features

def engineer_features(df):
    team_stats = create_team_stats(df)
    h2h_features = create_head_to_head_features(df)
    df['form'] = create_form_features(df)
    df = create_odds_features(df)

    # Merge additional features
    df = pd.merge(df, team_stats, left_on='team_id', right_index=True, suffixes=('', '_team'))
    df = pd.merge(df, team_stats, left_on='opponent_id', right_index=True, suffixes=('', '_opponent'))
    df = pd.merge(df, h2h_features, on=['team_id', 'opponent_id'])

    # Create interaction features
    df['goal_diff_expectation'] = df['goals_scored_team'] - df['goals_conceded_opponent']
    df['form_diff'] = df['form_team'] - df['form_opponent']

    # Select best features
    target = 'result'  # Assuming 'result' is the target variable
    features = df.columns.drop(target)
    selected_features = select_best_features(df[features], df[target])

    return df[selected_features + [target]]

if __name__ == "__main__":
    from data_fetcher import get_all_data
    from data_preprocessing import preprocess_data

    pinnacle_data, livescore_data, api_football_data = get_all_data()
    preprocessed_data = preprocess_data(pinnacle_data, livescore_data, api_football_data)
    engineered_data = engineer_features(preprocessed_data)
    print(engineered_data.head())
    print(engineered_data.info())
