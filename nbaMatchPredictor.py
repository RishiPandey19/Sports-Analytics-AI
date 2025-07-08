import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, BoxScoreAdvancedV2, scoreboardv2
from nba_api.stats.static import teams
from requests.exceptions import ReadTimeout, ConnectionError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setup
data_dir = "nba_data"
os.makedirs(data_dir, exist_ok=True)

TEAM_DICT = {team['full_name']: team['id'] for team in teams.get_teams()}
ID_TO_NAME = {v: k for k, v in TEAM_DICT.items()}

# Columns
BASIC_COLS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'PLUS_MINUS']
ADV_COLS = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT', 'REB_PCT', 'EFG_PCT', 'TS_PCT']
ROLLING_WINDOW = 5
EXTRA_GAMES = 3  # fetch this many extra games beyond rolling window for safety

def fetch_recent_team_games(team_id, num_games=ROLLING_WINDOW + EXTRA_GAMES, season="2023-24"):
    finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable=season)
    games = finder.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values('GAME_DATE', ascending=False)
    recent_games = games.head(num_games).sort_values('GAME_DATE')  # ascending date order for rolling
    return recent_games

def fetch_advanced_stats_with_retry(team_id, games_df, max_retries=3):
    adv_stats_list = []
    for game_id in games_df['GAME_ID']:
        for attempt in range(max_retries):
            try:
                adv_df = BoxScoreAdvancedV2(game_id=game_id).get_data_frames()[0]
                team_adv = adv_df[adv_df['TEAM_ID'] == team_id]
                if not team_adv.empty:
                    adv_stats_list.append(team_adv[['GAME_ID'] + ADV_COLS])
                time.sleep(0.5)  # short delay to avoid rate limiting
                break
            except (ReadTimeout, ConnectionError) as e:
                wait = 2 ** attempt
                print(f"Timeout/Error fetching {game_id}, retry {attempt+1} in {wait}s: {e}")
                time.sleep(wait)
            except Exception as e:
                print(f"Non-retryable error for game {game_id}: {e}")
                break
    if adv_stats_list:
        adv_stats = pd.concat(adv_stats_list)
        adv_stats = adv_stats.sort_values('GAME_ID')
        return adv_stats
    else:
        return pd.DataFrame(columns=['GAME_ID'] + ADV_COLS)

def collect_team_data(team_name, season="2023-24"):
    team_id = TEAM_DICT[team_name]
    basic_stats = fetch_recent_team_games(team_id, season=season)
    if basic_stats.empty:
        return pd.DataFrame()
    adv_stats = fetch_advanced_stats_with_retry(team_id, basic_stats)
    merged = pd.merge(basic_stats, adv_stats, on='GAME_ID', how='left')
    merged['TEAM'] = team_name
    merged = merged.sort_values('GAME_DATE').reset_index(drop=True)
    return merged

def prepare_dataset(season="2023-24"):
    print("Collecting recent data for all teams with rolling averages of basic + advanced stats...")
    all_dfs = []
    team_list = list(TEAM_DICT.keys())#[:10]  number of teams
    for name in team_list:
        print(f"Fetching data for {name}...")
        df = collect_team_data(name, season=season)
        if not df.empty:
            stats_cols = BASIC_COLS[:-1] + ADV_COLS
            rolling_stats = (
                df.groupby('TEAM', group_keys=False)[stats_cols]
                .apply(lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean().shift(1))
            )
            rolling_stats = rolling_stats.reset_index(drop=True)
            combined = df.reset_index(drop=True).join(rolling_stats, rsuffix='_ROLL')
            combined = combined.dropna(subset=[col + '_ROLL' for col in stats_cols])
            all_dfs.append(combined)
    if not all_dfs:
        raise ValueError("No data collected with rolling averages. Try adjusting the rolling window or team selection.")
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(os.path.join(data_dir, "nba_basic_advanced_rolling_recent.csv"), index=False)
    return full_df

def train_model(df):
    feature_cols = [col + '_ROLL' for col in BASIC_COLS[:-1] + ADV_COLS]
    df = df.dropna(subset=feature_cols + ['PLUS_MINUS'])
    if df.empty:
        raise ValueError("No valid data after cleaning for training.")
    X = df[feature_cols]
    y = (df['PLUS_MINUS'] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print("Model Accuracy with Basic + Advanced Rolling Averages (Recent Games):", acc)
    return clf

def get_latest_rolling_features(team_name, df):
    team_games = df[df['TEAM'] == team_name]
    if team_games.empty:
        return None
    last_game = team_games.iloc[-1]
    feature_cols = [col + '_ROLL' for col in BASIC_COLS[:-1] + ADV_COLS]
    features = last_game[feature_cols]
    return pd.DataFrame([features.values], columns=feature_cols)

def predict_today_games(model, rolling_df):
    print("\n=== Today's Game Predictions ===")
    scoreboard = scoreboardv2.ScoreboardV2()
    games_df = scoreboard.get_data_frames()[0]
    feature_cols = [col + '_ROLL' for col in BASIC_COLS[:-1] + ADV_COLS]

    for _, row in games_df.iterrows():
        home_team = ID_TO_NAME[row['HOME_TEAM_ID']]
        away_team = ID_TO_NAME[row['VISITOR_TEAM_ID']]

        home_stats = get_latest_rolling_features(home_team, rolling_df)
        away_stats = get_latest_rolling_features(away_team, rolling_df)

        if home_stats is None:
            home_stats = pd.DataFrame(np.random.rand(len(feature_cols)).reshape(1, -1), columns=feature_cols)
        if away_stats is None:
            away_stats = pd.DataFrame(np.random.rand(len(feature_cols)).reshape(1, -1), columns=feature_cols)

        home_pred = model.predict_proba(home_stats)[0][1]
        away_pred = model.predict_proba(away_stats)[0][1]

        winner = home_team if home_pred > away_pred else away_team
        print(f"{away_team} at {home_team} → Predicted Winner: {winner}")

def mock_matchup(model, rolling_df, team1, team2):
    feature_cols = [col + '_ROLL' for col in BASIC_COLS[:-1] + ADV_COLS]

    team1_stats = get_latest_rolling_features(team1, rolling_df)
    team2_stats = get_latest_rolling_features(team2, rolling_df)

    if team1_stats is None:
        team1_stats = pd.DataFrame(np.random.rand(len(feature_cols)).reshape(1, -1), columns=feature_cols)
    if team2_stats is None:
        team2_stats = pd.DataFrame(np.random.rand(len(feature_cols)).reshape(1, -1), columns=feature_cols)

    team1_pred = model.predict_proba(team1_stats)[0][1]
    team2_pred = model.predict_proba(team2_stats)[0][1]

    winner = team1 if team1_pred > team2_pred else team2
    print(f"\n=== Mock Matchup: {team1} vs {team2} → Predicted Winner: {winner} ===")

if __name__ == "__main__":
    rolling_df = prepare_dataset(season="2023-24")
    model = train_model(rolling_df)
    predict_today_games(model, rolling_df)
    mock_matchup(model, rolling_df, "Boston Celtics", "Los Angeles Lakers")
