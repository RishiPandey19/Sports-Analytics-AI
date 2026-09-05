from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from nba_api.stats.endpoints import BoxScoreAdvancedV2, leaguegamefinder, scoreboardv2
from nba_api.stats.static import teams
from requests.exceptions import ConnectionError, ReadTimeout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_DIR = "nba_data"
DEFAULT_SEASON = "2023-24"
BASIC_COLS = ["PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FT_PCT", "PLUS_MINUS"]
ADV_COLS = ["OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "REB_PCT", "EFG_PCT", "TS_PCT"]
STATS_COLS = BASIC_COLS[:-1] + ADV_COLS
FEATURE_COLS = [f"{col}_ROLL" for col in STATS_COLS]
ROLLING_WINDOW = 5
DEFAULT_GAMES_PER_TEAM = 45

TEAM_DICT = {team["full_name"]: team["id"] for team in teams.get_teams()}
ID_TO_NAME = {team_id: name for name, team_id in TEAM_DICT.items()}


def fetch_recent_team_games(team_id: int, num_games: int, season: str) -> pd.DataFrame:
    finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable=season)
    games = finder.get_data_frames()[0]
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE", ascending=False)
    return games.head(num_games).sort_values("GAME_DATE")


def fetch_advanced_stats_with_retry(
    team_id: int,
    games_df: pd.DataFrame,
    max_retries: int = 3,
) -> pd.DataFrame:
    advanced_frames = []

    for game_id in games_df["GAME_ID"]:
        for attempt in range(max_retries):
            try:
                advanced_df = BoxScoreAdvancedV2(game_id=game_id).get_data_frames()[0]
                team_advanced = advanced_df[advanced_df["TEAM_ID"] == team_id]
                if not team_advanced.empty:
                    advanced_frames.append(team_advanced[["GAME_ID"] + ADV_COLS])
                time.sleep(0.5)
                break
            except (ReadTimeout, ConnectionError) as error:
                wait_seconds = 2**attempt
                print(f"Retrying advanced stats for {game_id} in {wait_seconds}s: {error}")
                time.sleep(wait_seconds)
            except Exception as error:
                print(f"Skipping advanced stats for {game_id}: {error}")
                break

    if not advanced_frames:
        return pd.DataFrame(columns=["GAME_ID"] + ADV_COLS)

    return pd.concat(advanced_frames, ignore_index=True).sort_values("GAME_ID")


def collect_team_data(team_name: str, season: str, games_per_team: int) -> pd.DataFrame:
    team_id = TEAM_DICT[team_name]
    basic_stats = fetch_recent_team_games(team_id, num_games=games_per_team, season=season)
    if basic_stats.empty:
        return pd.DataFrame()

    advanced_stats = fetch_advanced_stats_with_retry(team_id, basic_stats)
    merged = pd.merge(basic_stats, advanced_stats, on="GAME_ID", how="left")
    merged["TEAM"] = team_name
    return merged.sort_values("GAME_DATE").reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    rolling_stats = (
        df.groupby("TEAM", group_keys=False)[STATS_COLS]
        .apply(lambda frame: frame.rolling(window=ROLLING_WINDOW, min_periods=1).mean().shift(1))
        .reset_index(drop=True)
    )
    combined = df.reset_index(drop=True).join(rolling_stats, rsuffix="_ROLL")
    return combined.dropna(subset=FEATURE_COLS)


def prepare_dataset(season: str = DEFAULT_SEASON, games_per_team: int = DEFAULT_GAMES_PER_TEAM) -> pd.DataFrame:
    print("Collecting NBA team data and computing rolling pre-game features...")
    all_frames = []

    for team_name in TEAM_DICT:
        print(f"Fetching data for {team_name}...")
        team_df = collect_team_data(team_name, season=season, games_per_team=games_per_team)
        if not team_df.empty:
            all_frames.append(add_rolling_features(team_df))

    if not all_frames:
        raise ValueError("No data collected with valid rolling features.")

    full_df = pd.concat(all_frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "nba_basic_advanced_rolling_recent.csv")
    full_df.to_csv(output_path, index=False)
    print(f"Saved prepared dataset to {output_path}")
    return full_df


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    clean_df = df.dropna(subset=FEATURE_COLS + ["PLUS_MINUS"])
    if clean_df.empty:
        raise ValueError("No valid data after cleaning for training.")

    X = clean_df[FEATURE_COLS]
    y = (clean_df["PLUS_MINUS"] > 0).astype(int)
    if y.nunique() < 2:
        raise ValueError("Training labels contain only one class.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"Holdout accuracy: {accuracy:.3f}")
    print(f"Majority-class baseline: {baseline:.3f}")
    return model


def get_latest_rolling_features(team_name: str, df: pd.DataFrame) -> pd.DataFrame | None:
    team_games = df[df["TEAM"] == team_name].sort_values("GAME_DATE")
    if team_games.empty:
        return None

    features = team_games.iloc[-1][FEATURE_COLS]
    if features.isna().any():
        return None

    return pd.DataFrame([features.values], columns=FEATURE_COLS)


def predict_matchup(
    model: RandomForestClassifier,
    rolling_df: pd.DataFrame,
    team1: str,
    team2: str,
) -> str | None:
    team1_stats = get_latest_rolling_features(team1, rolling_df)
    team2_stats = get_latest_rolling_features(team2, rolling_df)

    if team1_stats is None or team2_stats is None:
        missing = [team for team, stats in [(team1, team1_stats), (team2, team2_stats)] if stats is None]
        print(f"Skipping {team1} vs {team2}: missing rolling features for {', '.join(missing)}.")
        return None

    team1_probability = model.predict_proba(team1_stats)[0][1]
    team2_probability = model.predict_proba(team2_stats)[0][1]
    winner = team1 if team1_probability > team2_probability else team2
    print(
        f"{team1} win score: {team1_probability:.3f} | "
        f"{team2} win score: {team2_probability:.3f} -> Predicted winner: {winner}"
    )
    return winner


def predict_today_games(
    model: RandomForestClassifier,
    rolling_df: pd.DataFrame,
    game_date: str | None = None,
) -> None:
    print("\n=== NBA Game Predictions ===")
    scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date) if game_date else scoreboardv2.ScoreboardV2()
    games_df = scoreboard.get_data_frames()[0]

    if games_df.empty:
        print("No games found for the selected date.")
        return

    for _, row in games_df.iterrows():
        home_team = ID_TO_NAME.get(row["HOME_TEAM_ID"])
        away_team = ID_TO_NAME.get(row["VISITOR_TEAM_ID"])
        if not home_team or not away_team:
            print("Skipping game with unknown team id.")
            continue
        print(f"\n{away_team} at {home_team}")
        predict_matchup(model, rolling_df, home_team, away_team)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple NBA matchup predictor.")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="NBA season, such as 2023-24.")
    parser.add_argument("--games-per-team", type=int, default=DEFAULT_GAMES_PER_TEAM)
    parser.add_argument("--game-date", default=None, help="Optional scoreboard date in MM/DD/YYYY format.")
    parser.add_argument("--team1", default="Boston Celtics", help="First team for the mock matchup.")
    parser.add_argument("--team2", default="Los Angeles Lakers", help="Second team for the mock matchup.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rolling_df = prepare_dataset(season=args.season, games_per_team=args.games_per_team)
    trained_model = train_model(rolling_df)
    predict_today_games(trained_model, rolling_df, game_date=args.game_date)
    print(f"\n=== Mock Matchup: {args.team1} vs {args.team2} ===")
    predict_matchup(trained_model, rolling_df, args.team1, args.team2)
