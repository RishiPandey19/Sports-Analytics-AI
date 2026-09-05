# Sports Analytics AI

A Python project for predicting NBA matchup winners from recent team performance. The script collects NBA game logs, joins basic and advanced box-score metrics, builds rolling pre-game features, and trains a Random Forest classifier to estimate which team has the stronger win profile for a matchup.

This is a sports analytics learning project, not a betting system. The goal is to show an end-to-end workflow: data collection, feature engineering, model training, evaluation against a baseline, and game-level prediction.

## What It Does

- Pulls NBA team game logs with `nba_api`
- Adds advanced box-score metrics such as offensive rating, defensive rating, effective field goal percentage, true shooting percentage, and rebound percentage
- Builds 5-game rolling averages shifted by one game so predictions use only prior team performance
- Trains a `RandomForestClassifier` to predict whether a team finished a game with positive plus-minus
- Reports holdout accuracy and a majority-class baseline
- Predicts scheduled games for a selected date or runs a mock matchup between two teams
- Skips predictions when team features are unavailable instead of generating fake replacement statistics
- Falls back to available real box-score features if advanced box-score data is unavailable from the NBA endpoint

## Tech Stack

- Python 3.10+
- pandas
- nba_api
- scikit-learn
- requests

## Setup

```bash
git clone https://github.com/RishiPandey19/Sports-Analytics-AI.git
cd Sports-Analytics-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Train on recent games from the default season and predict the current scoreboard:

```bash
python nbaMatchPredictor.py
```

Use a specific NBA season:

```bash
python nbaMatchPredictor.py --season 2023-24
```

Use more or fewer recent games per team:

```bash
python nbaMatchPredictor.py --games-per-team 60
```

Run a faster smoke check with only selected teams:

```bash
python nbaMatchPredictor.py \
  --teams "Boston Celtics,Los Angeles Lakers" \
  --games-per-team 8
```

Predict games for a specific scoreboard date:

```bash
python nbaMatchPredictor.py --game-date 12/25/2023
```

Run a custom mock matchup:

```bash
python nbaMatchPredictor.py \
  --team1 "Boston Celtics" \
  --team2 "Los Angeles Lakers"
```

Prepared data is saved under `nba_data/nba_basic_advanced_rolling_recent.csv`.

## Modeling Approach

Each team-game row is converted into pre-game features by taking rolling averages of recent performance and shifting those averages by one game. That avoids using the target game's own statistics as inputs.

The target is whether the team finished the game with positive plus-minus. For matchup prediction, the script compares each team's model-estimated win score from its latest rolling features and selects the higher score.

Features include:

- Points, rebounds, assists, steals, and blocks
- Field goal percentage and free throw percentage
- Offensive, defensive, and net rating
- Assist, rebound, effective field goal, and true shooting percentages

If advanced box-score data is unavailable for a run, the model trains on the real basic box-score features that were successfully collected rather than filling missing values with synthetic statistics.

## Evaluation

The script prints two numbers after training:

```text
Holdout accuracy: 0.xxx
Majority-class baseline: 0.xxx
```

The baseline matters because NBA win/loss data can be imbalanced over small samples. A useful model should be compared against that simple baseline, not only against raw accuracy.

## Limitations

- Uses recent box-score history only; it does not include injuries, rest days, travel, betting lines, roster changes, or home-court adjustments as explicit features.
- `nba_api` depends on NBA stats endpoints that can rate-limit or temporarily fail.
- The Random Forest score is useful for comparing teams in this simplified feature space, but it is not calibrated as a true betting probability.
- The default dataset is intentionally lightweight so the project can run locally.

## Repository Contents

```text
Sports-Analytics-AI/
  nbaMatchPredictor.py
  README.md
  requirements.txt
  .gitignore
```
