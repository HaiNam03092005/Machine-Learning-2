"""
NBA Championship Prediction - Machine Learning Pipeline
=======================================================
Targets:
  1. Is_Playoff    - Did the team make the playoffs?
  2. Is_Semifinal  - Did the team reach the Conference Finals?
  3. Is_Champion   - Did the team win the championship?

Data: Basketball-Reference season-average team stats (2010-2025)
Predict: 2026 season outcomes
"""

# ==============================================================
# 0.  IMPORTS
# ==============================================================
import sys
import io

# Force UTF-8 stdout so the script works on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import joblib


# ==============================================================
# 1.  DATA LOADING  (mirrors clean_dataset.py)
# ==============================================================
FOLDER_PATH = r"C:\Users\ADMIN\Desktop\NBA_prediction\nba_history"

nba_history = {
    2010: {"champion": "Los Angeles Lakers",    "semifinals": ["Los Angeles Lakers", "Phoenix Suns", "Boston Celtics", "Orlando Magic"]},
    2011: {"champion": "Dallas Mavericks",      "semifinals": ["Dallas Mavericks", "Miami Heat", "Chicago Bulls", "Oklahoma City Thunder"]},
    2012: {"champion": "Miami Heat",            "semifinals": ["Miami Heat", "Oklahoma City Thunder", "Boston Celtics", "San Antonio Spurs"]},
    2013: {"champion": "Miami Heat",            "semifinals": ["Miami Heat", "San Antonio Spurs", "Indiana Pacers", "Memphis Grizzlies"]},
    2014: {"champion": "San Antonio Spurs",     "semifinals": ["San Antonio Spurs", "Miami Heat", "Indiana Pacers", "Oklahoma City Thunder"]},
    2015: {"champion": "Golden State Warriors", "semifinals": ["Golden State Warriors", "Cleveland Cavaliers", "Atlanta Hawks", "Houston Rockets"]},
    2016: {"champion": "Cleveland Cavaliers",   "semifinals": ["Cleveland Cavaliers", "Golden State Warriors", "Oklahoma City Thunder", "Toronto Raptors"]},
    2017: {"champion": "Golden State Warriors", "semifinals": ["Golden State Warriors", "Cleveland Cavaliers", "Boston Celtics", "San Antonio Spurs"]},
    2018: {"champion": "Golden State Warriors", "semifinals": ["Golden State Warriors", "Cleveland Cavaliers", "Boston Celtics", "Houston Rockets"]},
    2019: {"champion": "Toronto Raptors",       "semifinals": ["Toronto Raptors", "Golden State Warriors", "Milwaukee Bucks", "Portland Trail Blazers"]},
    2020: {"champion": "Los Angeles Lakers",    "semifinals": ["Los Angeles Lakers", "Miami Heat", "Boston Celtics", "Denver Nuggets"]},
    2021: {"champion": "Milwaukee Bucks",       "semifinals": ["Milwaukee Bucks", "Phoenix Suns", "Atlanta Hawks", "Los Angeles Clippers"]},
    2022: {"champion": "Golden State Warriors", "semifinals": ["Golden State Warriors", "Boston Celtics", "Miami Heat", "Dallas Mavericks"]},
    2023: {"champion": "Denver Nuggets",        "semifinals": ["Denver Nuggets", "Miami Heat", "Boston Celtics", "Los Angeles Lakers"]},
    2024: {"champion": "Boston Celtics",        "semifinals": ["Boston Celtics", "Dallas Mavericks", "Indiana Pacers", "Minnesota Timberwolves"]},
    2025: {"champion": "Oklahoma City Thunder", "semifinals": ["Oklahoma City Thunder", "Indiana Pacers", "New York Knicks", "Minnesota Timberwolves"]},
    2026: {"champion": None,                    "semifinals": ["Cleveland Cavaliers", "New York Knicks", "San Antonio Spurs", "Oklahoma City Thunder"]},
}


def load_and_label_data(folder_path):
    """Load all XLS files, concatenate, and attach target labels."""
    file_list = glob.glob(os.path.join(folder_path, "*.xls*"))
    if not file_list:
        raise FileNotFoundError("No XLS files found in: " + folder_path)

    all_seasons = []
    for fp in file_list:
        fname = os.path.basename(fp)
        season_year = int(fname.split(".")[0])
        try:
            df = pd.read_html(fp)[0]
        except Exception:
            df = pd.read_excel(fp)
        df["Season"] = season_year
        all_seasons.append(df)

    master = pd.concat(all_seasons, ignore_index=True)

    # Flatten MultiIndex columns produced by read_html
    if isinstance(master.columns, pd.MultiIndex):
        master.columns = master.columns.get_level_values(1)

    # Restore 'Season' if flattening wiped it
    if "" in master.columns:
        master.rename(columns={"": "Season"}, inplace=True)

    # Drop unnamed / empty columns
    drop_cols = [c for c in master.columns if "Unnamed" in str(c) or str(c).strip() == ""]
    master.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ---- Target labels ------------------------------------------------
    master["Is_Playoff"] = master["Team"].apply(lambda x: 1 if "*" in str(x) else 0)
    master["Team"] = master["Team"].str.replace("*", "", regex=False).str.strip()

    def assign_labels(row):
        season, team = row["Season"], row["Team"]
        is_semi, is_champ = 0, 0
        if season in nba_history:
            if team in nba_history[season]["semifinals"]:
                is_semi = 1
            if team == nba_history[season]["champion"]:
                is_champ = 1
        return pd.Series([is_semi, is_champ])

    master[["Is_Semifinal", "Is_Champion"]] = master.apply(assign_labels, axis=1)
    return master


# ==============================================================
# 2.  FEATURE ENGINEERING
# ==============================================================
# Core Basketball-Reference columns (numeric team stats)
STAT_COLS = [
    "W", "L", "W/L%",
    "PS/G",   # Points scored per game
    "PA/G",   # Points allowed per game
    "SRS",    # Simple Rating System  (strongest single predictor)
    "MOV",    # Margin of Victory
    "SOS",    # Strength of Schedule
]

ENGINEERED = [
    "Point_Diff",  # PS/G - PA/G
    "Win_Rate",    # W / (W + L)
]

TARGETS = ["Is_Playoff", "Is_Semifinal", "Is_Champion"]


def engineer_features(df):
    """Create derived features and coerce stat columns to numeric."""
    df = df.copy()

    for col in STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove repeated header rows embedded in the data (Team == 'Team')
    df = df[df["Team"] != "Team"].copy()

    if "PS/G" in df.columns and "PA/G" in df.columns:
        df["Point_Diff"] = df["PS/G"] - df["PA/G"]
    if "W" in df.columns and "L" in df.columns:
        df["Win_Rate"] = df["W"].astype(float) / (df["W"].astype(float) + df["L"].astype(float) + 1e-9)

    return df


def get_feature_cols(df):
    """Return feature columns that exist in the dataframe."""
    candidates = STAT_COLS + ENGINEERED
    return [c for c in candidates if c in df.columns]


# ==============================================================
# 3.  MODEL DEFINITIONS
# ==============================================================
def build_ensemble(random_state=42):
    """Soft-voting ensemble: Logistic Regression + Random Forest + GBM."""
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    rf  = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=random_state)
    gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=random_state)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gbm", gbm)],
        voting="soft",
    )
    return ensemble


# ==============================================================
# 4.  TRAINING & EVALUATION
# ==============================================================
def train_and_evaluate(X_train, y_train, X_test, y_test, target_name, random_state=42):
    """Train ensemble, cross-validate, and print evaluation metrics."""
    print("\n" + "=" * 60)
    print("  TARGET: " + target_name)
    print("=" * 60)

    model = build_ensemble(random_state)

    # 5-fold stratified cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print("  Cross-Val ROC-AUC : {:.4f} +/- {:.4f}".format(cv_auc.mean(), cv_auc.std()))

    # Final fit on full training set
    model.fit(X_train, y_train)

    # Test set evaluation
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)

    print("  Test  ROC-AUC    : {:.4f}".format(test_auc))
    print("\n  Classification Report (test set):")
    print(classification_report(y_test, y_pred, digits=3))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


# ==============================================================
# 5.  MAIN PIPELINE
# ==============================================================
# Known 2026 Conference Finalists (teams in the current playoffs)
FINALISTS_2026 = [
    "Cleveland Cavaliers",
    "New York Knicks",
    "San Antonio Spurs",
    "Oklahoma City Thunder",
]


def main():
    print("[NBA] NBA PREDICTION -- MACHINE LEARNING PIPELINE")
    print("=" * 60)

    # ---- 5.1  Load & label data ------------------------------------
    print("\n[1/5] Loading data ...")
    master_df = load_and_label_data(FOLDER_PATH)
    master_df = engineer_features(master_df)

    FEATURE_COLS = get_feature_cols(master_df)
    print("      Features used : " + str(FEATURE_COLS))
    print("      Total rows    : " + str(len(master_df)))

    # ---- 5.2  Train / test / predict split -------------------------
    # Seasons 2010-2024 => training
    # Season  2025      => hold-out test
    # Season  2026      => live prediction
    #
    # NOTE: There is no 2026 XLS file yet (season is still in progress).
    # We use the 2025 regular-season stats for the 4 known Conference
    # Finalists as a best proxy for predicting the 2026 champion.
    print("\n[2/5] Splitting data ...")
    TRAIN_SEASONS = list(range(2010, 2025))
    TEST_SEASON   = 2025

    train_df = master_df[master_df["Season"].isin(TRAIN_SEASONS)].dropna(subset=FEATURE_COLS)
    test_df  = master_df[master_df["Season"] == TEST_SEASON].dropna(subset=FEATURE_COLS)

    # Build prediction set: 2025 stats for the 2026 Conference Finalists
    pred_df = (
        master_df[
            (master_df["Season"] == TEST_SEASON) &
            (master_df["Team"].isin(FINALISTS_2026))
        ]
        .dropna(subset=FEATURE_COLS)
        .copy()
    )
    pred_df["Note"] = "2025 stats used as 2026 proxy"

    print("      Train rows    : " + str(len(train_df)))
    print("      Test rows     : " + str(len(test_df)))
    print("      Predict rows  : " + str(len(pred_df)) + "  (2026 Conference Finalists, 2025 stats)")

    X_train = train_df[FEATURE_COLS].values.astype(float)
    X_test  = test_df[FEATURE_COLS].values.astype(float)
    X_pred  = pred_df[FEATURE_COLS].values.astype(float)

    # ---- 5.3  Train one model per target ---------------------------
    print("\n[3/5] Training models ...")
    trained_models = {}
    for target in TARGETS:
        y_train = train_df[target]
        y_test  = test_df[target]

        if y_test.sum() == 0:
            print("\n  [WARNING] Skipping AUC for " + target + " -- no positive samples in test set.")
            model = build_ensemble()
            model.fit(X_train, y_train)
            trained_models[target] = model
            continue

        model = train_and_evaluate(X_train, y_train, X_test, y_test, target)
        trained_models[target] = model

    # ---- 5.4  2026 Predictions ------------------------------------
    print("\n[4/5] Predicting 2026 season ...")
    results_2026 = pred_df[["Team", "Season"] + FEATURE_COLS].copy()

    for target in TARGETS:
        proba = trained_models[target].predict_proba(X_pred)[:, 1]
        results_2026["P(" + target + ")"] = proba

    results_2026 = results_2026.sort_values("P(Is_Champion)", ascending=False).reset_index(drop=True)

    display_cols = ["Team", "P(Is_Playoff)", "P(Is_Semifinal)", "P(Is_Champion)"]
    available    = [c for c in display_cols if c in results_2026.columns]

    print("\n  [RANKINGS]  2026 NBA Championship Probability Rankings:")
    print("  " + "-" * 68)
    with pd.option_context("display.max_rows", 40, "display.float_format", "{:.3f}".format):
        print(results_2026[available].to_string(index=False))

    print("\n  [TOP-3]  Top Championship Candidates:")
    for rank, row in results_2026.head(3).iterrows():
        print("     #{:d}  {:<30}  Champion Prob: {:.1%}".format(rank + 1, row["Team"], row["P(Is_Champion)"]))

    # ---- 5.5  Save models & predictions ---------------------------
    print("\n[5/5] Saving artifacts ...")
    save_dir = os.path.dirname(os.path.abspath(__file__))

    for target, model in trained_models.items():
        model_path = os.path.join(save_dir, "model_" + target + ".pkl")
        joblib.dump(model, model_path)
        print("      Saved -> " + model_path)

    csv_path = os.path.join(save_dir, "predictions_2026.csv")
    results_2026[available].to_csv(csv_path, index=False)
    print("      Saved -> " + csv_path)

    print("\n[DONE]  Pipeline complete!\n")
    return trained_models, results_2026


# ==============================================================
# ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    trained_models, predictions_2026 = main()
