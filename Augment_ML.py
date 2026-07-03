"""
Ade_ML3.py

Leakage-safe ML workflow for evaluating the effect of training-set size and
synthetic-data augmentation on LR_Ade prediction.

This script is a simplified version of the primary workflow and is designed to:
  - treat the first N_RAW_ROWS rows as the real/raw dataset,
  - treat all remaining rows as synthetic data,
  - train on five dataset sizes:
        1) raw only,
        2) raw + 100 synthetic rows,
        3) raw + 200 synthetic rows,
        4) raw + 300 synthetic rows,
        5) raw + all synthetic rows,
  - for EACH dataset size, create its own 80/20 random train/test split from
    the assembled subset (mixed raw + synthetic),
  - compare Random Forest and Extra Trees only,
  - save tables only (no plots, no statistical baselines, no Ridge,
    no nested LOLO/LOPO).

Author: Pooria Ghorbani Bam
"""

# -------------------------------
# Step 0: Imports + user settings
# -------------------------------
import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import re
import json
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

warnings.filterwarnings("ignore", category=FutureWarning)



# User inputs (edit as needed)

INPUT_FILE = "Ade_ML3.xlsx"
SHEET_NAME = "Sheet1"
TARGET_COL = "LR_Ade"

# Rows 0:(N_RAW_ROWS-1) are treated as raw data; remaining rows are synthetic.
N_RAW_ROWS = 35
SYNTHETIC_SET_SIZES = [0, 25, 50, 100, 200, 300, 400]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 10
MISSING_COL_THRESHOLD = 20.0   # drop predictor columns with >20% missing

# Candidate preprocessing options folded into GridSearchCV
SCALER_CANDIDATES = [
    "passthrough"
]

K_CANDIDATES_BASE = []

# Candidate ML models and their hyperparameter grids
RF_PARAM_GRID = {
    "model__n_estimators": [100, 150],
    "model__max_depth": [None, 15],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1],
    "model__max_features": [None, 1.0],
}
ET_PARAM_GRID = {
    "model__n_estimators": [100, 150],
    "model__max_depth": [None, 15],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1],
    "model__max_features": [None, 1.0],
}



# Utilities: folders and metrics


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": _rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def safe_kfold(n_samples: int, preferred_splits: int, random_state: int) -> KFold:
    """Return a safe shuffled KFold object for the available sample size."""
    n_splits = min(preferred_splits, max(2, n_samples))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _jsonify_param(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return repr(v)



# Step 1: Load data + QA/QC summary statistics

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
df = df.reset_index(drop=True)
is_synth_full = pd.Series(0, index=df.index)
is_synth_full.iloc[N_RAW_ROWS:] = 1

print("Shape:", df.shape)
print("Target missing fraction:", df[TARGET_COL].isna().mean())
print("Duplicate rows:", df.duplicated().sum())
if "Location" in df.columns:
    print("\nLocation counts:")
    print(df["Location"].value_counts(dropna=False))



# Step 2: Handle BDL tokens + missingness table

def add_bdl_indicators_and_coerce_numeric(
    df_in: pd.DataFrame,
    bdl_cols: List[str],
    token_regex: str = r"^\s*[Xx]_isBDL\s*$"
) -> pd.DataFrame:
    dfc = df_in.copy()
    pat = re.compile(token_regex)

    for c in bdl_cols:
        s = dfc[c].astype(str)
        is_bdl = s.str.match(pat)
        dfc[c + "_isBDL"] = is_bdl.astype(int)
        dfc.loc[is_bdl, c] = np.nan
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

    return dfc


# Keep empty if this prepared input sheet has no explicit BDL-tokenized predictors.
BDL_COLS: List[str] = []

dfc = add_bdl_indicators_and_coerce_numeric(df, BDL_COLS)

missing_pct = (dfc.isna().mean() * 100).sort_values(ascending=False)
print("\nTop missingness (%):")
print(missing_pct.head(15))

if BDL_COLS:
    bdl_rates = {c: 100 * dfc[c + "_isBDL"].mean() for c in BDL_COLS}
    print("\nBDL rates (%):")
    print(pd.Series(bdl_rates).sort_values(ascending=False))


# Step 3: Build predictor matrix (Location dropped)

def build_feature_matrix(dfc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns X and y.

    Location is dropped from X to avoid site-identity leakage.
    """
    y = dfc[TARGET_COL].copy()
    X = dfc.drop(columns=[TARGET_COL, "Location"], errors="ignore").copy()
    return X, y


FEATURE_SET = "full"
X, y = build_feature_matrix(dfc)
groups = None


# Drop predictor columns with >20% missingness

feature_missing_pct = X.isna().mean() * 100
drop_cols_high_missing = feature_missing_pct[feature_missing_pct > MISSING_COL_THRESHOLD].index.tolist()

if drop_cols_high_missing:
    print(f"\nDropping predictors with >{MISSING_COL_THRESHOLD:.1f}% missingness:")
    print(pd.Series(feature_missing_pct[drop_cols_high_missing]).sort_values(ascending=False))
    X = X.drop(columns=drop_cols_high_missing, errors="ignore")

# Categorical columns actually present in X (Location is NOT in X by design)
CAT_COLS = ["Season"]  # keep Location out of predictors
for c in CAT_COLS:
    if c in X.columns:
        X[c] = X[c].astype(str)

num_cols = [c for c in X.columns if c not in CAT_COLS]
cat_cols = [c for c in CAT_COLS if c in X.columns]


# Complete-case analysis on retained predictors:
# no imputation; rows with any missing predictor/target are removed

row_mask = (~y.isna()) & (~X.isna().any(axis=1))
X = X.loc[row_mask].reset_index(drop=True)
y = y.loc[row_mask].reset_index(drop=True)
is_syn = is_synth_full.loc[row_mask].reset_index(drop=True)

if groups is not None:
    groups = groups[row_mask.to_numpy()]

print("\nFeature set:", FEATURE_SET)
print("X shape after dropping high-missing columns and incomplete rows:", X.shape)
print("Using Location as groups only:", groups is not None)
print("Retained predictors:", list(X.columns))
print("Dropped high-missing predictors:", drop_cols_high_missing)
print("Rows retained for modeling:", len(X))



# Step 4: Leakage-safe preprocessing pipeline (inside CV only)

def make_preprocess(num_cols_local, cat_cols_local) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),  # swapped in GridSearch
                ]),
                num_cols_local,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols_local,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )



# Output folders

input_path = os.path.abspath(INPUT_FILE)
base_dir = os.path.dirname(input_path)
out_dir = os.path.join(base_dir, "ML_Ade_Augmentation_Results_25")
tab_dir = os.path.join(out_dir, "tables")
_ensure_dir(out_dir)
_ensure_dir(tab_dir)



# Step 5: GridSearch helper (RF and ET only)

scoring = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}


def grid_search_for_model(
    model_name: str,
    model,
    model_param_grid: Dict[str, Any],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    cv_obj,
    num_cols_local=None,
    cat_cols_local=None,
) -> Tuple[GridSearchCV, Pipeline]:
    """Run GridSearchCV with preprocessing hyperparameters included."""
    if num_cols_local is None:
        num_cols_local = num_cols
    if cat_cols_local is None:
        cat_cols_local = cat_cols

    pipe = Pipeline([
        ("preprocess", make_preprocess(num_cols_local, cat_cols_local)),
        ("var", VarianceThreshold()),
        ("select", "passthrough"),
        ("model", model),
    ])

    pre_tmp = make_preprocess(num_cols_local, cat_cols_local)
    p_encoded = pre_tmp.fit(X_fit).transform(X_fit).shape[1]
    k_candidates = [k for k in K_CANDIDATES_BASE if k <= p_encoded] + ["all"]
    selector_candidates = ["passthrough"] + [
        SelectKBest(score_func=f_regression, k=k) for k in k_candidates
    ]

    param_grid = {
        "preprocess__num__scaler": SCALER_CANDIDATES,
        "select": selector_candidates,
    }
    param_grid.update(model_param_grid)

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit="RMSE",
        cv=cv_obj,
        n_jobs=-1,
        return_train_score=True,
        verbose=0,
    )
    grid.fit(X_fit, y_fit)

    best_pipe = grid.best_estimator_
    print(f"[{model_name}] best CV RMSE = {-grid.best_score_:.4f}")
    print(f"[{model_name}] best params = {grid.best_params_}")
    return grid, best_pipe


from sklearn.model_selection import KFold

# ---- Real / synthetic partitions by FLAG (not by position) ----
real_idx = np.where(is_syn.values == 0)[0]
syn_idx  = np.where(is_syn.values == 1)[0]
X_raw, y_raw = X.iloc[real_idx].reset_index(drop=True), y.iloc[real_idx].reset_index(drop=True)
X_syn, y_syn = X.iloc[syn_idx].reset_index(drop=True), y.iloc[syn_idx].reset_index(drop=True)
print(f"\nReal rows: {len(X_raw)} | Synthetic rows: {len(X_syn)}")

# ---- Evaluation settings ----
N_REPEATS   = 10      # reduce to 10 if too slow
OUTER_FOLDS = 5       # 35 real -> ~7 held out per fold; all 35 tested out-of-sample
INNER_FOLDS = 5

rng_master = np.random.RandomState(RANDOM_STATE)
synth_perm = rng_master.permutation(len(X_syn)) if len(X_syn) else np.array([], dtype=int)

def synth_subset(requested_size):
    if requested_size == "all":
        k = len(X_syn); label = "Raw + all synthetic"
    else:
        k = int(min(requested_size, len(X_syn)))
        label = "Raw only" if k == 0 else f"Raw + {k} synthetic"
    if k == 0:
        return X_syn.iloc[[]].copy(), y_syn.iloc[[]].copy(), 0, label
    idx = synth_perm[:k]
    return X_syn.iloc[idx].reset_index(drop=True), y_syn.iloc[idx].reset_index(drop=True), k, label

model_specs = {
    "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), RF_PARAM_GRID),
    "Extra Trees":   (ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1), ET_PARAM_GRID),
}

per_repeat_rows = []   # one metric row per (repeat, level): pooled over all real points

for rep in range(N_REPEATS):
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=1000 + rep)
    for requested_size in SYNTHETIC_SET_SIZES:
        X_syn_sub, y_syn_sub, k_used, label = synth_subset(requested_size)
        y_true_pool, y_pred_pool = [], []

        for tr_idx, te_idx in outer.split(X_raw):
            # TEST = real only; never contains synthetic
            X_te_real, y_te_real = X_raw.iloc[te_idx], y_raw.iloc[te_idx]
            # TRAIN = real-train fold + k synthetic
            X_tr = pd.concat([X_raw.iloc[tr_idx], X_syn_sub], axis=0, ignore_index=True)
            y_tr = pd.concat([y_raw.iloc[tr_idx], y_syn_sub], axis=0, ignore_index=True)

            inner_cv = safe_kfold(len(X_tr), INNER_FOLDS, RANDOM_STATE)
            best_pipe, best_cv = None, np.inf
            for name, (mdl, grid_) in model_specs.items():
                g, bp = grid_search_for_model(name, mdl, grid_, X_fit=X_tr, y_fit=y_tr, cv_obj=inner_cv)
                if -g.best_score_ < best_cv:
                    best_cv, best_pipe = -g.best_score_, bp

            best_pipe.fit(X_tr, y_tr)
            y_pred_pool.append(best_pipe.predict(X_te_real))
            y_true_pool.append(np.asarray(y_te_real, dtype=float))

        y_true_pool = np.concatenate(y_true_pool)   # all 35 real points, out-of-sample
        y_pred_pool = np.concatenate(y_pred_pool)
        m = regression_metrics(y_true_pool, y_pred_pool)
        per_repeat_rows.append({"Repeat": rep, "Subset": label, "k_used": k_used, **m})

# ---- Aggregate across repeats: mean ± SD per augmentation level ----
per_repeat_df = pd.DataFrame(per_repeat_rows)
summary_df = (per_repeat_df
    .groupby(["Subset", "k_used"], sort=False)[["RMSE", "MAE", "R2"]]
    .agg(["mean", "std"])
    .reset_index())
print(summary_df)

per_repeat_df.to_csv(os.path.join(tab_dir, "Table_S9_augmentation_real_test_per_repeat.csv"), index=False)
summary_df.to_csv(os.path.join(tab_dir, "Table_S9_augmentation_real_test_summary.csv"), index=False)

