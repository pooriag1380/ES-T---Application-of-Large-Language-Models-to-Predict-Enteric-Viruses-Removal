"""
Ent_ML2.py

Leakage-safe ML workflow for predicting LR_Ent with:
  - Standard split + KFold GridSearch (internal performance + figures)
  - Nested Leave-One-Location-Out (LOLO/LOPO) evaluation:
      Outer loop  : LeaveOneGroupOut (Location held out)
      Inner loop  : GroupKFold over remaining Locations (tuning)
                   (falls back to KFold if not enough groups)

Author: Pooria Ghorbani Bam
"""

# -------------------------------
# Step 0: Imports + user settings
# -------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import re
import json
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.ioff()

import matplotlib.cm as cm
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    LeaveOneGroupOut,
    GroupKFold,
    cross_val_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# User inputs (edit as needed)
# -----------------------------
INPUT_FILE = "Ent_ML2.xlsx"
SHEET_NAME = "Sheet1"
TARGET_COL = "LR_Ent"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 10

MISSING_COL_THRESHOLD = 20.0   # drop predictor columns with >20% missing

# Nested LOPO settings
RUN_NESTED_LOPO = False
INNER_GROUPK_SPLITS_MAX = 5  # inner tuning CV splits across locations (max)

# Figures / outputs
SAVE_FIGURES = True
SHOW_FIGURES = False
FIG_DPI = 600
plt.rcParams.update({'font.family': 'Times New Roman'})

# Permutation importance settings
PERM_N_REPEATS = 20
PERM_MAX_SAMPLES = 1.0

# Robustness check: repeat RF/ET with different seeds
RF_SEEDS = [1, 2, 3]

# Candidate preprocessing options to fold into GridSearchCV
SCALER_CANDIDATES = [
    "passthrough",
    StandardScaler(),
    RobustScaler(),
    MinMaxScaler(),
]

# Feature-selection candidates are defined *after* we know p_encoded
K_CANDIDATES_BASE = [10, 20, 40]
SELECTOR_CANDIDATES = None

# Candidate ML models and their hyperparameter grids
RF_PARAM_GRID = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [None, 15, 20],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1],
    "model__max_features": [None, 1.0, "sqrt"],
}
ET_PARAM_GRID = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [None, 15, 20],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1],
    "model__max_features": [None, 1.0, "sqrt"],
}


RIDGE_PARAM_GRID = {
    "model__alpha": [0.1, 10.0, 100.0]
}


# ---------------------------------
# Utilities: folders, metrics, plots
# ---------------------------------
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

def set_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 22,
        "axes.titlesize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 16,
        "axes.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })

def save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    if not SHOW_FIGURES:
        plt.close(fig)

def plot_actual_vs_pred(y_train, yhat_train, y_test, yhat_test, out_path: str) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_train, yhat_train, s=50, color='blue', label='Train data', alpha=0.6)
    ax.scatter(y_test, yhat_test, s=50, color='orange', label='Test data', alpha=0.6)

    y_all = np.concatenate([y_train, y_test])
    lo = float(np.nanmin(y_all))
    hi = float(np.nanmax(y_all))
    pad = 0.05 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=2)

    x_fit = np.concatenate([y_train, y_test])
    y_fit = np.concatenate([yhat_train, yhat_test])
    coef = np.polyfit(x_fit, y_fit, deg=1)
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, coef[0] * xs + coef[1], linewidth=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured Enterovirus Log-removal")
    ax.set_ylabel("Predicted Enterovirus Log-removal")
    ax.legend(frameon=True)

    m = regression_metrics(y_test, yhat_test)
    ax.text(
        0.02, 0.98,
        f"Test RMSE={m['RMSE']:.3f}\nTest MAE={m['MAE']:.3f}\nTest R²={m['R2']:.3f}",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.15)
    )

    save_fig(fig, out_path)

def plot_permutation_importance(imp_df: pd.DataFrame, out_path: str, top_n: int = 10) -> None:
    set_plot_style()
    d = imp_df.sort_values("rmse_increase_mean", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(14, 10))
    norm = plt.Normalize(d["rmse_increase_mean"].min(), d["rmse_increase_mean"].max())
    colors = cm.viridis(norm(d["rmse_increase_mean"]))
    ax.barh(d["feature"], d["rmse_increase_mean"], color=colors, edgecolor='black')
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    save_fig(fig, out_path)

def plot_missingness(missing_pct: pd.Series, out_path: str, top_n: int = 20) -> None:
    set_plot_style()
    d = missing_pct.sort_values(ascending=False).head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh(d.index, d.values)
    ax.set_xlabel("Missingness (%)")
    ax.set_title(f"Top {top_n} missingness rates (post-BDL recoding)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    save_fig(fig, out_path)

def plot_model_comparison(perf: pd.DataFrame, out_path: str) -> None:
    set_plot_style()

    d = perf[["Model", "RMSE"]].copy()

    # Optional: shorten long names for cleaner x labels
    d["Model"] = d["Model"].replace({
        "OLS (multiple linear regression)": "OLS",
        "PLS regression": "PLS"
    })

    # If your PLS label includes number of comps, shorten it too
    d["Model"] = d["Model"].str.replace(
        r"PLS regression",
        r"PLS",
        regex=True
    )

    d = d.sort_values("RMSE", ascending=True).reset_index(drop=True)
    x = np.arange(len(d))

    colors = cm.viridis(np.linspace(0.2, 0.85, len(d)))

    fig, ax = plt.subplots(figsize=(10.2, 5.6), constrained_layout=True)

    bars = ax.bar(
        x,
        d["RMSE"].values,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        width=0.72,
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(d["Model"], ha="center")
    ax.set_ylabel("Test RMSE")
    ax.set_xlabel("Model")

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = d["RMSE"].max()
    ax.set_ylim(0, ymax * 1.14)

    for bar, val in zip(bars, d["RMSE"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=16
        )

    save_fig(fig, out_path)

def plot_pls_tuning(n_comp: List[int], rmse_cv: List[float], out_path: str) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(n_comp, rmse_cv, marker="o")
    ax.set_xlabel("PLS components")
    ax.set_ylabel("CV RMSE")
    ax.set_title("PLS tuning on training set")
    ax.grid(True, linestyle="--", alpha=0.5)
    save_fig(fig, out_path)



# ---------------------------------------------
# Step 1: Load data + QA/QC summary statistics
# ---------------------------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

print("Shape:", df.shape)
print("Target missing fraction:", df[TARGET_COL].isna().mean())
print("Duplicate rows:", df.duplicated().sum())
print("\nGroup counts:")
for g in ["Location"]:
    if g in df.columns:
        print(f"\n{g}:")
        print(df[g].value_counts(dropna=False))


# ---------------------------------------------
# Step 2: Handle BDL tokens + missingness table
# ---------------------------------------------
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

BDL_COLS = [
    "TSS_eff (mg/L)",
    "Ammonia_eff (mg/L)",
    "Somatic_eff (PFU/ml)",
    "Fspecific_eff (PFU/ml)",
    "PMMoV_eff (cop/ml)",
    "ToBRFV_eff (cop/ml)",
    "CrA_eff (cop/ml)",
]

dfc = add_bdl_indicators_and_coerce_numeric(df, BDL_COLS)

missing_pct = (dfc.isna().mean() * 100).sort_values(ascending=False)
print("\nTop missingness (%):")
print(missing_pct.head(15))

bdl_rates = {c: 100 * dfc[c + "_isBDL"].mean() for c in BDL_COLS}
print("\nBDL rates (%):")
print(pd.Series(bdl_rates).sort_values(ascending=False))


# ---------------------------------------------------
# Step 3: Define feature sets (full vs influent-only)
# ---------------------------------------------------
def build_feature_matrix(
    dfc: pd.DataFrame,
    feature_set: str = "full"
) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """
    Returns X, y, groups (Location labels) for nested LOPO.

    IMPORTANT:
      - Location is *not* used as a predictor (dropped from X) to avoid leakage.
      - Location is returned as groups for LeaveOneGroupOut splitting.
    """
    y = dfc[TARGET_COL].copy()

    groups = None
    if "Location" in dfc.columns:
        groups = dfc["Location"].astype(str).values

    if feature_set == "full":
        X = dfc.drop(columns=[TARGET_COL, "Location"], errors="ignore").copy()

    elif feature_set == "influent_only":
        all_cols = dfc.drop(columns=[TARGET_COL], errors="ignore").columns.tolist()
        drop_cols = [c for c in all_cols if ("_eff" in c) or c.startswith("LR_") or c.endswith("_isBDL")]
        drop_cols = list(set(drop_cols + ["Location"]))  # ensure Location not used as feature
        X = dfc.drop(columns=[TARGET_COL] + drop_cols, errors="ignore").copy()

    else:
        raise ValueError("feature_set must be 'full' or 'influent_only'.")

    return X, y, groups

FEATURE_SET = "full"
X, y, groups = build_feature_matrix(dfc, feature_set=FEATURE_SET)

# ---------------------------------------------------
# Drop predictor columns with >20% missingness
# ---------------------------------------------------
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

# ---------------------------------------------------
# Complete-case analysis on retained predictors:
# no imputation; rows with any missing predictor/target are removed
# ---------------------------------------------------
row_mask = (~y.isna()) & (~X.isna().any(axis=1))

X = X.loc[row_mask].reset_index(drop=True)
y = y.loc[row_mask].reset_index(drop=True)

if groups is not None:
    groups = groups[row_mask.to_numpy()]

print("\nFeature set:", FEATURE_SET)
print("X shape after dropping high-missing columns and incomplete rows:", X.shape)
print("Using Location as groups only:", groups is not None)
print("Retained predictors:", list(X.columns))
print("Dropped high-missing predictors:", drop_cols_high_missing)
print("Rows retained for modeling:", len(X))


# -------------------------------------------------------------------
# Step 4: Leakage-safe preprocessing pipeline (inside CV only)
# -------------------------------------------------------------------
def make_preprocess(num_cols_local, cat_cols_local) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("scaler", StandardScaler()),  # swapped in GridSearch
            ]), num_cols_local),
            ("cat", Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols_local),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


# -----------------------------
# Output folders
# -----------------------------
input_path = os.path.abspath(INPUT_FILE)
base_dir = os.path.dirname(input_path)
out_dir = os.path.join(base_dir, "ML_Ent_Results")
fig_dir = os.path.join(out_dir, "figures")
tab_dir = os.path.join(out_dir, "tables")
_ensure_dir(out_dir)
_ensure_dir(fig_dir)
_ensure_dir(tab_dir)

# Save missingness plot
if SAVE_FIGURES:
    plot_missingness(missing_pct, os.path.join(fig_dir, "Fig_S0_missingness.png"), top_n=20)


# --------------------------------------------------------
# Step 5: GridSearch helper (accepts CV object + groups)
# --------------------------------------------------------
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
    groups_fit: Optional[np.ndarray] = None,
    num_cols_local=None,
    cat_cols_local=None,
) -> Tuple[GridSearchCV, Pipeline]:
    """
    Runs GridSearchCV where preprocessing hyperparameters (scaler + selector)
    are included in the grid.

    cv_obj can be KFold, GroupKFold, LeaveOneGroupOut, etc.
    If cv_obj needs groups, pass groups_fit.
    """
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

    # Build selector candidates safely based on encoded dimensionality on THIS training data
    pre_tmp = make_preprocess(num_cols_local, cat_cols_local)
    p_encoded = pre_tmp.fit(X_fit).transform(X_fit).shape[1]
    _k_candidates = [k for k in K_CANDIDATES_BASE if k <= p_encoded] + ["all"]
    selector_candidates = ["passthrough"] + [SelectKBest(score_func=f_regression, k=k) for k in _k_candidates]

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
        verbose=0
    )

    if groups_fit is None:
        grid.fit(X_fit, y_fit)
    else:
        grid.fit(X_fit, y_fit, groups=groups_fit)

    best_pipe = grid.best_estimator_
    print(f"\n[{model_name}] best CV RMSE = {-grid.best_score_:.4f}")
    print(f"[{model_name}] best params = {grid.best_params_}")
    return grid, best_pipe


# ------------------------------------------------------------
# Step 6A: INTERNAL split + model selection (unchanged idea)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# SFR reporting
pre_tmp = make_preprocess(num_cols, cat_cols)
p_raw = X.shape[1]
n_train = X_train.shape[0]
p_encoded_train = pre_tmp.fit(X_train).transform(X_train).shape[1]
print(f"SFR (raw predictors) = {n_train}/{p_raw} = {n_train/p_raw:.2f}")
print(f"SFR (after one-hot)  = {n_train}/{p_encoded_train} = {n_train/p_encoded_train:.2f}")

# Standard inner CV for INTERNAL selection
cv_internal = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

rf_grid, rf_best = grid_search_for_model(
    "RandomForest",
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
    RF_PARAM_GRID,
    X_fit=X_train,
    y_fit=y_train,
    cv_obj=cv_internal
)

et_grid, et_best = grid_search_for_model(
    "ExtraTrees",
    ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
    ET_PARAM_GRID,
    X_fit=X_train,
    y_fit=y_train,
    cv_obj=cv_internal
)

ridge_grid, ridge_best = grid_search_for_model(
    "Ridge",
    Ridge(),
    RIDGE_PARAM_GRID,
    X_fit=X_train,
    y_fit=y_train,
    cv_obj=cv_internal
)

candidates = {
    "RandomForest": (rf_grid, rf_best, -rf_grid.best_score_),
    "ExtraTrees": (et_grid, et_best, -et_grid.best_score_),
    "Ridge": (ridge_grid, ridge_best, -ridge_grid.best_score_),
}
best_name = min(candidates, key=lambda k: candidates[k][2])
best_grid, best_model, best_cv_rmse = candidates[best_name]
print(f"\nSelected best ML model (internal CV): {best_name} (CV RMSE={best_cv_rmse:.4f})")


# -----------------------------------------------------------
# Step 6B: INTERNAL held-out test evaluation (+ figure)
# -----------------------------------------------------------
def evaluate_on_test(model_pipe: Pipeline, X_train, y_train, X_test, y_test):
    model_pipe.fit(X_train, y_train)
    yhat_train = model_pipe.predict(X_train)
    yhat_test = model_pipe.predict(X_test)
    m_test = regression_metrics(y_test, yhat_test)
    return m_test, yhat_train, yhat_test

perf_rows = []
ml_models = {
    "Random Forest": rf_best,
    "Extra Trees": et_best,
    "Ridge": ridge_best,
}

pred_cache = {}
for label, mdl in ml_models.items():
    m_test, yhat_tr, yhat_te = evaluate_on_test(mdl, X_train, y_train, X_test, y_test)
    perf_rows.append({"Model": label, **m_test})
    pred_cache[label] = (yhat_tr, yhat_te)

yhat_mean = np.repeat(float(y_train.mean()), len(y_test))
m_base = regression_metrics(y_test, yhat_mean)
# perf_rows.append({"Model": "Baseline", **m_base})

perf_df = pd.DataFrame(perf_rows).sort_values("RMSE")
perf_df.to_csv(os.path.join(tab_dir, "Table_S1_model_performance_internal_split.csv"), index=False)
print("\nINTERNAL held-out test set performance:")
print(perf_df)


best_model.fit(X_train, y_train)
yhat_train_best = best_model.predict(X_train)
yhat_test_best = best_model.predict(X_test)

if SAVE_FIGURES:
    plot_actual_vs_pred(
        y_train.values, yhat_train_best,
        y_test.values, yhat_test_best,
        os.path.join(fig_dir, "Fig_S1_actual_vs_pred_INTERNAL.png"),
    )


# -----------------------------------------------------
# Step 7: Statistical models for comparison (OLS + PLS)
# -----------------------------------------------------
feat_pipe = Pipeline([
    ("preprocess", best_model.named_steps["preprocess"]),
    ("var", best_model.named_steps["var"]),
    ("select", best_model.named_steps["select"]),
])

Xtr = feat_pipe.fit_transform(X_train, y_train)
Xte = feat_pipe.transform(X_test)

ols = sm.OLS(y_train.values, sm.add_constant(Xtr, has_constant="add")).fit()
pred_ols = ols.predict(sm.add_constant(Xte, has_constant="add"))
m_ols = regression_metrics(y_test, pred_ols)

max_comp = int(min(15, Xtr.shape[1], Xtr.shape[0] - 1))
cand_comp = list(range(2, max_comp + 1))
rmse_cv = []
for nc in cand_comp:
    pls = PLSRegression(n_components=nc)
    scores = cross_val_score(pls, Xtr, y_train, cv=cv_internal, scoring="neg_root_mean_squared_error")
    rmse_cv.append(float(-scores.mean()))

best_nc = cand_comp[int(np.argmin(rmse_cv))]
pls_best = PLSRegression(n_components=best_nc).fit(Xtr, y_train)
pred_pls = pls_best.predict(Xte).ravel()
m_pls = regression_metrics(y_test, pred_pls)

perf_df2 = pd.concat([
    perf_df,
    pd.DataFrame([
        {"Model": "OLS (multiple linear regression)", **m_ols},
        {"Model": f"PLS regression ({best_nc} comps)", **m_pls},
    ])
], ignore_index=True).sort_values("RMSE")

perf_df2.to_csv(os.path.join(tab_dir, "Table_S1_model_performance_INTERNAL_plus_stats.csv"), index=False)
print("\nINTERNAL model comparison incl. statistical baselines:")
print(perf_df2)

if SAVE_FIGURES:
    plot_pls_tuning(cand_comp, rmse_cv, os.path.join(fig_dir, "Fig_S2_pls_tuning_INTERNAL.png"))

if SAVE_FIGURES:
    plot_model_comparison(
        perf_df2,
        os.path.join(fig_dir, "Fig_S3_model_comparison_rmse_INTERNAL_plus_stats.png")
    )

# ----------------------------------------------
# Step 8: Explainability: permutation importance
# ----------------------------------------------
perm = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=PERM_N_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_samples=PERM_MAX_SAMPLES
)

imp = pd.DataFrame({
    "feature": X_test.columns,
    "rmse_increase_mean": perm.importances_mean,
    "rmse_increase_std": perm.importances_std
}).sort_values("rmse_increase_mean", ascending=False)

imp.to_csv(os.path.join(tab_dir, "Table_S2_permutation_importance_INTERNAL.csv"), index=False)
print("\nTop permutation importances (INTERNAL test set):")
print(imp.head(15))

if SAVE_FIGURES:
    plot_permutation_importance(
        imp,
        os.path.join(fig_dir, "Fig_S4_permutation_importance_INTERNAL.png"),
        top_n=15
    )


# -----------------------------------------
# Step 9: Robustness to RF/ET randomness
# -----------------------------------------
robust_rows = []
for seed in RF_SEEDS:
    if best_name not in ["RandomForest", "ExtraTrees"]:
        break

    mdl = clone(best_model)
    try:
        mdl.named_steps["model"].set_params(random_state=seed)
    except Exception:
        pass

    mdl.fit(X_train, y_train)
    yhat = mdl.predict(X_test)
    m = regression_metrics(y_test, yhat)
    robust_rows.append({"Seed": seed, **m})

if robust_rows:
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(os.path.join(tab_dir, "Table_S3_seed_robustness_INTERNAL.csv"), index=False)
    print("\nSeed robustness (INTERNAL split):")
    print(robust_df)
    print("\nSeed robustness summary:")
    print(robust_df[["RMSE", "MAE", "R2"]].agg(["mean", "std"]))




# -------------------------
# Step 10: Save run summary
# -------------------------
def _jsonify_param(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return repr(v)

best_params_json = {k: _jsonify_param(v) for k, v in best_grid.best_params_.items()}

run_summary = {
    "input_file": INPUT_FILE,
    "sheet_name": SHEET_NAME,
    "target_col": TARGET_COL,
    "feature_set": FEATURE_SET,
    "internal_split": {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv_folds": int(CV_FOLDS),
        "best_model_name": best_name,
        "best_cv_rmse": float(best_cv_rmse),
        "best_params": best_params_json,
    },
    "outputs": {
        "out_dir": out_dir,
        "figures_dir": fig_dir,
        "tables_dir": tab_dir,
    },
}

with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(run_summary, f, indent=2)

print(f"\nSaved outputs to: {out_dir}")
print("Run summary written to run_summary.json")