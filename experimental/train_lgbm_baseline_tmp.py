#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM baseline training script for M5-style data.

Usage:
    python train_lgbm_baseline.py

What it does:
- Loads selected store files from `newdata_evaluation/processed_*.csv`.
- Splits by time: train d_1每1913, valid d_1914每1941.
- Fits categorical encoders on train only, then trains LightGBM.
- Reports RMSE/MAPE on the validation split.

Adjustable knobs:
- STORES: which store files to use (start small, then add more).
- DATA_DIR: use newfinaldata for quick dry runs (only d_1每1913, no val).
- LIGHTGBM_PARAMS: tweak leaves/learning_rate, etc.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import json

from wrmsse_official import WRMSSEEvaluator  

# ??/??? WRMSSE evaluation ID ? validation ?????
WRMSSE_SALES_FILE = Path("data/sales_train_validation.csv")
WRMSSE_ENABLED = False
try:
    _WRMSSE_OFFICIAL = WRMSSEEvaluator(sales_file=WRMSSE_SALES_FILE) if WRMSSE_ENABLED else None
except Exception as e:
    print(f"Warning: WRMSSEEvaluator import/init failed ({e}); wrmsse will raise if called.")
    _WRMSSE_OFFICIAL = None

# Force disable WRMSSE during tuning
WRMSSE_ENABLED = False
_WRMSSE_OFFICIAL = None

WEIGHT_DIR = Path("weight")
WEIGHT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_PATH = WEIGHT_DIR / "summary_auto.json"
BLEND_WEIGHT_PATH = WEIGHT_DIR / "blend_weights_auto.json"


STATE_GROUPS = {
    "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
    "TX": ["TX_1", "TX_2", "TX_3"],
    "WI": ["WI_1", "WI_2", "WI_3"],
}

LEVEL_MODE = "store"

# All stores
STORE_LIST: List[str] = [s for stores in STATE_GROUPS.values() for s in stores]

# Stores that should force random search even if STORE_PARAMS exists
# Empty: use fixed per-store params for all stores.
SEARCH_STORES = set(STORE_LIST) - {"CA_3", "WI_2"}  # ????

# Store-specific tuned params (will override BASE_PARAMS when present)
STORE_PARAMS = {
    "CA_1": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.03291170222996989,
        "num_leaves": 383,
        "max_depth": -1,
        "feature_fraction": 0.7987286816083341,
        "bagging_fraction": 0.8443754341594838,
        "bagging_freq": 5,
        "min_data_in_leaf": 120,
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
        "n_estimators": 2000,
        "max_bin": 511,
    },
    "CA_2": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.021571100632025438,
        "num_leaves": 255,
        "max_depth": -1,
        "feature_fraction": 0.8836245222229365,
        "bagging_fraction": 0.8328070668543215,
        "bagging_freq": 3,
        "min_data_in_leaf": 120,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
        "n_estimators": 1800,
        "max_bin": 511,
    },
    "CA_3": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.019738923414965245,
        "num_leaves": 319,
        "max_depth": 10,
        "feature_fraction": 0.8714576239732946,
        "bagging_fraction": 0.8080588969530361,
        "bagging_freq": 3,
        "min_data_in_leaf": 150,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "n_estimators": 2000,
        "max_bin": 511,
    },
    "CA_4": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.02654910862466569,
        "num_leaves": 383,
        "max_depth": 10,
        "feature_fraction": 0.9834467544005815,
        "bagging_fraction": 0.7906891653855229,
        "bagging_freq": 5,
        "min_data_in_leaf": 180,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "n_estimators": 2200,
        "max_bin": 511,
    },
    "TX_1": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.030801474444336212,
        "num_leaves": 319,
        "max_depth": -1,
        "feature_fraction": 0.8342369743978673,
        "bagging_fraction": 0.9899973308685395,
        "bagging_freq": 7,
        "min_data_in_leaf": 180,
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
        "n_estimators": 1800,
        "max_bin": 511,
    },
    "TX_2": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.02620774853906023,
        "num_leaves": 383,
        "max_depth": -1,
        "feature_fraction": 0.8002060361355822,
        "bagging_fraction": 0.8746486273764073,
        "bagging_freq": 5,
        "min_data_in_leaf": 150,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "n_estimators": 1800,
        "max_bin": 511,
    },
    "TX_3": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.01861087438340826,
        "num_leaves": 255,
        "max_depth": -1,
        "feature_fraction": 0.8177510797053789,
        "bagging_fraction": 0.7873106339756981,
        "bagging_freq": 3,
        "min_data_in_leaf": 120,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "n_estimators": 1800,
        "max_bin": 511,
    },
    "WI_1": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.02814410986626996,
        "num_leaves": 383,
        "max_depth": -1,
        "feature_fraction": 0.9762849449709258,
        "bagging_fraction": 0.7900643911832694,
        "bagging_freq": 7,
        "min_data_in_leaf": 180,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
        "n_estimators": 2200,
        "max_bin": 511,
    },
    "WI_2": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.02797710656495476,
        "num_leaves": 319,
        "max_depth": -1,
        "feature_fraction": 0.8225640975142336,
        "bagging_fraction": 0.7261280295918504,
        "bagging_freq": 3,
        "min_data_in_leaf": 150,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "n_estimators": 2000,
        "max_bin": 511,
    },
    "WI_3": {
        "objective": "regression",
        "metric": ["rmse", "mape"],
        "learning_rate": 0.03125218432826508,
        "num_leaves": 383,
        "max_depth": -1,
        "feature_fraction": 0.8555882063960265,
        "bagging_fraction": 0.8892247633789941,
        "bagging_freq": 5,
        "min_data_in_leaf": 120,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
        "n_estimators": 2200,
        "max_bin": 511,
    },
}

# For dept/item-level training, optionally restrict to a subset (empty list => all)
DEPT_FILTER: List[str] = []  # e.g., ["FOODS_1", "HOUSEHOLD_1"]
ITEM_FILTER: List[str] = []  # e.g., ["FOODS_1_001"]

# Directory containing processed CSVs.
# Use "newdata_evaluation" for full 1941d history; "newfinaldata" only has d_1?913.
DATA_DIR = Path("newfinaldata")

# Time split (matching M5 validation scheme)
TRAIN_END = 1913  # inclusive
VAL_END = 1941  # inclusive

# Feature lists
NUM_SCALED = [
    "sell_price_scaled",
    "baseline_price_scaled",
    "discount_scaled",
    "promo_intensity_scaled",
    "lag_7_scaled",
    "lag_30_scaled",
    "rolling_mean_7_scaled",
    "rolling_mean_30_scaled",
    "rolling_std_7_scaled",
    "rolling_std_30_scaled",
    "lag_1_scaled",
    "lag_14_scaled",
    "lag_28_scaled",
    "lag_56_scaled",
    "lag_84_scaled",
    "rolling_mean_14_scaled",
    "rolling_mean_28_scaled",
    "rolling_mean_56_scaled",
    "rolling_mean_84_scaled",
    "rolling_std_14_scaled",
    "rolling_std_28_scaled",
    "rolling_std_56_scaled",
    "rolling_std_84_scaled",
    "price_ratio_scaled",
    "discount_pct_scaled",
    "snap_wday_scaled",
    "promo_holiday_scaled",
    "promo_wday_sin_scaled",
    "promo_wday_cos_scaled",
    "discount_snap_scaled",
    # rolling median/min/max (scaled)
    "rolling_median_7_scaled",
    "rolling_median_14_scaled",
    "rolling_median_28_scaled",
    "rolling_median_30_scaled",
    "rolling_median_56_scaled",
    "rolling_median_84_scaled",
    "rolling_min_7_scaled",
    "rolling_min_14_scaled",
    "rolling_min_28_scaled",
    "rolling_min_30_scaled",
    "rolling_min_56_scaled",
    "rolling_min_84_scaled",
    "rolling_max_7_scaled",
    "rolling_max_14_scaled",
    "rolling_max_28_scaled",
    "rolling_max_30_scaled",
    "rolling_max_56_scaled",
    "rolling_max_84_scaled",
    # price momentum / z-score (scaled)
    "sell_price_week_chg_scaled",
    "sell_price_month_chg_scaled",
    "sell_price_z28_scaled",
    "discount_week_chg_scaled",
    "discount_month_chg_scaled",
    "discount_z28_scaled",
    # streaks / holiday distance (scaled)
    "promo_streak_scaled",
    "holiday_streak_scaled",
    "days_since_holiday_scaled",
    "days_until_holiday_scaled",
]

CYCLIC = ["wday_sin", "wday_cos", "month_sin", "month_cos", "quarter_sin", "quarter_cos"]

BIN_COLS = ["snap_CA", "snap_TX", "snap_WI", "IsHoliday", "IsPromotion"]

CAT_COLS = [
    "state_id",
    "store_id",
    "cat_id",
    "dept_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "item_id",
    "id",
]

TARGET_COL = "sales"

# LightGBM hyperparameters (tuned baseline)
BASE_PARAMS = dict(
    objective="regression",
    metric=["rmse", "mape"],
    learning_rate=0.03,
    num_leaves=319,
    max_depth=-1,
    feature_fraction=0.8038614760496599,
    bagging_fraction=0.7567022188313345,
    bagging_freq=5,
    min_data_in_leaf=120,
    lambda_l1=0.1,
    lambda_l2=1.0,
    n_estimators=1800,  # allow more trees; early stopping will pick best_iter
    max_bin=511,  # higher binning for high-cardinality categories
)

# Random search config
DO_RANDOM_SEARCH = True  # enable_search
N_TRIALS = 10
EARLY_STOPPING = 100  # more patient on full data

# Optional simple blending (primary + one or two tweaked variants)
ENABLE_BLEND = False  # ?糸???百????
BLEND_WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]  # allow pure base or alt
NO_BLEND_STORES = {"TX_1", "WI_3"}  # skip blending for these stores

# Time series CV config (per-store)
USE_CV = False  # global default
CV_STORES = {"CA_3", "WI_2"}  # stores to force CV even if USE_CV=False
CV_VAL_LEN = 28
CV_FOLDS = 2

# Target transform (log1p helps long-tail); if False uses raw sales
USE_LOG1P_TARGET = False

# Time-decay weighting for recent days (applied to training only)
TIME_DECAY = True  # set True to enable
DECAY_HALF_LIFE = 90  # days; smaller => heavier recent weighting

# Incremental summary log (jsonl) to avoid losing progress if a later task crashes
SUMMARY_PATH = Path("logs/summary_lgbm.jsonl")


# ----------------------
# Helpers
# ----------------------
def read_store(store: str, data_dir: Path, usecols: List[str]) -> pd.DataFrame:
    path = data_dir / f"processed_{store}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, usecols=usecols, engine="pyarrow")
    df["d_int"] = df["d"].str.replace("d_", "", regex=False).astype(int)
    return df


def set_categorical(train: pd.Series, val: pd.Series) -> tuple[pd.Series, pd.Series]:
    cats = pd.CategoricalDtype(categories=train.dropna().unique())
    return train.astype(cats), val.astype(cats)


def build_datasets(
    stores: List[str],
    dept_filter: List[str] | None = None,
    item_filter: List[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Columns that exist in CSV
    feature_cols = NUM_SCALED + CYCLIC + BIN_COLS + CAT_COLS
    generated_cols = [
        "item_freq",
        "id_freq",
        "item_freq_scaled",
        "id_freq_scaled",
        "item_te",
        "id_te",
        "item_te_scaled",
        "id_te_scaled",
    ]
    usecols = [c for c in feature_cols + [TARGET_COL, "d"] if c not in generated_cols]

    dfs = [read_store(store, DATA_DIR, usecols) for store in stores]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # Optional filters
    if dept_filter:
        df = df[df["dept_id"].isin(dept_filter)]
    if item_filter:
        df = df[df["item_id"].isin(item_filter)]

    # Split
    train_df = df[df["d_int"] <= TRAIN_END].copy()
    val_df = df[(df["d_int"] > TRAIN_END) & (df["d_int"] <= VAL_END)].copy()

    if val_df.empty:
        raise ValueError("Validation set is empty. Use newdata_evaluation or adjust VAL_END.")

    # Categorical encoding: fit on train, transform val
    for col in CAT_COLS:
        train_df[col], val_df[col] = set_categorical(train_df[col].astype(str), val_df[col].astype(str))

    # Frequency encoding for high-cardinality item_id/id (fit on train only)
    item_counts = train_df["item_id"].value_counts()
    id_counts = train_df["id"].value_counts()
    train_df["item_freq"] = train_df["item_id"].map(item_counts).fillna(0).astype("float32")
    val_df["item_freq"] = val_df["item_id"].map(item_counts).fillna(0).astype("float32")
    train_df["id_freq"] = train_df["id"].map(id_counts).fillna(0).astype("float32")
    val_df["id_freq"] = val_df["id"].map(id_counts).fillna(0).astype("float32")
    # Scale freq by train min-max
    for col in ["item_freq", "id_freq"]:
        mn = train_df[col].min()
        mx = train_df[col].max()
        rng = mx - mn
        if rng == 0:
            train_df[col + "_scaled"] = 0.0
            val_df[col + "_scaled"] = 0.0
        else:
            train_df[col + "_scaled"] = (train_df[col] - mn) / rng
            val_df[col + "_scaled"] = (val_df[col] - mn) / rng

    # Target encoding (mean encoded on train only)
    global_mean = train_df[TARGET_COL].mean()

    def mean_encode(series: pd.Series, target: pd.Series, alpha: float = 5.0) -> pd.Series:
        counts = series.value_counts()
        means = train_df.groupby(series.name)[TARGET_COL].mean()
        # smoothing: (count * mean + alpha * global_mean) / (count + alpha)
        smoothing = (means * counts + alpha * global_mean) / (counts + alpha)
        return series.map(smoothing)

    train_df["item_te"] = mean_encode(train_df["item_id"], train_df[TARGET_COL])
    val_df["item_te"] = val_df["item_id"].map(train_df.groupby("item_id")[TARGET_COL].mean())
    val_df["item_te"] = val_df["item_te"].fillna(global_mean)

    train_df["id_te"] = mean_encode(train_df["id"], train_df[TARGET_COL])
    val_df["id_te"] = val_df["id"].map(train_df.groupby("id")[TARGET_COL].mean())
    val_df["id_te"] = val_df["id_te"].fillna(global_mean)

    for col in ["item_te", "id_te"]:
        mn = train_df[col].min()
        mx = train_df[col].max()
        rng = mx - mn
        if rng == 0:
            train_df[col + "_scaled"] = 0.0
            val_df[col + "_scaled"] = 0.0
        else:
            train_df[col + "_scaled"] = (train_df[col] - mn) / rng
            val_df[col + "_scaled"] = (val_df[col] - mn) / rng

    # Extend feature list with generated cols
    feature_cols = feature_cols + [
        "item_freq_scaled",
        "id_freq_scaled",
        "item_te_scaled",
        "id_te_scaled",
    ]

    # Binary to int8 to save memory
    for col in BIN_COLS:
        train_df[col] = train_df[col].astype("int8")
        val_df[col] = val_df[col].astype("int8")

    # Floats to float32
    for col in NUM_SCALED + CYCLIC:
        train_df[col] = train_df[col].astype("float32")
        val_df[col] = val_df[col].astype("float32")

    return train_df, val_df, feature_cols


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, min_denom: float = 1e-3) -> float:
    """MAPE that ignores targets near zero to avoid exploding values."""
    mask = np.abs(y_true) > min_denom
    if not mask.any():
        return float("nan")
    denom = np.abs(y_true[mask])
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(np.mean(2.0 * num / denom))



def wrmsse(train_df: pd.DataFrame, val_df: pd.DataFrame, preds: np.ndarray, id_col: str = "id") -> float:
    """WRMSSE disabled during tuning."""
    return 0.0



def train_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], params: dict) -> tuple[lgb.LGBMRegressor, dict]:
    X_train = train_df[feature_cols]
    y_train_raw = train_df[TARGET_COL].astype("float32")
    X_val = val_df[feature_cols]
    y_val_raw = val_df[TARGET_COL].astype("float32")

    sample_weight = None
    if TIME_DECAY:
        # weight = exp(-ln(2) * age / half_life); recent days get higher weight
        age = TRAIN_END - train_df["d_int"]
        sample_weight = np.exp(-np.log(2) * age / DECAY_HALF_LIFE).astype("float32")

    if USE_LOG1P_TARGET:
        y_train = np.log1p(y_train_raw)
        y_val = np.log1p(y_val_raw)
    else:
        y_train = y_train_raw
        y_val = y_val_raw

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=50)],
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    if USE_LOG1P_TARGET:
        preds = np.expm1(preds)
        y_val_used = y_val_raw.values
    else:
        y_val_used = y_val.values

    # Some sklearn versions lack squared=False; compute RMSE manually for compatibility.
    rmse = float(np.sqrt(mean_squared_error(y_val_used, preds)))
    mape = safe_mape(y_val_used, preds)
    smape_val = smape(y_val_used, preds)
    wrmsse_val = wrmsse(train_df, val_df, preds)
    metrics = {"rmse": rmse, "mape": mape, "smape": smape_val, "wrmsse": wrmsse_val, "iter": model.best_iteration_}
    print(
        f"Validation RMSE: {rmse:.4f} | MAPE (>0.001 mask): {mape:.4f} | SMAPE: {smape_val:.4f} | WRMSSE: {wrmsse_val:.4f}"
    )
    return model, metrics


def time_series_folds(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create simple time-series folds: fold1 uses train<=TRAIN_END-CV_VAL_LEN, val=last CV_VAL_LEN before TRAIN_END; fold2 uses standard val."""
    folds = []
    # fold 1
    val1_start = TRAIN_END - CV_VAL_LEN + 1
    val1_end = TRAIN_END
    train1 = df[df["d_int"] < val1_start]
    val1 = df[(df["d_int"] >= val1_start) & (df["d_int"] <= val1_end)]
    if not val1.empty and len(train1) > 0:
        folds.append((train1, val1))
    # fold 2 (current standard val)
    train2 = df[df["d_int"] <= TRAIN_END]
    val2 = df[(df["d_int"] > TRAIN_END) & (df["d_int"] <= VAL_END)]
    if not val2.empty and len(train2) > 0:
        folds.append((train2, val2))
    return folds


def cv_evaluate(df: pd.DataFrame, feature_cols: List[str], params: dict) -> dict:
    folds = time_series_folds(df)
    metrics_list = []
    for i, (tr, va) in enumerate(folds, 1):
        print(f"  CV fold {i}: train={len(tr):,}, val={len(va):,}")
        model, metrics = train_lgbm(tr, va, feature_cols, params)
        metrics_list.append(metrics)
    # average metrics
    avg = {k: float(np.nanmean([m[k] for m in metrics_list])) for k in ["rmse", "mape", "smape", "wrmsse"]}
    avg["iter"] = int(np.nanmean([m["iter"] for m in metrics_list]))
    return avg


def tweak_params_for_blend(params: dict) -> dict:
    """Create a slightly different param set for blending: lower lr, more trees."""
    p = params.copy()
    p["learning_rate"] = max(1e-4, params.get("learning_rate", 0.03) * 0.8)
    p["n_estimators"] = int(params.get("n_estimators", 1800) * 1.3)
    return p


def tweak_params_for_blend2(params: dict) -> dict:
    """Another variant: slightly higher lr, smaller leaves, higher feature fraction."""
    p = params.copy()
    p["learning_rate"] = min(0.1, params.get("learning_rate", 0.03) * 1.1)
    p["num_leaves"] = max(31, int(params.get("num_leaves", 255) * 0.8))
    p["n_estimators"] = int(params.get("n_estimators", 1800) * 1.1)
    p["feature_fraction"] = min(1.0, params.get("feature_fraction", 0.8) * 1.05)
    p["bagging_fraction"] = min(1.0, params.get("bagging_fraction", 0.8) * 1.05)
    return p


def random_params(base: dict) -> dict:
    """Sample around a base param set to keep search local."""

    def jitter(value: float, low_ratio=0.8, high_ratio=1.25):
        return max(1e-4, value * random.uniform(low_ratio, high_ratio))

    def pick_near(value: int, options: list[int]):
        # choose nearest few options around value
        sorted_opts = sorted(options, key=lambda x: abs(x - value))
        return random.choice(sorted_opts[:3])

    p = base.copy()
    p.update(
        num_leaves=pick_near(base.get("num_leaves", 255), [255, 319, 383]),
        learning_rate=jitter(base.get("learning_rate", 0.03), 0.6, 0.9),
        feature_fraction=min(1.0, max(0.6, jitter(base.get("feature_fraction", 0.85), 0.95, 1.1))),
        bagging_fraction=min(1.0, max(0.6, jitter(base.get("bagging_fraction", 0.8), 0.95, 1.1))),
        bagging_freq=random.choice([3, 5, 7]),
        min_data_in_leaf=pick_near(base.get("min_data_in_leaf", 120), [120, 150, 180]),
        lambda_l1=random.choice([0.1, 0.5, 1.0]),
        lambda_l2=random.choice([0.5, 1.0, 2.0]),
        max_depth=random.choice([-1, 8, 10]),
        n_estimators=pick_near(base.get("n_estimators", 1800), [1800, 2000, 2200]),
    )
    return p


def main() -> None:
    if LEVEL_MODE == "state":
        tasks = [(f"state {state}", stores, None, None, False, None) for state, stores in STATE_GROUPS.items()]
    elif LEVEL_MODE == "store":
        tasks = []
        for store in STORE_LIST:
            override = None if store in SEARCH_STORES else STORE_PARAMS.get(store)
            tasks.append((f"store {store}", [store], None, None, True, override))
    elif LEVEL_MODE == "dept":
        tasks = [("dept_level", STORE_LIST, DEPT_FILTER or None, None, False, None)]
    elif LEVEL_MODE == "item":
        tasks = [("item_level", STORE_LIST, None, ITEM_FILTER or None, False, None)]
    else:
        raise ValueError(f"Unknown LEVEL_MODE={LEVEL_MODE}")

    summary = []

    for label, stores, dept_filter, item_filter, enable_search, override_params in tasks:
        print(f"\n===== Training {label} | stores: {stores} from {DATA_DIR} =====")
        train_df, val_df, feature_cols = build_datasets(stores, dept_filter=dept_filter, item_filter=item_filter)
        print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")
        print(f"Features: {len(feature_cols)}")

        do_search = (DO_RANDOM_SEARCH or enable_search) and override_params is None
        if do_search:
            best = None
            base_for_search = override_params or STORE_PARAMS.get(stores[0], BASE_PARAMS)
            for i in range(1, min(N_TRIALS, 10) + 1):
                params = random_params(base_for_search)
                print(f"\n=== Trial {i}/{N_TRIALS} params: {params}")
                model, metrics = train_lgbm(train_df, val_df, feature_cols, params)
                score = metrics["rmse"]
                if (best is None) or (score < best["rmse"]):
                    best = {
                        "rmse": score,
                        "mape": metrics["mape"],
                        "smape": metrics["smape"],
                        "params": params,
                        "iter": model.best_iteration_,
                    }
            print("\nBest trial:")
            print(best)
            summary.append({"label": label, "stores": stores, **best})
        else:
            params = override_params or BASE_PARAMS
            use_cv_here = USE_CV or (stores[0] in CV_STORES)
            if use_cv_here:
                merged_df = pd.concat([train_df, val_df], ignore_index=True)
                metrics = cv_evaluate(merged_df, feature_cols, params)
                print(
                    f"CV avg -> RMSE {metrics['rmse']:.4f} | MAPE {metrics['mape']:.4f} | SMAPE {metrics['smape']:.4f} | WRMSSE {metrics['wrmsse']:.4f}"
                )
                result = {
                    "label": label,
                    "stores": stores,
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "smape": metrics["smape"],
                    "wrmsse": metrics["wrmsse"],
                    "iter": metrics["iter"],
                    "params": params,
                }
                model = None
            else:
                model, metrics = train_lgbm(train_df, val_df, feature_cols, params)
                print(f"Best iteration: {model.best_iteration_}")
                result = {
                    "label": label,
                    "stores": stores,
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "smape": metrics["smape"],
                    "wrmsse": metrics["wrmsse"],
                    "iter": metrics["iter"],
                    "params": params,
                }
            # Optional blending with tweaked params (skip if store in NO_BLEND_STORES)
            if ENABLE_BLEND and (stores[0] not in NO_BLEND_STORES) and not use_cv_here:
                alt_params = tweak_params_for_blend(params)
                alt_model, alt_metrics = train_lgbm(train_df, val_df, feature_cols, alt_params)
                alt_params2 = tweak_params_for_blend2(params)
                alt_model2, alt_metrics2 = train_lgbm(train_df, val_df, feature_cols, alt_params2)
                y_val = val_df[TARGET_COL].values.astype("float32")
                preds_base = model.predict(val_df[feature_cols], num_iteration=model.best_iteration_)
                preds_alt = alt_model.predict(val_df[feature_cols], num_iteration=alt_model.best_iteration_)
                preds_alt2 = alt_model2.predict(val_df[feature_cols], num_iteration=alt_model2.best_iteration_)
                best_blend = None
                # two-model search
                for w in BLEND_WEIGHTS:
                    blended = w * preds_base + (1 - w) * preds_alt
                    rmse_b = float(np.sqrt(mean_squared_error(y_val, blended)))
                    mape_b = safe_mape(y_val, blended)
                    smape_b = smape(y_val, blended)
                    wrmsse_b = wrmsse(train_df, val_df, blended)
                    if (best_blend is None) or (rmse_b < best_blend["rmse"]):
                        best_blend = {"type": "2model", "w_base": w, "w_alt": 1 - w, "rmse": rmse_b, "mape": mape_b, "smape": smape_b, "wrmsse": wrmsse_b}
                # three-model search (coarse grid ensuring weights sum to 1 and non-negative)
                for wb in [0.2, 0.4, 0.6]:
                    for wa in [0.2, 0.4, 0.6]:
                        wc = 1.0 - wb - wa
                        if wc < 0 or wc > 0.8:
                            continue
                        blended = wb * preds_base + wa * preds_alt + wc * preds_alt2
                        rmse_b = float(np.sqrt(mean_squared_error(y_val, blended)))
                        mape_b = safe_mape(y_val, blended)
                        smape_b = smape(y_val, blended)
                        wrmsse_b = wrmsse(train_df, val_df, blended)
                        if (best_blend is None) or (rmse_b < best_blend["rmse"]):
                            best_blend = {"type": "3model", "w_base": wb, "w_alt": wa, "w_alt2": wc, "rmse": rmse_b, "mape": mape_b, "smape": smape_b, "wrmsse": wrmsse_b}
                print(
                    f"Blended ({best_blend.get('type')}) RMSE {best_blend['rmse']:.4f} | MAPE {best_blend['mape']:.4f} | SMAPE {best_blend['smape']:.4f} | WRMSSE {best_blend['wrmsse']:.4f} "
                    f"(w_base={best_blend.get('w_base')}, w_alt={best_blend.get('w_alt')}, w_alt2={best_blend.get('w_alt2')})"
                )
                result.update(
                    {
                        "blend_rmse": best_blend["rmse"],
                        "blend_mape": best_blend["mape"],
                        "blend_smape": best_blend["smape"],
                        "blend_wrmsse": best_blend["wrmsse"],
                        "blend_w_base": best_blend["w_base"],
                        "blend_w_alt": best_blend.get("w_alt"),
                        "blend_w_alt2": best_blend.get("w_alt2"),
                        "blend_type": best_blend.get("type"),
                        "alt_iter": alt_model.best_iteration_,
                        "alt2_iter": alt_model2.best_iteration_,
                    }
                )
            # Persist incremental result to disk to avoid reruns on later failures
            try:
                SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
                with SUMMARY_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False))
                    f.write("\n")
            except Exception as e:
                print(f"Warning: failed to write summary log ({e})")

            summary.append(result)

    print("\n===== Summary per task =====")
    for row in summary:
        print(row)

    # Save full summary and blend weights for later reuse
    try:
        with SUMMARY_PATH.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        blend_weights = {}
        for row in summary:
            store_key = row["stores"][0] if row.get("stores") else row.get("label")
            if row.get("blend_type"):
                blend_weights[store_key] = {
                    "type": row.get("blend_type"),
                    "w_base": row.get("blend_w_base", 1.0),
                    "w_alt": row.get("blend_w_alt", 0.0),
                    "w_alt2": row.get("blend_w_alt2", 0.0),
                }
            else:
                blend_weights[store_key] = {"type": "none", "w_base": 1.0, "w_alt": 0.0, "w_alt2": 0.0}
        with BLEND_WEIGHT_PATH.open("w", encoding="utf-8") as f:
            json.dump(blend_weights, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to {SUMMARY_PATH} and blend weights to {BLEND_WEIGHT_PATH}")
    except Exception as e:
        print(f"Warning: failed to save summary/weights ({e})")


if __name__ == "__main__":
    main()

