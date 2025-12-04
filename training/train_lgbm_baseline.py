#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM baseline training script for M5-style data with WRMSSE monitoring.

Usage:
    python train_lgbm_baseline.py

The workflow:
1. Reads per-store processed CSVs with rolling/lags.
2. Builds core (low-variance) versus extra feature sets.
3. Trains LightGBM with RMSE early stopping but WRMSSE callback every WRMSSE_EVAL_FREQ.
4. Optionally blends with tweaked models if WRMSSE improves.
5. Logs metrics/wrmsse history to weight_v2/summary_delay120_v2.json and blend weights.
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
from pathlib import Path
import argparse
from typing import Callable, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from wrmsse_official import WRMSSEEvaluator

DATA_DIR = Path("newfinaldata")  # switch to processed_v2/v3 as needed
TRAIN_END = 1913
VAL_END = 1941

WRMSSE_SALES_FILE = Path("data/sales_train_validation.csv")
WRMSSE_ENABLED = True
WRMSSE_EVAL_FREQ = 40
WRMSSE_PATIENCE = 4
WRMSSE_IMPROVE_TOL = 1e-5
try:
    _WRMSSE_OFFICIAL = WRMSSEEvaluator(sales_file=WRMSSE_SALES_FILE)
except Exception as e:
    print(f"Warning: WRMSSEEvaluator import/init failed ({e}); wrmsse will raise if called.")
    _WRMSSE_OFFICIAL = None

if _WRMSSE_OFFICIAL is None:
    WRMSSE_ENABLED = False

WEIGHT_DIR = Path("weight_v2")
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = WEIGHT_DIR / "summary_delay120_v2.json"
BLEND_WEIGHT_PATH = WEIGHT_DIR / "delay_120_weight_v2.json"
VISUALIZE_SCRIPT = Path("visualize_and_blend.py")
VISUALIZE_BASE = Path("future_finaldata/submission_with_val.csv")
VISUALIZE_ALT = Path("future_finaldata/submission_with_val_cmodel.csv")

STATE_GROUPS = {
    "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
    "TX": ["TX_1", "TX_2", "TX_3"],
    "WI": ["WI_1", "WI_2", "WI_3"],
}

STATE_PARAM_SOURCE = {
    "CA": "CA_1",
    "WI": "WI_1",
    "TX": "TX_1",
}

STORE_LIST: List[str] = [
    "CA_1",
    "CA_2",
    "CA_3",
    "CA_4",
    "WI_1",
    "WI_2",
    "WI_3",
    "TX_1",
    "TX_2",
    "TX_3",
]
SEARCH_STORES = {"CA_1", "TX_1"}
STORE_PARAMS = {
    "CA_1": {"learning_rate": 0.035, "num_leaves": 220, "min_data_in_leaf": 180},
    "CA_2": {"learning_rate": 0.035, "num_leaves": 220, "min_data_in_leaf": 180},
    "CA_3": {"learning_rate": 0.035, "num_leaves": 220, "min_data_in_leaf": 180},
    "CA_4": {"learning_rate": 0.035, "num_leaves": 220, "min_data_in_leaf": 180},
    "TX_1": {"learning_rate": 0.035, "num_leaves": 200, "min_data_in_leaf": 200},
    "TX_2": {"learning_rate": 0.035, "num_leaves": 200, "min_data_in_leaf": 200},
    "TX_3": {"learning_rate": 0.035, "num_leaves": 200, "min_data_in_leaf": 200},
    "WI_1": {"learning_rate": 0.036, "num_leaves": 200, "min_data_in_leaf": 220},
    "WI_2": {"learning_rate": 0.036, "num_leaves": 200, "min_data_in_leaf": 220},
    "WI_3": {"learning_rate": 0.036, "num_leaves": 200, "min_data_in_leaf": 220},
}
DEPT_FILTER: List[str] = []
ITEM_FILTER: List[str] = []

CORE_LAG_SCALED = ["lag_1_scaled", "lag_7_scaled", "lag_28_scaled"]
CORE_ROLL_SCALED = ["rolling_mean_7_scaled", "rolling_mean_14_scaled", "rolling_std_7_scaled"]
CORE_PRICE_SCALED = [
    "sell_price_scaled",
    "baseline_price_scaled",
    "discount_scaled",
    "promo_intensity_scaled",
    "price_ratio_scaled",
    "discount_pct_scaled",
]
CORE_PROMO_SCALED = [
    "snap_wday_scaled",
    "promo_holiday_scaled",
    "promo_wday_sin_scaled",
    "promo_wday_cos_scaled",
    "discount_snap_scaled",
]
CORE_GENERATED_SCALED = [
    "item_freq_scaled",
    "id_freq_scaled",
    "item_te_scaled",
    "id_te_scaled",
]
CORE_NUM_SCALED = CORE_LAG_SCALED + CORE_ROLL_SCALED + CORE_PRICE_SCALED + CORE_PROMO_SCALED

EXTRA_LAG_SCALED = ["lag_14_scaled", "lag_56_scaled", "lag_84_scaled"]
EXTRA_ROLL_MEAN = [
    "rolling_mean_7_scaled",
    "rolling_mean_14_scaled",
    "rolling_mean_28_scaled",
    "rolling_mean_30_scaled",
    "rolling_mean_56_scaled",
    "rolling_mean_84_scaled",
]
EXTRA_ROLL_STD = [
    "rolling_std_7_scaled",
    "rolling_std_14_scaled",
    "rolling_std_28_scaled",
    "rolling_std_30_scaled",
    "rolling_std_56_scaled",
    "rolling_std_84_scaled",
]
EXTRA_ROLL_MED = [
    "rolling_median_7_scaled",
    "rolling_median_14_scaled",
    "rolling_median_28_scaled",
    "rolling_median_30_scaled",
    "rolling_median_56_scaled",
    "rolling_median_84_scaled",
]
EXTRA_ROLL_MIN = [
    "rolling_min_7_scaled",
    "rolling_min_14_scaled",
    "rolling_min_28_scaled",
    "rolling_min_30_scaled",
    "rolling_min_56_scaled",
    "rolling_min_84_scaled",
]
EXTRA_ROLL_MAX = [
    "rolling_max_7_scaled",
    "rolling_max_14_scaled",
    "rolling_max_28_scaled",
    "rolling_max_30_scaled",
    "rolling_max_56_scaled",
    "rolling_max_84_scaled",
]
EXTRA_PRICE_MOMENTUM = [
    "sell_price_week_chg_scaled",
    "sell_price_month_chg_scaled",
    "sell_price_z28_scaled",
    "discount_week_chg_scaled",
    "discount_month_chg_scaled",
    "discount_z28_scaled",
]
EXTRA_SEASONAL = [
    "promo_streak_scaled",
    "holiday_streak_scaled",
    "days_since_holiday_scaled",
    "days_until_holiday_scaled",
]
SELECTED_EXTRAS = [
    "lag_14_scaled",
    "lag_56_scaled",
    "lag_84_scaled",
    "rolling_mean_7_scaled",
    "rolling_mean_14_scaled",
    "rolling_mean_28_scaled",
    "sell_price_week_chg_scaled",
    "sell_price_month_chg_scaled",
    "discount_week_chg_scaled",
    "promo_streak_scaled",
    "days_since_holiday_scaled",
]
EXTRA_NUM_SCALED = (
    EXTRA_LAG_SCALED
    + EXTRA_ROLL_MEAN
    + EXTRA_ROLL_STD
    + EXTRA_ROLL_MED
    + EXTRA_ROLL_MIN
    + EXTRA_ROLL_MAX
    + EXTRA_PRICE_MOMENTUM
    + EXTRA_SEASONAL
)
NUM_SCALED = CORE_NUM_SCALED + EXTRA_NUM_SCALED

CYCLIC = [
    "wday_sin",
    "wday_cos",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
]
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

BASE_PARAMS = dict(
    objective="regression",
    metric=["rmse", "mape"],
    learning_rate=0.04,
    num_leaves=255,
    max_depth=10,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    min_data_in_leaf=240,
    lambda_l1=0.5,
    lambda_l2=1.0,
    n_estimators=2400,
    max_bin=511,
)
DO_RANDOM_SEARCH = False
N_TRIALS = 10
EARLY_STOPPING = 50
ENABLE_BLEND = False
BLEND_WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
NO_BLEND_STORES = set()
USE_CV = False
CV_STORES = {"CA_3", "WI_2"}
CV_VAL_LEN = 28
CV_FOLDS = 2
USE_LOG1P_TARGET = False
TIME_DECAY = True
DECAY_HALF_LIFE = 180
GROUP_DEFINITIONS = {
    "group_a": ["CA_1", "CA_2", "CA_4", "TX_1"],
    "group_b": ["CA_3", "WI_1", "WI_2", "TX_3"],
}
CANDIDATE_DIR = Path("weight_v2")
CANDIDATE_PATTERN = CANDIDATE_DIR / "group_params_{group}_{model_type}.json"
GROUP_CANDIDATE_CACHE: dict[str, dict[str, list[dict]]] = {}

CHUNK_FILE_PATH: Path | None = None

STORE_TO_GROUP = {store: grp for grp, stores in GROUP_DEFINITIONS.items() for store in stores}


def prune_low_variance(
    train_df: pd.DataFrame, cols: list[str], threshold: float = 1e-6
) -> list[str]:
    present = [c for c in cols if c in train_df.columns]
    if not present:
        return []
    chunk_size = 1_000_000
    sum_x = {c: 0.0 for c in present}
    sum_x2 = {c: 0.0 for c in present}
    count = 0
    for chunk in np.array_split(train_df[present], max(1, len(train_df) // chunk_size)):
        chunk = chunk.astype("float64", copy=False)
        cnt = len(chunk)
        count += cnt
        for c in present:
            series = chunk[c]
            sum_x[c] += series.sum()
            sum_x2[c] += (series ** 2).sum()
    kept = []
    if count == 0:
        return present
    for c in present:
        mean = sum_x[c] / count
        var = sum_x2[c] / count - mean * mean
        if var > threshold:
            kept.append(c)
    return kept


def _resolve_store_override(store: str) -> dict | None:
    override = STORE_PARAMS.get(store)
    if override:
        return override
    for state, stores in STATE_GROUPS.items():
        if store in stores:
            source = STATE_PARAM_SOURCE.get(state)
            if source and source in STORE_PARAMS:
                return STORE_PARAMS[source]
    return None


def merge_store_params(store: str, override: dict | None = None) -> dict:
    params = BASE_PARAMS.copy()
    store_override = _resolve_store_override(store)
    if store_override:
        params.update(store_override)
    if override:
        params.update(override)
    return params


def load_group_candidates(group: str, model_type: str) -> list[dict]:
    group_cache = GROUP_CANDIDATE_CACHE.setdefault(group, {})
    if model_type in group_cache:
        return group_cache[model_type]
    pattern = str(CANDIDATE_PATTERN)
    path = Path(pattern.format(group=group, model_type=model_type))
    params: list[dict] = []
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                params = [entry.get("params", {}) for entry in json.load(f) if isinstance(entry, dict)]
        except Exception:
            params = []
    group_cache[model_type] = params
    return params


def read_store(store: str, data_dir: Path, usecols: List[str]) -> pd.DataFrame:
    if CHUNK_FILE_PATH:
        path = CHUNK_FILE_PATH
    else:
        path = data_dir / f"processed_{store}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        chunk["d_int"] = chunk["d"].str.replace("d_", "", regex=False).astype(int)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df


def set_categorical(train: pd.Series, val: pd.Series) -> tuple[pd.Series, pd.Series]:
    cats = pd.CategoricalDtype(categories=train.dropna().unique())
    return train.astype(cats), val.astype(cats)


def build_datasets(
    stores: List[str], dept_filter: List[str] | None = None, item_filter: List[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    feature_cols_available = NUM_SCALED + CYCLIC + BIN_COLS + CAT_COLS
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
    usecols = [c for c in feature_cols_available + [TARGET_COL, "d"] if c not in generated_cols]
    dfs = [read_store(store, DATA_DIR, usecols) for store in stores]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    df = df.loc[:, ~df.columns.duplicated()]
    if dept_filter:
        df = df[df["dept_id"].isin(dept_filter)]
    if item_filter:
        df = df[df["item_id"].isin(item_filter)]
    df["id"] = df["id"].str.replace("_evaluation", "_validation", regex=False)
    numeric_cols = [
        c
        for c in feature_cols_available
        if c in df.columns and c not in CAT_COLS
    ]
    for col in numeric_cols:
        df.loc[:, col] = df[col].astype("float32")
    train_df = df[df["d_int"] <= TRAIN_END]
    val_df = df[(df["d_int"] > TRAIN_END) & (df["d_int"] <= VAL_END)]
    if val_df.empty:
        raise ValueError("Validation set empty.")
    for col in CAT_COLS:
        train_df[col], val_df[col] = set_categorical(train_df[col].astype(str), val_df[col].astype(str))
    item_counts = train_df["item_id"].value_counts()
    id_counts = train_df["id"].value_counts()
    train_df.loc[:, "item_freq"] = train_df["item_id"].map(item_counts).fillna(0).astype("float32")
    val_df.loc[:, "item_freq"] = val_df["item_id"].map(item_counts).fillna(0).astype("float32")
    train_df.loc[:, "id_freq"] = train_df["id"].map(id_counts).fillna(0).astype("float32")
    val_df.loc[:, "id_freq"] = val_df["id"].map(id_counts).fillna(0).astype("float32")
    for col in ["item_freq", "id_freq"]:
        mn = train_df[col].min()
        mx = train_df[col].max()
        rng = mx - mn
        if rng == 0:
            train_df.loc[:, col + "_scaled"] = 0.0
            val_df.loc[:, col + "_scaled"] = 0.0
        else:
            train_df.loc[:, col + "_scaled"] = (train_df[col] - mn) / rng
            val_df.loc[:, col + "_scaled"] = (val_df[col] - mn) / rng
    global_mean = train_df[TARGET_COL].mean()
    def mean_encode(series: pd.Series, target: pd.Series, alpha: float = 5.0) -> pd.Series:
        counts = series.value_counts()
        means = train_df.groupby(series.name)[TARGET_COL].mean()
        smoothing = (means * counts + alpha * global_mean) / (counts + alpha)
        return series.map(smoothing)
    train_df["item_te"] = mean_encode(train_df["item_id"], train_df[TARGET_COL])
    val_df["item_te"] = val_df["item_id"].map(train_df.groupby("item_id")[TARGET_COL].mean()).fillna(global_mean)
    train_df["id_te"] = mean_encode(train_df["id"], train_df[TARGET_COL])
    val_df["id_te"] = val_df["id"].map(train_df.groupby("id")[TARGET_COL].mean()).fillna(global_mean)
    for col in ["item_te", "id_te"]:
        mn = train_df[col].min()
        mx = train_df[col].max()
        rng = mx - mn
        if rng == 0:
            train_df.loc[:, col + "_scaled"] = 0.0
            val_df.loc[:, col + "_scaled"] = 0.0
        else:
            train_df.loc[:, col + "_scaled"] = (train_df[col] - mn) / rng
            val_df.loc[:, col + "_scaled"] = (val_df[col] - mn) / rng
    numeric_core_candidates = CORE_NUM_SCALED + CORE_GENERATED_SCALED
    kept_core_numeric = prune_low_variance(train_df, numeric_core_candidates)
    feature_cols_core = list(dict.fromkeys(kept_core_numeric + CORE_GENERATED_SCALED + CYCLIC + BIN_COLS + CAT_COLS))
    extra_candidates = [c for c in EXTRA_NUM_SCALED if c not in kept_core_numeric]
    # limit extras to high variance subset
    extra_candidates = [c for c in extra_candidates if c in SELECTED_EXTRAS]
    kept_extra = prune_low_variance(train_df, extra_candidates)
    feature_cols_full = list(dict.fromkeys(feature_cols_core + kept_extra))
    for col in BIN_COLS:
        train_df[col] = train_df[col].astype("int8")
        val_df[col] = val_df[col].astype("int8")
    for col in NUM_SCALED + CYCLIC:
        train_df[col] = train_df[col].astype("float32")
        val_df[col] = val_df[col].astype("float32")
    return train_df, val_df, feature_cols_core, feature_cols_full


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, min_denom: float = 1e-3) -> float:
    mask = np.abs(y_true) > min_denom
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(np.mean(2.0 * num / denom))


def wrmsse(val_df: pd.DataFrame, preds: np.ndarray) -> float:
    if not WRMSSE_ENABLED or _WRMSSE_OFFICIAL is None:
        return 0.0
    valid_ids = val_df["id"].astype(str).str.strip()
    available_meta = _WRMSSE_OFFICIAL.meta["id"].astype(str).str.strip()
    in_meta = valid_ids.isin(available_meta)
    print(
        f"WRMSSE debug: {valid_ids.nunique()} unique ids, "
        f"{in_meta.sum()} match metadata, {len(valid_ids) - in_meta.sum()} missing"
    )
    # print("val sample ids:", valid_ids.unique()[:10])
    # print("first 10 meta ids:", available_meta.unique()[:10])
    # unmatched = valid_ids[~in_meta]
    # print("sample unmatched ids:", unmatched.unique()[:10])
    y_true = val_df[["id", "d", TARGET_COL]].copy()
    y_true["d"] = y_true["d"].astype(str).str.replace("d_", "").astype(int)
    y_true = y_true.rename(columns={TARGET_COL: "sales"})
    y_pred = y_true.copy()
    y_pred["sales"] = preds.astype("float32")
    score, _ = _WRMSSE_OFFICIAL.compute_wrmsse(y_true, y_pred)
    return float(score)


def make_wrmsse_monitor(
    val_df: pd.DataFrame, feature_cols: List[str], eval_freq: int, patience: int
) -> tuple[Callable, dict]:
    state = {"history": [], "best_score": float("inf"), "best_iter": 0, "wait": 0}

    def callback(env):
        if not WRMSSE_ENABLED or _WRMSSE_OFFICIAL is None:
            return
        if eval_freq <= 0 or env.iteration % eval_freq != 0:
            return
        preds = env.model.predict(val_df[feature_cols], num_iteration=env.iteration)
        score = wrmsse(val_df, preds)
        state["history"].append((env.iteration, score))
        if score + WRMSSE_IMPROVE_TOL < state["best_score"]:
            state["best_score"] = score
            state["best_iter"] = env.iteration
            state["wait"] = 0
        else:
            state["wait"] += 1
            if state["wait"] >= patience:
                env.model.stop_training = True

    return callback, state


def train_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], params: dict) -> tuple[lgb.LGBMRegressor, dict]:
    X_train = train_df[feature_cols]
    y_train_raw = train_df[TARGET_COL].astype("float32")
    X_val = val_df[feature_cols]
    y_val_raw = val_df[TARGET_COL].astype("float32")
    sample_weight = None
    if TIME_DECAY:
        age = TRAIN_END - train_df["d_int"]
        sample_weight = np.exp(-np.log(2) * age / DECAY_HALF_LIFE).astype("float32")
    if USE_LOG1P_TARGET:
        y_train = np.log1p(y_train_raw)
        y_val = np.log1p(y_val_raw)
    else:
        y_train = y_train_raw
        y_val = y_val_raw
    callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=50)]
    wr_state = {"history": [], "best_score": float("inf"), "best_iter": 0, "wait": 0}
    if WRMSSE_ENABLED and _WRMSSE_OFFICIAL is not None:
        wr_cb, wr_state = make_wrmsse_monitor(val_df, feature_cols, WRMSSE_EVAL_FREQ, WRMSSE_PATIENCE)
        callbacks.append(wr_cb)
    model = lgb.LGBMRegressor(**params)
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=CAT_COLS,
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - start
    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    if USE_LOG1P_TARGET:
        preds = np.expm1(preds)
        y_val_used = y_val_raw.values
    else:
        y_val_used = y_val.values
    rmse = float(np.sqrt(mean_squared_error(y_val_used, preds)))
    mape = safe_mape(y_val_used, preds)
    smape_val = smape(y_val_used, preds)
    wrmsse_val = wrmsse(val_df, preds)
    if not wr_state["history"] or wr_state["history"][-1][0] != model.best_iteration_:
        wr_state["history"].append((model.best_iteration_, wrmsse_val))
    best_wrmsse = (
        min(wr_state["best_score"], wrmsse_val)
        if wr_state["best_score"] != float("inf")
        else wrmsse_val
    )
    best_wrmsse_iter = wr_state["best_iter"] or model.best_iteration_
    metrics = {
        "rmse": rmse,
        "mape": mape,
        "smape": smape_val,
        "wrmsse": wrmsse_val,
        "iter": model.best_iteration_,
        "train_time": train_time,
        "wrmsse_history": wr_state["history"],
        "best_wrmsse": float(best_wrmsse),
        "best_wrmsse_iter": best_wrmsse_iter,
    }
    print(
        f"Validation RMSE: {rmse:.4f} | MAPE (>0.001 mask): {mape:.4f} | SMAPE: {smape_val:.4f} | WRMSSE: {wrmsse_val:.4f}"
    )
    return model, metrics


def evaluate_candidate_for_store(
    train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], params: dict
) -> dict | None:
    if val_df.empty:
        return None
    model, metrics = train_lgbm(train_df, val_df, feature_cols, params)
    prev_mask = train_df["d_int"].between(1886, TRAIN_END)
    prev_df = train_df[prev_mask]
    preds_val = model.predict(val_df[feature_cols], num_iteration=model.best_iteration_)
    score_val = wrmsse(val_df, preds_val)
    preds_prev = None
    score_prev = float("inf")
    if not prev_df.empty:
        preds_prev = model.predict(prev_df[feature_cols])
        score_prev = wrmsse(prev_df, preds_prev)
    avg_score = (score_prev + score_val) / 2.0
    return {
        "model": model,
        "params": params,
        "metrics": metrics,
        "score_prev": score_prev,
        "score_val": score_val,
        "score_avg": avg_score,
        "preds_val": preds_val,
        "val_df": val_df,
    }


def evaluate_candidates_for_store(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    candidate_params: list[dict],
) -> dict | None:
    best = None
    if not candidate_params:
        return None
    for params in candidate_params:
        result = evaluate_candidate_for_store(train_df, val_df, feature_cols, params)
        if not result:
            continue
        if best is None or result["score_avg"] < best["score_avg"]:
            best = result
    return best


def find_best_blend(val_df: pd.DataFrame, preds_base: np.ndarray, preds_alt: np.ndarray) -> dict:
    y_val = val_df[TARGET_COL].astype("float32").values
    best = {
        "type": "main",
        "w_base": 1.0,
        "w_alt": 0.0,
        "wrmsse": wrmsse(val_df, preds_base),
    }
    if preds_alt is None:
        return best
    for w in BLEND_WEIGHTS:
        blended = w * preds_base + (1.0 - w) * preds_alt
        score = wrmsse(val_df, blended)
        if score < best["wrmsse"]:
            best = {
                "type": "2model",
                "w_base": w,
                "w_alt": 1.0 - w,
                "wrmsse": score,
            }
    return best


def finalize_blend_weights(decisions: list[dict]) -> dict:
    weights = {}
    for dec in decisions:
        store = dec["stores"][0]
        auto_step = dec["auto_decision"]
        if auto_step == "allow" and dec["main_result"] and dec["c_result"]:
            blend_info = find_best_blend(
                dec["main_result"]["val_df"],
                dec["main_result"]["preds_val"],
                dec["c_result"]["preds_val"],
            )
        else:
            # default to main model only
            base_preds = (
                dec["main_result"]["preds_val"] if dec["main_result"] else None
            )
            blend_info = {
                "type": "main",
                "w_base": 1.0,
                "w_alt": 0.0,
                "wrmsse": wrmsse(dec["main_result"]["val_df"], base_preds)
                if base_preds is not None
                else None,
            }
        weights[store] = blend_info
    return weights


def get_chunk_key() -> str | None:
    if CHUNK_FILE_PATH is None:
        return None
    return CHUNK_FILE_PATH.name


def merge_weight_entries(
    base_weights: dict[str, dict], new_weights: dict[str, dict], chunk_key: str | None
) -> dict[str, dict]:
    merged = base_weights.copy()
    for store, entry in new_weights.items():
        updated = dict(entry)
        if chunk_key:
            prev = merged.get(store, {})
            history = prev.get("chunk_history", [])
            history = list(history) if isinstance(history, list) else []
            snapshot = updated.copy()
            snapshot.pop("chunk_history", None)
            history.append({"chunk": chunk_key, "weight": snapshot})
            updated["chunk_history"] = history
        merged[store] = updated
    return merged
def time_series_folds(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    folds = []
    val1_start = TRAIN_END - CV_VAL_LEN + 1
    val1_end = TRAIN_END
    train1 = df[df["d_int"] < val1_start]
    val1 = df[(df["d_int"] >= val1_start) & (df["d_int"] <= val1_end)]
    if not val1.empty and len(train1) > 0:
        folds.append((train1, val1))
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
        _, metrics = train_lgbm(tr, va, feature_cols, params)
        metrics_list.append(metrics)
    avg = {k: float(np.nanmean([m[k] for m in metrics_list])) for k in ["rmse", "mape", "smape", "wrmsse"]}
    avg["iter"] = int(np.nanmean([m["iter"] for m in metrics_list]))
    return avg


def tweak_params_for_blend(params: dict) -> dict:
    p = params.copy()
    p["learning_rate"] = max(1e-4, params.get("learning_rate", 0.03) * 0.8)
    p["n_estimators"] = int(params.get("n_estimators", 1800) * 1.3)
    p["num_leaves"] = int(params.get("num_leaves", 255) * 1.2)
    p["min_data_in_leaf"] = max(50, int(params.get("min_data_in_leaf", 200) * 0.8))
    return p


def tweak_params_for_blend2(params: dict) -> dict:
    p = params.copy()
    p["learning_rate"] = min(0.1, params.get("learning_rate", 0.03) * 1.1)
    p["num_leaves"] = max(31, int(params.get("num_leaves", 255) * 0.7))
    p["n_estimators"] = int(params.get("n_estimators", 1800) * 1.15)
    p["feature_fraction"] = min(1.0, params.get("feature_fraction", 0.8) * 1.05)
    p["bagging_fraction"] = min(1.0, params.get("bagging_fraction", 0.8) * 1.05)
    p["min_data_in_leaf"] = int(params.get("min_data_in_leaf", 200) * 1.2)
    return p


def tweak_params_for_blend3(params: dict) -> dict:
    p = params.copy()
    p["learning_rate"] = max(1e-4, params.get("learning_rate", 0.03) * 0.7)
    p["num_leaves"] = max(63, int(params.get("num_leaves", 255) * 0.8))
    p["min_data_in_leaf"] = int(params.get("min_data_in_leaf", 200) * 1.5)
    p["n_estimators"] = int(params.get("n_estimators", 1800) * 1.2)
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-store training with chunked inputs.")
    parser.add_argument("--stores", nargs="+", help="Subset of stores to run (e.g. CA_1).")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of stores to include in each batch when chunking the default store list.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Zero-based index of the batch to run when --batch-size is set.",
    )
    parser.add_argument(
        "--single-candidate",
        action="store_true",
        help="Evaluate only the first main/c candidate (for quick testing).",
    )
    parser.add_argument(
        "--chunk-file",
        type=Path,
        help="Path to the chunked CSV to read for this run.",
    )
    return parser.parse_args()


def chunk_store_list(stores: List[str], batch_size: int) -> list[list[str]]:
    return [stores[i : i + batch_size] for i in range(0, len(stores), batch_size)]


def load_existing_summary() -> list[dict]:
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_existing_weights() -> dict:
    if BLEND_WEIGHT_PATH.exists():
        with BLEND_WEIGHT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            combined = {}
            for entry in data:
                if isinstance(entry, dict):
                    combined.update(entry)
            return combined
        if isinstance(data, dict):
            return data
    return {}


def main() -> None:
    global CHUNK_FILE_PATH
    args = parse_args()
    CHUNK_FILE_PATH = args.chunk_file

    existing_summary = load_existing_summary()
    existing_stores = {entry["stores"][0] for entry in existing_summary}
    existing_weights = load_existing_weights()

    tasks = []
    store_sequence = args.stores or STORE_LIST
    if args.batch_size:
        batches = chunk_store_list(store_sequence, args.batch_size)
        if args.batch_index >= len(batches):
            raise ValueError("batch_index is out of range for the provided batch_size")
        store_sequence = batches[args.batch_index]

    for store in store_sequence:
        if store in existing_stores and args.chunk_file is None:
            print(f"Skipping {store} (already processed)")
            continue
        override = None if store in SEARCH_STORES else STORE_PARAMS.get(store)
        tasks.append((f"store {store}", [store], None, None, override))

    if not tasks:
        print("No new stores to process.")
        return

    summary = []
    decisions = []
    for label, stores, dept_filter, item_filter, override_params in tasks:
        print(f"\n===== Training {label} | stores: {stores} from {DATA_DIR} =====")
        train_df, val_df, feature_cols_core, feature_cols_full = build_datasets(
            stores, dept_filter=dept_filter, item_filter=item_filter
        )
        feature_cols = list(dict.fromkeys(feature_cols_core))
        extra_count = len(feature_cols_full) - len(feature_cols_core)
        print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")
        print(
            f"Features: core={len(feature_cols_core):,}, extras={extra_count:,} (full={len(feature_cols_full):,})"
        )
        group = STORE_TO_GROUP.get(stores[0], "group_a")
        main_candidates = load_group_candidates(group, "main")
        if not main_candidates:
            main_candidates = [merge_store_params(stores[0], override_params)]
        if args.single_candidate:
            main_candidates = main_candidates[:1]
        print(f"Evaluating main/c candidate pools for group {group}")
        main_result = evaluate_candidates_for_store(train_df, val_df, feature_cols, main_candidates)
        c_result = None
        if ENABLE_BLEND:
            c_candidates = load_group_candidates(group, "c_model")
            if not c_candidates:
                c_candidates = [merge_store_params(stores[0], override_params)]
            if args.single_candidate:
                c_candidates = c_candidates[:1]
            c_result = evaluate_candidates_for_store(train_df, val_df, feature_cols_full, c_candidates)
        auto_decision = "ban"
        delta_wrmsse = None
        if main_result and c_result:
            delta_wrmsse = c_result["score_avg"] - main_result["score_avg"]
            both_better = (
                c_result["score_prev"] < main_result["score_prev"]
                and c_result["score_val"] < main_result["score_val"]
            )
            if delta_wrmsse < -0.02 and both_better:
                auto_decision = "allow"
            elif abs(delta_wrmsse) <= 0.01:
                auto_decision = "neutral"
            else:
                auto_decision = "ban"
        if auto_decision == "ban":
            NO_BLEND_STORES.add(stores[0])
        decision = {
            "label": label,
            "stores": stores,
            "group": group,
            "main_params": main_result["params"] if main_result else None,
            "main_wrmsse_prev": main_result["score_prev"] if main_result else None,
            "main_wrmsse_curr": main_result["score_val"] if main_result else None,
            "main_wrmsse": main_result["score_avg"] if main_result else None,
            "c_params": c_result["params"] if c_result else None,
            "c_wrmsse_prev": c_result["score_prev"] if c_result else None,
            "c_wrmsse_curr": c_result["score_val"] if c_result else None,
            "c_wrmsse": c_result["score_avg"] if c_result else None,
            "auto_decision": auto_decision,
            "wrmsse_delta": delta_wrmsse,
            "main_result": main_result,
            "c_result": c_result,
        }
        decisions.append(decision)
        summary_entry = {k: decision[k] for k in decision if k not in {"main_result", "c_result"}}
        if chunk_key := get_chunk_key():
            summary_entry["chunk"] = chunk_key
        summary.append(summary_entry)
    print("\n===== Summary per task =====")
    for row in summary:
        print(row)
    try:
        final_summary = existing_summary + summary
        with SUMMARY_PATH.open("w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)
        blend_weights = finalize_blend_weights(decisions)
        base_weights = existing_weights if isinstance(existing_weights, dict) else {}
        final_weights = merge_weight_entries(base_weights, blend_weights, get_chunk_key())
        with BLEND_WEIGHT_PATH.open("w", encoding="utf-8") as f:
            json.dump(final_weights, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to {SUMMARY_PATH} and blend weights to {BLEND_WEIGHT_PATH}")
        if (
            VISUALIZE_SCRIPT.exists()
            and VISUALIZE_BASE.exists()
            and VISUALIZE_ALT.exists()
        ):
            cmd = [
                "python",
                str(VISUALIZE_SCRIPT),
                "--summary",
                str(SUMMARY_PATH),
                "--weights",
                str(BLEND_WEIGHT_PATH),
                "--base",
                str(VISUALIZE_BASE),
                "--alt",
                str(VISUALIZE_ALT),
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.SubprocessError as e:
                print(f"Visualization step failed: {e}")
    except Exception as e:
        print(f"Warning: failed to save summary/weights ({e})")


if __name__ == "__main__":
    main()
