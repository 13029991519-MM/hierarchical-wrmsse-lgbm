"""
Global random search over store groups and model types.

Step1 in your workflow: for each (model_type, group) combination, run limited random search
and save the top-performing parameter sets for later store-level experiments.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from statistics import mean

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_lgbm_baseline import (
    BASE_PARAMS,
    CORE_GENERATED_SCALED,
    CORE_LAG_SCALED,
    CORE_ROLL_SCALED,
    CORE_PRICE_SCALED,
    CYCLIC,
    BIN_COLS,
    CAT_COLS,
    DATA_DIR,
    EXTRA_LAG_SCALED,
    EXTRA_ROLL_MEAN,
    EXTRA_ROLL_STD,
    EXTRA_ROLL_MED,
    EXTRA_ROLL_MIN,
    EXTRA_ROLL_MAX,
    EXTRA_PRICE_MOMENTUM,
    EXTRA_SEASONAL,
    TARGET_COL,
    TRAIN_END,
    VAL_END,
    read_store,
    set_categorical,
)
from wrmsse_official import WRMSSEEvaluator

OUT_DIR = Path("weight_v2")
OUT_DIR.mkdir(exist_ok=True)

CORE_FEATURES = (
    CORE_LAG_SCALED
    + CORE_ROLL_SCALED
    + CORE_PRICE_SCALED
    + CYCLIC
    + BIN_COLS
    + CAT_COLS
)

EXTRA_FEATURES = (
    CORE_FEATURES
    + EXTRA_LAG_SCALED
    + EXTRA_ROLL_MEAN
    + EXTRA_ROLL_STD
    + EXTRA_ROLL_MED
    + EXTRA_ROLL_MIN
    + EXTRA_ROLL_MAX
    + EXTRA_PRICE_MOMENTUM
    + EXTRA_SEASONAL
)

GROUPS = {
    "group_a": ["CA_1", "CA_2", "CA_4", "TX_1"],
    "group_b": ["CA_3", "WI_1", "WI_2"],
}

MODEL_TYPES = {
    "main": CORE_FEATURES,
    "c_model": EXTRA_FEATURES,
}


def random_params(base: dict) -> dict:
    return {
        **base,
        "learning_rate": max(1e-3, base["learning_rate"] * random.uniform(0.8, 1.2)),
        "num_leaves": random.choice([160, 200, 255, 300, 383]),
        "feature_fraction": random.uniform(0.6, 0.95),
        "bagging_fraction": random.uniform(0.6, 0.95),
        "bagging_freq": random.choice([3, 5, 7]),
        "lambda_l1": random.choice([0.0, 0.1, 0.5]),
        "lambda_l2": random.choice([0.5, 1.0, 2.0]),
        "max_depth": random.choice([-1, 8, 10]),
        "min_data_in_leaf": random.choice([100, 150, 200]),
    }


def prepare_store_df(store: str, cols: list[str]) -> pd.DataFrame:
    df = read_store(store, DATA_DIR, usecols=cols)
    df["d_int"] = df["d"].str.replace("d_", "", regex=False).astype(int)
    df["id"] = df["id"].str.replace("_evaluation", "_validation", regex=False)
    for col in CAT_COLS:
        df[col] = df[col].astype(str)
    return df


def compute_wrmsse(ev: WRMSSEEvaluator, subset: pd.DataFrame, preds: np.ndarray) -> float:
    y_true = subset[["id", "d", TARGET_COL]].copy()
    y_true["d"] = y_true["d"].astype(str).str.replace("d_", "").astype(int)
    y_true = y_true.rename(columns={TARGET_COL: "sales"})
    y_pred = y_true.copy()
    y_pred["sales"] = preds.astype("float32")
    score, _ = ev.compute_wrmsse(y_true, y_pred)
    return float(score)


def evaluate_params(stores: list[str], feature_cols: list[str], params: dict) -> float:
    ev = WRMSSEEvaluator()
    scores = []
    cols = list(dict.fromkeys(feature_cols + [TARGET_COL, "d", "d_int", "id"]))
    ev = WRMSSEEvaluator()
    scores = []
    for store in stores:
        df = prepare_store_df(store, cols)
        train_df = df[df["d_int"] <= TRAIN_END].copy()
        val_df = df[(df["d_int"] > TRAIN_END) & (df["d_int"] <= VAL_END)].copy()
        if val_df.empty:
            continue
        for col in CAT_COLS:
            train_df[col], val_df[col] = set_categorical(train_df[col], val_df[col])
        feats = list(dict.fromkeys(feature_cols))
        model = lgb.LGBMRegressor(**params)
        model.fit(
            train_df[feats],
            train_df[TARGET_COL].astype("float32"),
            eval_set=[(val_df[feats], val_df[TARGET_COL].astype("float32"))],
        )
        preds_val = model.predict(val_df[feats])
        prev_mask = train_df["d_int"].between(1886, TRAIN_END)
        prev_df = train_df[prev_mask]
        if prev_df.empty:
            continue
        preds_prev = model.predict(prev_df[feats])
        score_prev = compute_wrmsse(ev, prev_df, preds_prev)
        score_val = compute_wrmsse(ev, val_df, preds_val)
        scores.append((score_prev + score_val) / 2.0)
    return float(np.mean(scores)) if scores else float("inf")


def search_group(group: str, model_type: str, trials: int, top_k: int) -> None:
    stores = GROUPS[group]
    features = MODEL_TYPES[model_type]
    base = BASE_PARAMS.copy()
    results = []
    for trial in range(trials):
        params = random_params(base)
        score = evaluate_params(stores, features, params)
        results.append({"params": params, "score": score})
        print(f"[{group} | {model_type}] Trial {trial+1}/{trials}: score={score:.4f}")
    best = sorted(results, key=lambda x: x["score"])[:top_k]
    out_path = OUT_DIR / f"group_params_{group}_{model_type}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(f"Saved top {top_k} candidates for {group}/{model_type} to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global random search for group/model combos.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--groups", nargs="+", default=list(GROUPS))
    parser.add_argument("--model_types", nargs="+", default=list(MODEL_TYPES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for group, model_type in itertools.product(args.groups, args.model_types):
        if group not in GROUPS or model_type not in MODEL_TYPES:
            continue
        search_group(group, model_type, args.trials, args.top_k)


if __name__ == "__main__":
    main()
