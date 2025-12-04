"""
Export validation predictions (d_1914–d_1941) for two LGBM variants:
- A: baseline feature set (full features), uses STORE_PARAMS if available else BASE_PARAMS.
- C: mask price/discount features (same as train_lgbm_mask_c.py).

Outputs:
- weight_v2/preds_val_A.csv
- weight_v2/preds_val_C.csv

Default stores: CA_1, TX_1, WI_1 (configurable via --stores).
Does not overwrite submissions; for offline weight search only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from train_lgbm_baseline import (
    BIN_COLS,
    CAT_COLS,
    CYCLIC,
    NUM_SCALED,
    STORE_PARAMS,
    TARGET_COL,
    TRAIN_END,
    VAL_END,
    safe_mape,
)

# Paths
DATA_DIR = Path("newfinaldata")
OUT_DIR = Path("weight_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_A = OUT_DIR / "preds_val_A.csv"
OUT_C = OUT_DIR / "preds_val_C.csv"

# Mask price/discount cols for variant C
PRICE_COLS = [c for c in NUM_SCALED if "price" in c or "discount" in c or "promo" in c]

# Params
BASE_PARAMS = dict(
    objective="regression",
    metric=["rmse", "mape"],
    learning_rate=0.05,
    num_leaves=255,
    max_depth=10,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    min_data_in_leaf=200,
    lambda_l1=0.5,
    lambda_l2=1.0,
    n_estimators=2400,
    max_bin=511,
)
PARAMS_C = dict(
    objective="regression",
    metric=["rmse", "mape"],
    learning_rate=0.05,
    num_leaves=255,
    max_depth=10,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    min_data_in_leaf=240,
    lambda_l1=0.5,
    lambda_l2=1.0,
    n_estimators=2400,
    max_bin=511,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export validation preds for A and C variants.")
    p.add_argument("--stores", type=str, default="CA_1,TX_1,WI_1", help="Comma-separated store_ids to run.")
    return p.parse_args()


def read_store(store: str, usecols: List[str]) -> pd.DataFrame:
    path = DATA_DIR / f"processed_{store}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=usecols)
    # cast categories/bools later
    df["d_int"] = df["d"].str.replace("d_", "", regex=False).astype(int)
    return df


def build_datasets(stores: List[str], feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usecols = feature_cols + [TARGET_COL, "d"]
    dfs = [read_store(s, usecols) for s in stores]
    df = pd.concat(dfs, ignore_index=True)
    train_df = df[df["d_int"] <= TRAIN_END].copy()
    val_df = df[(df["d_int"] > TRAIN_END) & (df["d_int"] <= VAL_END)].copy()
    for col in CAT_COLS:
        if col in train_df:
            cats = pd.CategoricalDtype(categories=train_df[col].dropna().unique())
            train_df[col] = train_df[col].astype(cats)
            val_df[col] = val_df[col].astype(cats)
    for col in BIN_COLS:
        if col in train_df:
            train_df[col] = train_df[col].astype("int8")
            val_df[col] = val_df[col].astype("int8")
    for col in feature_cols:
        if col not in CAT_COLS + BIN_COLS and col in train_df:
            train_df[col] = train_df[col].astype("float32")
            val_df[col] = val_df[col].astype("float32")
    return train_df, val_df


def train_and_pred(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], params: dict) -> Tuple[pd.DataFrame, dict]:
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype("float32")
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL].astype("float32")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mape = safe_mape(y_val.values, preds)
    return pd.DataFrame({"id": val_df["id"], "pred": preds, "d_int": val_df["d_int"]}), {"rmse": rmse, "mape": mape, "iter": model.best_iteration_}


def pivot_preds(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide F1..F28; keeps only d_1914..d_1941 to avoid huge matrices."""
    df = df.copy()
    df = df[(df["d_int"] >= 1914) & (df["d_int"] <= 1941)]
    pivot = df.pivot(index="id", columns="d_int", values="pred").astype("float32")
    cols = list(range(1914, 1942))
    missing = [c for c in cols if c not in pivot.columns]
    if missing:
        raise ValueError(f"Missing d columns for validation window: {missing}")
    pivot = pivot[cols]
    pivot.columns = [f"F{i}" for i in range(1, 29)]
    pivot.reset_index(inplace=True)
    return pivot


def main() -> None:
    args = parse_args()
    stores = [s.strip() for s in args.stores.split(",") if s.strip()]

    # Model A (full)
    feature_cols_a = NUM_SCALED + CYCLIC + BIN_COLS + CAT_COLS
    train_a, val_a = build_datasets(stores, feature_cols_a)
    params_a = STORE_PARAMS.get(stores[0], BASE_PARAMS) if len(stores) == 1 else BASE_PARAMS
    preds_a, metrics_a = train_and_pred(train_a, val_a, feature_cols_a, params_a)
    pivot_a = pivot_preds(preds_a)
    pivot_a.to_csv(OUT_A, index=False)
    print(f"Saved A preds to {OUT_A}, metrics {metrics_a}")

    # Model C (mask price)
    feature_cols_c = [c for c in NUM_SCALED if c not in PRICE_COLS] + CYCLIC + BIN_COLS + CAT_COLS
    train_c, val_c = build_datasets(stores, feature_cols_c)
    preds_c, metrics_c = train_and_pred(train_c, val_c, feature_cols_c, PARAMS_C)
    pivot_c = pivot_preds(preds_c)
    pivot_c.to_csv(OUT_C, index=False)
    print(f"Saved C preds to {OUT_C}, metrics {metrics_c}")

    summary = {
        "stores": stores,
        "A": metrics_a,
        "C": metrics_c,
    }
    with (OUT_DIR / "summary_val_A_C.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {OUT_DIR/'summary_val_A_C.json'}")


if __name__ == "__main__":
    main()
