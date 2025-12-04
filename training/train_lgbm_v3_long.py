"""
Train long-season LightGBM model on processed_v3 features (lag 91/182/365 + price stats), validate on d_1914–d_1941.
Default stores: CA_1, TX_1, WI_1. Outputs metrics to weight_v2/summary_v3_long.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

DATA_DIR = Path("processed_v3")
OUT_DIR = Path("weight_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_DIR / "summary_v3_long.json"

STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3"]
TRAIN_END = 1913
VAL_END = 1941
TARGET_COL = "sales"

PARAMS = dict(
    objective="regression",
    metric=["rmse", "mape"],
    learning_rate=0.05,
    num_leaves=383,
    max_depth=-1,
    feature_fraction=0.80,
    bagging_fraction=0.80,
    bagging_freq=5,
    min_data_in_leaf=120,
    lambda_l1=0.5,
    lambda_l2=1.0,
    n_estimators=2000,
    max_bin=255,
)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, min_denom: float = 1e-3) -> float:
    mask = np.abs(y_true) > min_denom
    if not mask.any():
        return float("nan")
    denom = np.abs(y_true[mask])
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom))


def read_store(store: str) -> pd.DataFrame:
    path = DATA_DIR / f"processed_{store}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["d_int"] = df["d_int"].astype(int)
    return df


def build_datasets(store: str) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    df = read_store(store)
    cat_cols = ["state_id", "store_id", "cat_id", "dept_id", "item_id", "id", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    bin_cols = [c for c in df.columns if c in ["snap_CA", "snap_TX", "snap_WI", "IsHoliday", "IsPromotion", "is_discounted"]]
    ignore_cols = set(["d", "d_int", TARGET_COL])
    str_cols = [c for c in df.columns if df[c].dtype == "string" and c not in cat_cols]
    num_cols = [c for c in df.columns if c not in cat_cols + bin_cols + list(ignore_cols) + str_cols]

    train_df = df[df["d_int"] <= TRAIN_END].copy()
    val_df = df[(df["d_int"] > TRAIN_END) & (df["d_int"] <= VAL_END)].copy()

    for col in cat_cols:
        if col in train_df:
            cats = pd.CategoricalDtype(categories=train_df[col].dropna().unique())
            train_df[col] = train_df[col].astype(cats)
            val_df[col] = val_df[col].astype(cats)
    for col in bin_cols:
        if col in train_df:
            train_df[col] = train_df[col].astype("int8")
            val_df[col] = val_df[col].astype("int8")
    for col in num_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").astype("float32")
        val_df[col] = pd.to_numeric(val_df[col], errors="coerce").astype("float32")

    feature_cols = num_cols + bin_cols + cat_cols
    return train_df, val_df, feature_cols, cat_cols


def train_eval(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str], params: dict):
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
        categorical_feature=[c for c in cat_cols if c in feature_cols],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mape = safe_mape(y_val.values, preds)
    return {"rmse": rmse, "mape": mape, "iter": model.best_iteration_}


def main() -> None:
    summary = []
    for store in STORES:
        print(f"\n=== Training v3 long-season model on {store} ===")
        train_df, val_df, feature_cols, cat_cols = build_datasets(store)
        metrics = train_eval(train_df, val_df, feature_cols, cat_cols, PARAMS)
        print(f"{store}: RMSE {metrics['rmse']:.4f} | MAPE {metrics['mape']:.4f} | iter {metrics['iter']}")
        summary.append({"store": store, **metrics})
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
