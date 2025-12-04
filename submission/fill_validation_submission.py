"""
Fill the _validation part of submission using models trained on d_1–1913.
Keeps existing _evaluation predictions (e.g., from future_finaldata/submission.csv).
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
from typing import List

import numpy as np
import pandas as pd

import train_lgbm_baseline as lgb_base


MODEL_TYPES = ("main", "c_model")
SUMMARY_PATH = pathlib.Path("weight_v2/summary_delay120_v2.json")

VAL_START = 1914
VAL_END = 1941  # inclusive


def train_single_model(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], params: dict):
    model, _ = lgb_base.train_lgbm(train_df, val_df, feature_cols, params)
    preds = model.predict(val_df[feature_cols], num_iteration=model.best_iteration_)
    out = val_df[["id", "d"]].copy()
    out["pred"] = preds
    return out


def load_summary() -> dict[str, dict]:
    if not SUMMARY_PATH.exists():
        return {}
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {
        entry["stores"][0]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("stores"), list) and entry["stores"]
    }


def select_params(store: str, summary: dict[str, dict], model_type: str) -> dict:
    entry = summary.get(store, {})
    key = "main_params" if model_type == "main" else "c_params"
    params = entry.get(key)
    if isinstance(params, dict):
        return params
    return lgb_base.merge_store_params(store)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stores",
        type=str,
        default="",
        help="comma-separated store ids to process (default: all)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_TYPES,
        default="main",
        help="Which model type to train for validation fill (default: main).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        help="Optional output path for the filled submission",
    )
    args = parser.parse_args()

    base_submission_path = pathlib.Path("future_finaldata/submission.csv")
    sample_path = pathlib.Path("data/sample_submission.csv")
    model_type = args.model_type
    default_out = (
        pathlib.Path("future_finaldata/submission_with_val.csv")
        if model_type == "main"
        else pathlib.Path("future_finaldata/submission_with_val_cmodel.csv")
    )
    out_path = args.out or default_out
    summary = load_summary()

    if not base_submission_path.exists():
        raise FileNotFoundError(f"Base submission not found: {base_submission_path}")

    sub = pd.read_csv(base_submission_path)
    sample = pd.read_csv(sample_path)

    # Ensure F columns are float to avoid dtype warnings
    f_cols = [c for c in sub.columns if c.startswith("F")]
    sub[f_cols] = sub[f_cols].astype(float)

    if args.stores:
        stores = [s.strip() for s in args.stores.split(",") if s.strip()]
    else:
        stores = list(lgb_base.STORE_LIST)
    for store in stores:
        print(f"Processing store {store} for validation fill...")
        train_df, val_df, feature_cols_core, feature_cols_full = lgb_base.build_datasets([store])
        # ensure numeric day for slicing; keep original d for pivot
        train_df["d_num"] = pd.to_numeric(train_df["d"].astype(str).str.replace("d_", ""), errors="coerce")
        val_df["d_num"] = pd.to_numeric(val_df["d"].astype(str).str.replace("d_", ""), errors="coerce")

        # keep only validation slice d_1914–1941
        val_df = val_df[(val_df["d_num"] >= VAL_START) & (val_df["d_num"] <= VAL_END)]
        feature_cols = feature_cols_core if model_type == "main" else feature_cols_full
        params = select_params(store, summary, model_type)
        preds_df = train_single_model(train_df, val_df, feature_cols, params)

        # pivot to wide F1..F28
        wide = preds_df.pivot(index="id", columns="d", values="pred")
        wide = wide.reindex(columns=[f"d_{d}" for d in range(VAL_START, VAL_END + 1)])
        # map d_1914 -> F1, ...
        f_map = {f"d_{d}": f"F{d - VAL_START + 1}" for d in range(VAL_START, VAL_END + 1)}

        sub_ids = set(sub["id"])
        for rid, row in wide.iterrows():
            if rid.endswith("_evaluation"):
                sub_id = rid.replace("_evaluation", "_validation")
            elif rid.endswith("_validation"):
                sub_id = rid
            else:
                sub_id = f"{rid}_validation"
            if sub_id not in sub_ids:
                continue
            for d_col, f_col in f_map.items():
                sub.loc[sub["id"] == sub_id, f_col] = row[d_col]
        # free memory
        del train_df, val_df, preds_df, wide
        gc.collect()

    # sanity: keep rows/cols aligned to sample
    sub = sub.loc[:, sample.columns]
    sub.to_csv(out_path, index=False)
    print(f"Wrote submission with validation filled: {out_path}")


if __name__ == "__main__":
    main()
