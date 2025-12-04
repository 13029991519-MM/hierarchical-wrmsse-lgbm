"""
Quick MAE/RMSE comparison on the validation window (d_1914–d_1941) between two submissions.

Inputs:
- submission_a: CSV with id, F1..F28 (e.g., original submission_with_val.csv)
- submission_b: CSV with id, F1..F28 (e.g., submission_with_val_reconciled.csv)
- sales: data/sales_train_evaluation.csv (provides d_1..d_1941 ground truth)

Behavior:
- Uses only `_validation` rows from submissions.
- Strips `_validation/_evaluation` suffix to align with sales ids.
- Computes overall MAE and RMSE on d_1914–d_1941.

Usage (PowerShell):
python evaluate_mae_rmse.py `
  --submission_a future_finaldata/submission_with_val.csv `
  --submission_b future_finaldata/submission_with_val_reconciled.csv `
  --sales data/sales_train_evaluation.csv
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MAE/RMSE on validation window for two submissions.")
    p.add_argument("--submission_a", type=Path, required=True, help="Path to submission A CSV.")
    p.add_argument("--submission_b", type=Path, required=True, help="Path to submission B CSV.")
    p.add_argument("--sales", type=Path, default=Path("data/sales_train_evaluation.csv"), help="Ground truth sales file.")
    return p.parse_args()


def load_validation(sub_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sub_path)
    df = df[df["id"].str.endswith("_validation")].copy()
    if df.empty:
        raise ValueError(f"No _validation rows in {sub_path}")
    f_cols = [f"F{i}" for i in range(1, 29)]
    missing = [c for c in f_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{sub_path} missing columns: {missing}")
    df["base_id"] = df["id"].str.replace("_validation", "", regex=False)
    df = df[["base_id"] + f_cols]
    return df


def load_ground_truth(sales_path: Path, base_ids: list[str]) -> pd.DataFrame:
    day_cols = [f"d_{1913 + i}" for i in range(1, 29)]  # d_1914..d_1941
    gt = pd.read_csv(sales_path)
    gt["base_id"] = (
        gt["id"]
        .str.replace("_validation", "", regex=False)
        .str.replace("_evaluation", "", regex=False)
    )
    gt = gt.set_index("base_id").loc[base_ids, day_cols]
    return gt


def mae_rmse(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    diff = pred - gt
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    return mae, rmse


def main() -> None:
    args = parse_args()
    sub_a = load_validation(args.submission_a)
    sub_b = load_validation(args.submission_b)

    # Align base_ids
    if not sub_a["base_id"].equals(sub_b["base_id"]):
        raise ValueError("Submissions have different validation base_id order/content.")
    base_ids = sub_a["base_id"].tolist()

    gt = load_ground_truth(args.sales, base_ids)

    f_cols = [f"F{i}" for i in range(1, 29)]
    pred_a = sub_a[f_cols].to_numpy(dtype=np.float64)
    pred_b = sub_b[f_cols].to_numpy(dtype=np.float64)
    gt_mat = gt.to_numpy(dtype=np.float64)

    mae_a, rmse_a = mae_rmse(pred_a, gt_mat)
    mae_b, rmse_b = mae_rmse(pred_b, gt_mat)

    print(f"Validation window d_1914–d_1941, series count {len(base_ids)}, cells {gt_mat.size}")
    print(f"A ({args.submission_a}): MAE={mae_a:.4f}, RMSE={rmse_a:.4f}")
    print(f"B ({args.submission_b}): MAE={mae_b:.4f}, RMSE={rmse_b:.4f}")
    delta_mae = mae_b - mae_a
    delta_rmse = rmse_b - rmse_a
    print(f"Delta (B - A): MAE={delta_mae:+.4f}, RMSE={delta_rmse:+.4f}")


if __name__ == "__main__":
    main()
