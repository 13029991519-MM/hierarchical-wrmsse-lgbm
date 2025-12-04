"""
Compute WRMSSE on d_1914-1941 for a blended submission against the official validation sales.

Usage:
    python evaluate_wrmsse_blended.py \
        --submission future_finaldata/submission_with_val_blended.csv \
        --sales data/sales_train_validation.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wrmsse_official import WRMSSEEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate WRMSSE for a blended validation submission.")
    parser.add_argument("--submission", type=Path, required=True, help="wide submission with F1..F28")
    parser.add_argument(
        "--sales",
        type=Path,
        default=Path("data/sales_train_validation.csv"),
        help="Official sales file providing the validation truth.",
    )
    return parser.parse_args()


def submission_to_long(submission_path: Path) -> pd.DataFrame:
    df = pd.read_csv(submission_path)
    if "id" not in df.columns:
        raise ValueError(f"{submission_path} missing id column")
    f_cols = [f"F{i}" for i in range(1, 29)]
    for col in f_cols:
        if col not in df.columns:
            raise ValueError(f"{submission_path} missing column {col}")
    df = df[df["id"].str.endswith("_validation")].copy()
    df = df[["id"] + f_cols]
    long = df.melt(id_vars=["id"], value_vars=f_cols, var_name="F", value_name="sales")
    long["d"] = long["F"].str.extract(r"F(\d+)").astype(int) + 1913
    long["sales"] = long["sales"].astype("float32")
    return long.drop(columns=["F"])


def truth_long(sales_path: Path) -> pd.DataFrame:
    wide = pd.read_csv(sales_path)
    day_cols = [f"d_{d}" for d in range(1914, 1942)]
    required = ["id"] + day_cols
    missing = [c for c in required if c not in wide.columns]
    if missing:
        raise ValueError(f"{sales_path} missing columns: {missing}")
    long = wide[required].melt(id_vars=["id"], var_name="d", value_name="sales")
    long["d"] = long["d"].str.replace("d_", "").astype(int)
    return long


def main() -> None:
    args = parse_args()
    evaluator = WRMSSEEvaluator()
    truth = truth_long(args.sales)
    preds = submission_to_long(args.submission)

    score, _ = evaluator.compute_wrmsse(truth, preds)
    print(f"WRMSSE on validation window (d_1914-1941): {score:.6f}")


if __name__ == "__main__":
    main()
