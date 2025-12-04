"""
Compute MAE/RMSE per hierarchy level on validation window (d_1914–d_1941) for one submission.

Inputs:
- submission: CSV with id, F1..F28 (validation rows required)
- sales: data/sales_train_evaluation.csv (ground truth with d_1..d_1941)
- hierarchy_cache/S.npz and nodes.json and bottom_ids.json (from build_hierarchy_and_g.py)

Behavior:
- Aligns validation rows to bottom_ids order.
- Aggregates bottom forecasts/ground truth to all nodes via S.
- Prints MAE/RMSE grouped by level (all/state/store/cat/dept/state_cat/state_dept/store_cat/store_dept/item/state_item/store_item).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-level MAE/RMSE on validation window.")
    p.add_argument("--submission", type=Path, required=True, help="Submission CSV with validation rows.")
    p.add_argument("--sales", type=Path, default=Path("data/sales_train_evaluation.csv"), help="Ground truth sales file.")
    p.add_argument("--cache_dir", type=Path, default=Path("hierarchy_cache"), help="Directory with S.npz, nodes.json, bottom_ids.json.")
    return p.parse_args()


def load_submission(sub_path: Path, bottom_ids: List[str]) -> np.ndarray:
    f_cols = [f"F{i}" for i in range(1, 29)]
    df = pd.read_csv(sub_path)
    df = df[df["id"].str.endswith("_validation")].copy()
    df["base_id"] = df["id"].str.replace("_validation", "", regex=False)
    df = df.set_index("base_id")
    missing = [bid for bid in bottom_ids if bid not in df.index]
    if missing:
        raise ValueError(f"Submission missing {len(missing)} validation ids, e.g. {missing[:5]}")
    sub_mat = df.loc[bottom_ids, f_cols].to_numpy(dtype=np.float64)  # (n_bottom, 28)
    return sub_mat


def load_ground_truth(sales_path: Path, bottom_ids: List[str]) -> np.ndarray:
    day_cols = [f"d_{1913 + i}" for i in range(1, 29)]
    gt = pd.read_csv(sales_path)
    gt["base_id"] = (
        gt["id"]
        .str.replace("_validation", "", regex=False)
        .str.replace("_evaluation", "", regex=False)
    )
    gt = gt.set_index("base_id")
    gt_mat = gt.loc[bottom_ids, day_cols].to_numpy(dtype=np.float64)
    return gt_mat


def mae_rmse(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    diff = pred - gt
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def main() -> None:
    args = parse_args()
    cache = args.cache_dir
    bottom_ids = json.loads((cache / "bottom_ids.json").read_text())
    nodes = json.loads((cache / "nodes.json").read_text())
    S = sparse.load_npz(cache / "S.npz")  # (n_nodes x n_bottom)

    pred_bottom = load_submission(args.submission, bottom_ids)  # (n_bottom, 28)
    gt_bottom = load_ground_truth(args.sales, bottom_ids)  # (n_bottom, 28)

    # Aggregate to nodes
    pred_nodes = S @ pred_bottom  # (n_nodes, 28)
    gt_nodes = S @ gt_bottom

    # Group nodes by level
    level_to_indices: Dict[str, List[int]] = {}
    for idx, n in enumerate(nodes):
        level_to_indices.setdefault(n["level"], []).append(idx)

    print(f"Validation d_1914–d_1941, bottom series {len(bottom_ids)}, nodes {len(nodes)}")
    for level, idxs in level_to_indices.items():
        p = pred_nodes[idxs, :]
        g = gt_nodes[idxs, :]
        metrics = mae_rmse(p, g)
        print(f"{level:12s} | MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} (rows={len(idxs)})")


if __name__ == "__main__":
    main()
