"""
Compare WRMSSE across two validation windows using the shared evaluator.

Usage:
    python evaluate_wrmsse_windows.py \
        --pred_current future_finaldata/submission_with_val.csv \
        --pred_prev future_finaldata/submission_with_val_prev.csv \
        --out_dir wrmsse_windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wrmsse_official import WRMSSEEvaluator, SALES_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and compare WRMSSE for two windows.")
    parser.add_argument("--pred_current", type=Path, required=True, help="Submission wide file covering d_1914-1941")
    parser.add_argument("--pred_prev", type=Path, required=True, help="Submission wide file covering d_1886-1913")
    parser.add_argument("--out_dir", type=Path, default=Path("wrmsse_windows"), help="Directory for CSV outputs")
    return parser.parse_args()


def wide_to_long(df: pd.DataFrame, start_day: int) -> pd.DataFrame:
    return WRMSSEEvaluator.wide_to_long(df, start_day=start_day)


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path} missing id column")
    return df


def truth_long(start_day: int, end_day: int) -> pd.DataFrame:
    wide = pd.read_csv(SALES_FILE)
    day_cols = [f"d_{d}" for d in range(start_day, end_day + 1)]
    id_cols = ["id"]
    long = wide[id_cols + day_cols].melt(id_vars=["id"], var_name="d", value_name="sales")
    long["d"] = long["d"].str.replace("d_", "").astype(int)
    return long[long["d"].between(start_day, end_day)]


def store_wrmsse(ev: WRMSSEEvaluator, truth: pd.DataFrame, pred: pd.DataFrame) -> pd.Series:
    truth_proc = ev._normalize_long_df(truth)
    pred_proc = ev._normalize_long_df(pred)
    merged = (
        truth_proc.rename(columns={"sales": "y_true"})
        .merge(pred_proc.rename(columns={"sales": "y_pred"}), on=["id", "d"], how="left")
        .fillna(0.0)
    )
    key_series = ev.level_keys[3]
    tmp = merged.copy()
    tmp["series"] = tmp["id"].map(key_series)
    agg = tmp.groupby(["series", "d"], observed=True)[["y_true", "y_pred"]].sum()
    se = (agg["y_pred"] - agg["y_true"]) ** 2
    numer = se.groupby("series").mean()
    scale = ev.scales[3].reindex(numer.index).fillna(1e-6)
    rmsse = (numer / scale).clip(min=0).apply(np.sqrt)
    weight = ev.weights[3].reindex(rmsse.index).fillna(0.0)
    store_scores = (rmsse * weight).groupby(key_series).sum()
    return store_scores


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    ev = WRMSSEEvaluator()

    current_df = load_preds(args.pred_current)
    prev_df = load_preds(args.pred_prev)

    truth_curr = truth_long(1914, 1941)
    truth_prev = truth_long(1886, 1913)

    curr_long = wide_to_long(current_df, start_day=1914)
    prev_long = wide_to_long(prev_df, start_day=1886)

    curr_score, curr_levels = ev.compute_wrmsse(truth_curr, curr_long)
    prev_score, prev_levels = ev.compute_wrmsse(truth_prev, prev_long)

    store_curr = store_wrmsse(ev, truth_curr, curr_long)
    store_prev = store_wrmsse(ev, truth_prev, prev_long)

    comp = (
        pd.DataFrame(
            {"wrmsse_prev": store_prev, "wrmsse_current": store_curr}
        )
        .dropna(how="all")
        .fillna(0.0)
    )
    comp["wrmsse_delta"] = comp["wrmsse_current"] - comp["wrmsse_prev"]
    def _decision(row: pd.Series) -> str:
        delta = row["wrmsse_delta"]
        if row["wrmsse_current"] < row["wrmsse_prev"] - 0.01 and delta < -0.01:
            return "allow"
        if abs(delta) <= 0.01:
            return "neutral"
        return "ban"

    comp["auto_decision"] = comp.apply(_decision, axis=1)
    comp.to_csv(out_dir / "store_wrmsse_comparison.csv")

    summary = {
        "wrmsse_1886_1913": prev_score,
        "wrmsse_1914_1941": curr_score,
        "store_count": len(comp),
    }
    decision_counts = Counter(comp["auto_decision"])
    summary["window_decisions"] = dict(decision_counts)
    summary["wrmsse_delta_mean"] = float(comp["wrmsse_delta"].mean())

    scatter_path = out_dir / "window_wrmsse_scatter.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = {"allow": "tab:green", "neutral": "tab:orange", "ban": "tab:red"}
    for decision, group in comp.groupby("auto_decision"):
        ax.scatter(
            group["wrmsse_prev"],
            group["wrmsse_current"],
            s=8,
            label=decision,
            c=colors.get(decision, "tab:grey"),
            alpha=0.7,
            edgecolors="none",
        )
    lim = [
        min(comp["wrmsse_prev"].min(), comp["wrmsse_current"].min()),
        max(comp["wrmsse_prev"].max(), comp["wrmsse_current"].max()),
    ]
    ax.plot(lim, lim, color="black", linewidth=0.8, linestyle="--", label="y=x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("WRMSSE prev window (1886-1913)")
    ax.set_ylabel("WRMSSE current window (1914-1941)")
    ax.set_title("Store-level WRMSSE stability")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    with open(out_dir / "window_wrmsse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("WRMSSE summary:")
    print(summary)
    print(f"Details saved to {out_dir / 'store_wrmsse_comparison.csv'} and {scatter_path}")


if __name__ == "__main__":
    main()
