"""
Visualize summary WRMSSE decisions and blend submissions using learned weights.

Usage:
    python visualize_and_blend.py \
        --summary weight_v2/summary_delay120_v2.json \
        --weights weight_v2/delay_120_weight_v2.json \
        --base future_finaldata/submission_with_val.csv \
        --alt future_finaldata/submission_with_val_cmodel.csv \
        --out blended/submission_with_val_blend.csv
        --fig blended/wrmsse_scatter.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

F_COLS = [f"F{i}" for i in range(1, 29)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize summary WRMSSE and blend submissions.")
    parser.add_argument("--summary", type=Path, default=Path("weight_v2/summary_delay120_v2.json"))
    parser.add_argument("--weights", type=Path, default=Path("weight_v2/delay_120_weight_v2.json"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--alt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("future_finaldata/submission_with_val_blended.csv"))
    parser.add_argument("--fig", type=Path, default=Path("blended/wrmsse_scatter.png"))
    return parser.parse_args()


def store_from_id(idx: str) -> str:
    parts = idx.split("_")
    if len(parts) >= 3:
        return parts[-1]
    return "unknown"


def load_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    weights = {}
    for store, entry in data.items():
        weights[store] = entry.get("w_alt", 0.0)
    return weights


def blend_submissions(base_path: Path, alt_path: Path, weights: dict[str, float], out_path: Path) -> None:
    base = pd.read_csv(base_path)
    alt = pd.read_csv(alt_path)
    alt_idx = alt.set_index("id")
    def blend_row(row):
        store = store_from_id(row["id"])
        w_alt = weights.get(store, 0.0)
        if w_alt <= 0.0:
            return row[F_COLS].values
        alt_row = alt_idx.loc[row["id"], F_COLS]
        blended = (1 - w_alt) * row[F_COLS].values + w_alt * alt_row.values
        return blended
    blended = base.copy()
    blended[F_COLS] = base.apply(lambda r: blend_row(r), axis=1, result_type="expand")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(out_path, index=False)
    print(f"Blended submission saved to {out_path}")


def plot_summary(summary_path: Path, fig_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for entry in data:
        main_wrmsse = entry.get("main_wrmsse")
        c_wrmsse = entry.get("c_wrmsse")
        decision = entry.get("auto_decision", "unknown")
        if main_wrmsse is None or c_wrmsse is None:
            continue
        records.append((main_wrmsse, c_wrmsse, decision))
    if not records:
        print("No WRMSSE entries found for scatter.")
        return
    df = pd.DataFrame(records, columns=["main_wrmsse", "c_wrmsse", "decision"])
    colors = {"allow": "tab:green", "neutral": "tab:orange", "ban": "tab:red"}
    plt.figure(figsize=(5, 5))
    for decision, group in df.groupby("decision"):
        plt.scatter(
            group["main_wrmsse"],
            group["c_wrmsse"],
            label=decision,
            color=colors.get(decision, "tab:gray"),
            alpha=0.7,
            s=30,
        )
    lim = [min(df["main_wrmsse"].min(), df["c_wrmsse"].min()), max(df["main_wrmsse"].max(), df["c_wrmsse"].max())]
    plt.plot(lim, lim, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Main model WRMSSE (avg windows)")
    plt.ylabel("C-model WRMSSE (avg windows)")
    plt.title("Store-level WRMSSE comparison")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"WRMSSE scatter saved to {fig_path}")


def main() -> None:
    args = parse_args()
    plot_summary(args.summary, args.fig)
    weights = load_weights(args.weights)
    blend_submissions(args.base, args.alt, weights, args.out)


if __name__ == "__main__":
    main()
