"""
Plot summary-level WRMSSE decisions and blend weight distribution.

Usage:
    python summary_weights_viz.py \
        --summary weight_v2/summary_delay120_v2.json \
        --weights weight_v2/delay_120_weight_v2.json \
        --out_dir blended
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize summary/weight stats.")
    parser.add_argument(
        "--summary", type=Path, default=Path("weight_v2/summary_delay120_v2.json")
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("weight_v2/delay_120_weight_v2.json")
    )
    parser.add_argument("--out_dir", type=Path, default=Path("blended"))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary.exists():
        print(f"Summary file {args.summary} missing. Skipping.")
        return
    summary = pd.read_json(args.summary)
    summary["wrmsse_delta"] = summary["wrmsse_delta"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    summary["auto_decision"].value_counts().plot(
        kind="bar", ax=axes[0], color=["tab:green", "tab:orange", "tab:red"]
    )
    axes[0].set_title("Auto decisions")
    axes[0].set_xlabel("Decision")
    axes[0].set_ylabel("Store count")

    summary.plot(
        kind="scatter",
        x="main_wrmsse",
        y="c_wrmsse",
        ax=axes[1],
        c="wrmsse_delta",
        cmap="coolwarm",
        colorbar=True,
        s=40,
    )
    axes[1].set_title("WRMSSE comparison (main vs C)")
    axes[1].set_xlabel("main WRMSSE")
    axes[1].set_ylabel("C WRMSSE")
    lim = [
        min(summary["main_wrmsse"].min(), summary["c_wrmsse"].min()),
        max(summary["main_wrmsse"].max(), summary["c_wrmsse"].max()),
    ]
    axes[1].plot(lim, lim, linestyle="--", color="black", linewidth=0.8)

    fig.tight_layout()
    scatter_path = out_dir / "summary_decisions.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"Saved summary scatter to {scatter_path}")

    if not args.weights.exists():
        print(f"Weight file {args.weights} missing. Plotting done.")
        return
    weights = pd.read_json(args.weights)
    weight_vals = [entry.get("w_alt", 0.0) for entry in weights.values()]
    plt.figure(figsize=(5, 3))
    plt.hist(weight_vals, bins=10, color="tab:blue", edgecolor="black")
    plt.title("Blend weight distribution")
    plt.xlabel("w_alt")
    plt.ylabel("Store count")
    hist_path = out_dir / "blend_weight_hist.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    print(f"Saved weight histogram to {hist_path}")


if __name__ == "__main__":
    main()
