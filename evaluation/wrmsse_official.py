"""
Official-style WRMSSE evaluator for M5.

- Precomputes and caches:
    * hierarchy keys for 12 levels
    * scale (RMSSE denominator) using d_1–d_1913
    * weights using last 28 days d_1886–d_1913
- compute_wrmsse(y_true, y_pred) expects long data with columns:
    id (string), d (int day number), sales (float).
  Returns scalar WRMSSE and per-level breakdown.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
CACHE_DIR = Path("future_finaldata/wrmsse_cache")
SALES_FILE = DATA_DIR / "sales_train_validation.csv"

TRAIN_END = 1913
WEIGHT_SPAN = 28
WEIGHT_START = TRAIN_END - WEIGHT_SPAN + 1  # 1886


def _gen_level_keys(meta: pd.DataFrame) -> Dict[int, pd.Series]:
    idx = meta["id"]
    keys = {
        1: pd.Series(["all"] * len(meta), index=idx),
        2: pd.Series(meta["state_id"].values, index=idx),
        3: pd.Series(meta["store_id"].values, index=idx),
        4: pd.Series(meta["cat_id"].values, index=idx),
        5: pd.Series(meta["dept_id"].values, index=idx),
        6: pd.Series((meta["state_id"] + "_" + meta["cat_id"]).values, index=idx),
        7: pd.Series((meta["state_id"] + "_" + meta["dept_id"]).values, index=idx),
        8: pd.Series((meta["store_id"] + "_" + meta["cat_id"]).values, index=idx),
        9: pd.Series((meta["store_id"] + "_" + meta["dept_id"]).values, index=idx),
        10: pd.Series(meta["item_id"].values, index=idx),
        11: pd.Series((meta["state_id"] + "_" + meta["item_id"]).values, index=idx),
        12: pd.Series((meta["store_id"] + "_" + meta["item_id"]).values, index=idx),
    }
    return keys


class WRMSSEEvaluator:
    def __init__(self, sales_file: Path = SALES_FILE, cache_dir: Path = CACHE_DIR):
        self.sales_file = sales_file
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta = None
        self.level_keys = None
        self.scales = None
        self.weights = None
        self._load_or_build()

    def _load_or_build(self):
        meta_p = self.cache_dir / "meta.pkl"
        keys_p = self.cache_dir / "level_keys.pkl"
        scales_p = self.cache_dir / "scales.pkl"
        weights_p = self.cache_dir / "weights.pkl"

        if meta_p.exists() and keys_p.exists() and scales_p.exists() and weights_p.exists():
            self.meta = pd.read_pickle(meta_p)
            self.level_keys = pd.read_pickle(keys_p)
            self.scales = pd.read_pickle(scales_p)
            self.weights = pd.read_pickle(weights_p)
            return

        wide = pd.read_csv(self.sales_file)
        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        self.meta = wide[id_cols].copy()
        day_cols = [f"d_{d}" for d in range(1, TRAIN_END + 1)]
        long = wide[id_cols + day_cols].melt(id_vars=id_cols, var_name="d", value_name="sales")
        long["d_num"] = long["d"].str.replace("d_", "").astype(int)

        self.level_keys = _gen_level_keys(self.meta)

        self.scales = {}
        for level, key_series in self.level_keys.items():
            tmp = long.copy()
            tmp["series"] = tmp["id"].map(key_series)
            tmp = tmp[tmp["d_num"] <= TRAIN_END]
            tmp = tmp.sort_values(["series", "d_num"])
            tmp["diff"] = tmp.groupby("series")["sales"].diff()
            denom = tmp.groupby("series")["diff"].apply(
                lambda x: np.mean(np.square(x.dropna())) if x.dropna().size > 0 else 0.0
            )
            denom = denom.replace(0, 1e-6)
            self.scales[level] = denom

        w_long = long[(long["d_num"] >= WEIGHT_START) & (long["d_num"] <= TRAIN_END)]
        total = w_long["sales"].sum()
        self.weights = {}
        for level, key_series in self.level_keys.items():
            tmp = w_long.copy()
            tmp["series"] = tmp["id"].map(key_series)
            sales_sum = tmp.groupby("series")["sales"].sum()
            self.weights[level] = sales_sum / (total if total != 0 else 1e-6)

        self.meta.to_pickle(meta_p)
        pd.to_pickle(self.level_keys, keys_p)
        pd.to_pickle(self.scales, scales_p)
        pd.to_pickle(self.weights, weights_p)

    def compute_wrmsse(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Tuple[float, Dict[int, float]]:
        y_true_proc = self._normalize_long_df(y_true)
        y_pred_proc = self._normalize_long_df(y_pred)
        y_true_proc = y_true_proc[y_true_proc["id"].isin(self.meta["id"])]
        y_pred_proc = y_pred_proc[y_pred_proc["id"].isin(self.meta["id"])]
        y_true_proc["id"] = y_true_proc["id"].astype(str)
        y_pred_proc["id"] = y_pred_proc["id"].astype(str)

        df = y_true_proc[["id", "d", "sales"]].rename(columns={"sales": "y_true"}).merge(
            y_pred_proc[["id", "d", "sales"]].rename(columns={"sales": "y_pred"}), on=["id", "d"], how="left"
        )
        df["y_pred"] = df["y_pred"].fillna(0.0)

        per_level = {}
        total_wrmsse = 0.0
        for level, key_series in self.level_keys.items():
            tmp = df.copy()
            tmp["series"] = tmp["id"].map(key_series)
            agg = tmp.groupby(["series", "d"], observed=True)[["y_true", "y_pred"]].sum()
            agg = agg.sort_index()
            se = (agg["y_pred"] - agg["y_true"]) ** 2
            numer = se.groupby("series").mean()
            scale = self.scales[level].reindex(numer.index).fillna(1e-6)
            rmsse = np.sqrt(numer / scale)
            weight = self.weights[level].reindex(rmsse.index).fillna(0.0)
            wrmsse_level = (rmsse * weight).sum()
            per_level[level] = wrmsse_level
            total_wrmsse += wrmsse_level
        return total_wrmsse, per_level

    @staticmethod
    def _normalize_long_df(df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        if "d" not in tmp.columns:
            raise ValueError("WRMSSE data must have column 'd' (int day index).")
        if tmp["d"].dtype == object:
            tmp["d"] = (
                tmp["d"]
                .astype(str)
                .str.replace("d_", "", regex=False)
                .str.replace("D_", "", regex=False)
                .astype(int)
            )
        return tmp

    @staticmethod
    def wide_to_long(preds_wide: pd.DataFrame, start_day: int = 1914) -> pd.DataFrame:
        rows = []
        value_cols = [c for c in preds_wide.columns if c.startswith("F")]
        for col in value_cols:
            f_idx = int(col.lstrip("F"))
            d = start_day + f_idx - 1
            temp = preds_wide[["id", col]].copy()
            temp = temp.rename(columns={col: "sales"})
            temp["d"] = d
            rows.append(temp[["id", "d", "sales"]])
        if not rows:
            return pd.DataFrame(columns=["id", "d", "sales"])
        return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    print("WRMSSEEvaluator ready; use compute_wrmsse(y_true_long, y_pred_long).")
