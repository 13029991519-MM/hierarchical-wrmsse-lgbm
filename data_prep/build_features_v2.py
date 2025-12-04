"""
Build processed_v2 features with long-season lags/rolls and price stats for M5 data.
Outputs per-store CSVs under processed_v2/processed_{store}.csv

Added features (beyond baseline-style):
- Lags: 91, 182, 365
- Rolling means/stds: 91, 182
- Price stats: sell_price_roll_7/28/56, price_ratio (curr/roll28), price_diff (curr - roll28),
  price_std_28, is_discounted (price < 0.97 * roll28)

Existing shorter lags/rolls (1/7/14/28/56/84) are also computed.

Usage (PowerShell):
python build_features_v2.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import argparse

SALES_PATH = Path("data/sales_train_evaluation.csv")
CAL_PATH = Path("data/calendar.csv")
PRICE_PATH = Path("data/sell_prices.csv")
OUT_DIR = Path("processed_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lags/rolls
SHORT_LAGS = [1, 7, 14, 28, 56, 84]
LONG_LAGS = [91, 182, 365]
ROLL_WINS = [7, 14, 28, 56, 91, 182]

# Optional: limit history to last N days to reduce memory (None = full); here full 1941 days
HISTORY_WINDOW = None


def build_cyclic(cal: pd.DataFrame) -> pd.DataFrame:
    cal = cal.copy()
    cal["d_int"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal["quarter"] = ((cal["month"] - 1) // 3 + 1).astype(int)
    cal["wday_sin"] = np.sin(2 * np.pi * cal["wday"] / 7)
    cal["wday_cos"] = np.cos(2 * np.pi * cal["wday"] / 7)
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12)
    cal["quarter_sin"] = np.sin(2 * np.pi * cal["quarter"] / 4)
    cal["quarter_cos"] = np.cos(2 * np.pi * cal["quarter"] / 4)
    # holiday flag
    cal["IsHoliday"] = ((cal["event_name_1"].notna()) | (cal["event_name_2"].notna())).astype(np.int8)
    return cal


def compute_lag_roll_series(series: pd.Series) -> pd.DataFrame:
    """Given a long series (sorted by d_int) compute lags/rolls inplace, return DataFrame."""
    df = pd.DataFrame({"sales": series})
    for lag in SHORT_LAGS + LONG_LAGS:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    for win in ROLL_WINS:
        df[f"rolling_mean_{win}"] = df["sales"].shift(1).rolling(win, min_periods=1).mean()
        df[f"rolling_std_{win}"] = df["sales"].shift(1).rolling(win, min_periods=1).std()
    return df


def process_store(store: str, sales: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame) -> None:
    df = sales[sales["id"].str.contains(f"_{store}_")].copy()
    if df.empty:
        print(f"Skip store {store}: no rows")
        return
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]
    if HISTORY_WINDOW:
        day_cols = [c for c in day_cols if int(c.split("_")[1]) > (1913 - HISTORY_WINDOW)]

    # melt to long to avoid huge intermediate arrays
    long_df = df[id_cols + day_cols].melt(id_vars=id_cols, var_name="d", value_name="sales")
    long_df["d_int"] = long_df["d"].str.replace("d_", "", regex=False).astype(int)
    long_df = long_df.sort_values(["id", "d_int"])
    # compute lags/rolls per id
    def add_feats(group: pd.DataFrame) -> pd.DataFrame:
        feats = compute_lag_roll_series(group["sales"])
        group = group.reset_index(drop=True)
        group = pd.concat([group, feats.reset_index(drop=True)], axis=1)
        return group

    long_df = long_df.groupby("id", group_keys=False).apply(add_feats)
    long_df = long_df.merge(cal, on="d_int", how="left")

    # price features: merge on (item_id, wm_yr_wk)
    prices_store = prices[prices["store_id"] == store].copy()
    long_df = long_df.merge(
        prices_store[["item_id", "wm_yr_wk", "sell_price"]],
        on=["item_id", "wm_yr_wk"],
        how="left",
    )
    long_df["sell_price"] = long_df["sell_price"].fillna(0).astype(np.float32)
    long_df.sort_values(["item_id", "d_int"], inplace=True)
    long_df["sell_price_roll_7"] = long_df.groupby("item_id")["sell_price"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    long_df["sell_price_roll_28"] = long_df.groupby("item_id")["sell_price"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    long_df["sell_price_roll_56"] = long_df.groupby("item_id")["sell_price"].transform(lambda s: s.shift(1).rolling(56, min_periods=1).mean())
    long_df["sell_price_std_28"] = long_df.groupby("item_id")["sell_price"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).std())
    long_df["price_ratio"] = long_df["sell_price"] / long_df["sell_price_roll_28"].replace(0, np.nan)
    long_df["price_diff"] = long_df["sell_price"] - long_df["sell_price_roll_28"]
    long_df["is_discounted"] = (long_df["price_diff"] < 0).astype(np.int8)
    long_df.fillna(0, inplace=True)

    out_path = OUT_DIR / f"processed_{store}.csv"
    long_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(long_df):,} rows")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build processed_v2 features with long lags/rolls and price stats.")
    p.add_argument("--stores", type=str, default="", help="Comma-separated store_ids to process (e.g., CA_1,TX_1). Empty = all.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cal = build_cyclic(pd.read_csv(CAL_PATH))
    prices = pd.read_csv(PRICE_PATH)
    sales = pd.read_csv(SALES_PATH)
    all_stores = sales["id"].str.extract(r".*_(\w+_\d)_")[0].unique()
    if args.stores.strip():
        stores = [s.strip() for s in args.stores.split(",") if s.strip()]
    else:
        stores = all_stores
    for store in stores:
        process_store(store, sales, cal, prices)


if __name__ == "__main__":
    main()
