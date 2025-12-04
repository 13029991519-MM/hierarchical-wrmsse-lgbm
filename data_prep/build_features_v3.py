"""
Build per-store feature files with long-season lags/rolls and price stats, written to processed_v3/.
Default仅处理代表店 CA_1, TX_1, WI_1，可通过 --stores 覆盖。

特征要点：
- 滞后：1,7,14,28,56,84,91,182,365
- 滚动均值/方差（shift 后）：7,14,28,56,91,182
- 价格：按 item_id 合并 sell_prices，计算 roll_7/28/56、std_28、ratio/diff、is_discounted
- 保留原日历特征（wday/month/year/wm_yr_wk/snap/event），生成 sin/cos

输出：processed_v3/processed_{store}.parquet （避免覆盖旧数据）

Usage (PowerShell):
python build_features_v3.py               # 仅 CA_1,TX_1,WI_1
python build_features_v3.py --stores CA_1,TX_1   # 指定门店
python build_features_v3.py --history_window 1000  # 仅保留最近 N 天
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

SALES_PATH = Path("data/sales_train_evaluation.csv")
CAL_PATH = Path("data/calendar.csv")
PRICE_PATH = Path("data/sell_prices.csv")
OUT_DIR = Path("processed_v3")

DEFAULT_STORES = ["CA_1", "TX_1", "WI_1"]
LAGS = [1, 7, 14, 28, 56, 84, 91, 182, 365]
ROLL_WINS = [7, 14, 28, 56, 91, 182]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build features v3 with long lags/rolls and price stats.")
    p.add_argument("--stores", type=str, default="", help="Comma-separated store_ids (e.g., CA_1,TX_1). Empty uses defaults.")
    p.add_argument("--history_window", type=int, default=None, help="Keep only last N days; None=full 1941.")
    return p.parse_args()


def build_calendar() -> pd.DataFrame:
    cal = pd.read_csv(CAL_PATH)
    cal["d_int"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal["quarter"] = ((cal["month"] - 1) // 3 + 1).astype(int)
    cal["wday_sin"] = np.sin(2 * np.pi * cal["wday"] / 7)
    cal["wday_cos"] = np.cos(2 * np.pi * cal["wday"] / 7)
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12)
    cal["quarter_sin"] = np.sin(2 * np.pi * cal["quarter"] / 4)
    cal["quarter_cos"] = np.cos(2 * np.pi * cal["quarter"] / 4)
    cal["IsHoliday"] = ((cal["event_name_1"].notna()) | (cal["event_name_2"].notna())).astype(np.int8)
    return cal


def add_sales_feats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("d_int").reset_index(drop=True)
    for lag in LAGS:
        g[f"lag_{lag}"] = g["sales"].shift(lag)
    for w in ROLL_WINS:
        g[f"roll_mean_{w}"] = g["sales"].shift(1).rolling(w, min_periods=1).mean()
        g[f"roll_std_{w}"] = g["sales"].shift(1).rolling(w, min_periods=1).std()
    return g


def process_store(store: str, sales: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame, history_window: int | None) -> None:
    df = sales[sales["id"].str.contains(f"_{store}_")].copy()
    if df.empty:
        print(f"Skip {store}: no rows")
        return
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]
    if history_window:
        max_d = max(int(c.split("_")[1]) for c in day_cols)
        day_cols = [c for c in day_cols if int(c.split("_")[1]) >= max_d - history_window + 1]
    long_df = df[id_cols + day_cols].melt(id_vars=id_cols, var_name="d", value_name="sales")
    long_df["d_int"] = long_df["d"].str.replace("d_", "", regex=False).astype(int)
    long_df = long_df.sort_values(["id", "d_int"])
    long_df = long_df.groupby("id", group_keys=False).apply(add_sales_feats)
    long_df = long_df.merge(cal, on="d_int", how="left")

    prices_store = prices[prices["store_id"] == store].copy()
    long_df = long_df.merge(prices_store[["item_id", "wm_yr_wk", "sell_price"]], on=["item_id", "wm_yr_wk"], how="left")
    long_df["sell_price"] = long_df["sell_price"].fillna(0).astype(np.float32)
    long_df.sort_values(["item_id", "d_int"], inplace=True)
    long_df["price_roll_7"] = long_df.groupby("item_id")["sell_price"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    long_df["price_roll_28"] = long_df.groupby("item_id")["sell_price"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    long_df["price_roll_56"] = long_df.groupby("item_id")["sell_price"].transform(lambda s: s.shift(1).rolling(56, min_periods=1).mean())
    long_df["price_std_28"] = long_df.groupby("item_id")["sell_price"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).std())
    long_df["price_ratio"] = long_df["sell_price"] / long_df["price_roll_28"].replace(0, np.nan)
    long_df["price_diff"] = long_df["sell_price"] - long_df["price_roll_28"]
    long_df["is_discounted"] = (long_df["price_diff"] < 0).astype(np.int8)
    long_df.fillna(0, inplace=True)

    # ensure string/object columns are string dtype to avoid Arrow inference issues
    obj_cols = long_df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        long_df[col] = long_df[col].astype("string")

    out_path = OUT_DIR / f"processed_{store}.parquet"
    long_df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} with {len(long_df):,} rows")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal = build_calendar()
    prices = pd.read_csv(PRICE_PATH)
    sales = pd.read_csv(SALES_PATH)
    all_stores = sales["id"].str.extract(r".*_(\w+_\d)_")[0].unique()
    stores = [s.strip() for s in args.stores.split(",") if s.strip()] if args.stores.strip() else DEFAULT_STORES
    for store in stores:
        process_store(store, sales, cal, prices, args.history_window)
        # free memory between stores
        import gc
        gc.collect()


if __name__ == "__main__":
    main()
