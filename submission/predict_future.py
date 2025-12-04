"""
Iterative 28-day forecast using trained LGBM models and stored blend weights.
This is a simplified scaffold: it trains on full history (d_11941) per store,
then rolls forward day by day to generate future features and predictions.

Inputs:
- newfinaldata/processed_*.csv : historical features up to d_1941 (per store)
- data/calendar.csv, data/sell_prices.csv : calendar/price info for future days
- weight/blend_weights_auto.json : per-store blend weights; stores in NO_BLEND_STORES use single model

Output:
- submission.csv saved under future_finaldata/ (28-day forecasts, one row per id)

Note: This is a lightweight implementation intended to give a working pipeline;
for production, you may want to optimize memory and add more safety checks.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = Path("newfinaldata")
CAL_PATH = Path("data/calendar.csv")
PRICE_PATH = Path("data/sell_prices.csv")
BLEND_PATH = Path("weight/blend_weights_auto.json")
ALT_BLEND_PATH = Path("weight_v2/delay_120_weight_v2.json")
OUT_DIR = Path("future_finaldata")
OUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative forecast for evaluation submission (only TX_1/TX_2 by default).")
    parser.add_argument(
        "--stores",
        type=str,
        default="TX_1,TX_2",
        help="Comma-separated stores to forecast (default: TX_1,TX_2).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("future_finaldata/submission_tx12_cmodel.csv"),
        help="Path for the new submission output (won't override submission.csv).",
    )
    return parser.parse_args()

# Columns (must match training)
NUM_SCALED = [
    "sell_price_scaled",
    "baseline_price_scaled",
    "discount_scaled",
    "promo_intensity_scaled",
    "lag_7_scaled",
    "lag_30_scaled",
    "rolling_mean_7_scaled",
    "rolling_mean_30_scaled",
    "rolling_std_7_scaled",
    "rolling_std_30_scaled",
    "lag_1_scaled",
    "lag_14_scaled",
    "lag_28_scaled",
    "lag_56_scaled",
    "lag_84_scaled",
    "rolling_mean_14_scaled",
    "rolling_mean_28_scaled",
    "rolling_mean_56_scaled",
    "rolling_mean_84_scaled",
    "rolling_std_14_scaled",
    "rolling_std_28_scaled",
    "rolling_std_56_scaled",
    "rolling_std_84_scaled",
    "price_ratio_scaled",
    "discount_pct_scaled",
    "snap_wday_scaled",
    "promo_holiday_scaled",
    "promo_wday_sin_scaled",
    "promo_wday_cos_scaled",
    "discount_snap_scaled",
    "item_freq_scaled",
    "id_freq_scaled",
    "item_te_scaled",
    "id_te_scaled",
    # rolling median/min/max (scaled)
    "rolling_median_7_scaled",
    "rolling_median_14_scaled",
    "rolling_median_28_scaled",
    "rolling_median_30_scaled",
    "rolling_median_56_scaled",
    "rolling_median_84_scaled",
    "rolling_min_7_scaled",
    "rolling_min_14_scaled",
    "rolling_min_28_scaled",
    "rolling_min_30_scaled",
    "rolling_min_56_scaled",
    "rolling_min_84_scaled",
    "rolling_max_7_scaled",
    "rolling_max_14_scaled",
    "rolling_max_28_scaled",
    "rolling_max_30_scaled",
    "rolling_max_56_scaled",
    "rolling_max_84_scaled",
    # price momentum / z-score (scaled)
    "sell_price_week_chg_scaled",
    "sell_price_month_chg_scaled",
    "sell_price_z28_scaled",
    "discount_week_chg_scaled",
    "discount_month_chg_scaled",
    "discount_z28_scaled",
    # streaks / holiday distance (scaled)
    "promo_streak_scaled",
    "holiday_streak_scaled",
    "days_since_holiday_scaled",
    "days_until_holiday_scaled",
]
# raw numeric columns used for scaling/lag creation
RAW_BASE_NUM = [
    "sell_price",
    "baseline_price",
    "discount",
    "promo_intensity",
    "price_ratio",
    "discount_pct",
    "snap_wday",
    "promo_holiday",
    "promo_wday_sin",
    "promo_wday_cos",
    "discount_snap",
    "promo_streak",
    "holiday_streak",
    "days_since_holiday",
    "days_until_holiday",
    "sell_price_week_chg",
    "sell_price_month_chg",
    "sell_price_z28",
    "discount_week_chg",
    "discount_month_chg",
    "discount_z28",
]
RAW_LAG_COLS = [f"lag_{k}" for k in [1, 7, 14, 28, 30, 56, 84]]
RAW_ROLL_MEAN = [f"rolling_mean_{k}" for k in [7, 14, 28, 30, 56, 84]]
RAW_ROLL_STD = [f"rolling_std_{k}" for k in [7, 14, 28, 30, 56, 84]]
RAW_ROLL_MED = [f"rolling_median_{k}" for k in [7, 14, 28, 30, 56, 84]]
RAW_ROLL_MIN = [f"rolling_min_{k}" for k in [7, 14, 28, 30, 56, 84]]
RAW_ROLL_MAX = [f"rolling_max_{k}" for k in [7, 14, 28, 30, 56, 84]]
CYCLIC = ["wday_sin", "wday_cos", "month_sin", "month_cos", "quarter_sin", "quarter_cos"]
BIN_COLS = ["snap_CA", "snap_TX", "snap_WI", "IsHoliday", "IsPromotion"]
CAT_COLS = [
    "state_id",
    "store_id",
    "cat_id",
    "dept_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "item_id",
    "id",
]
TARGET_COL = "sales"
NO_BLEND_STORES = set()  # 若全店单模， main 覆盖
# Parallel flag; set False if memory is tight
PARALLEL = False
MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))
# History window (days) to keep per store for faster inference; None=all
HISTORY_WINDOW = None  # use full history; set to int (e.g., 500) to truncate


def load_blend_weights() -> Dict[str, dict]:
    path = BLEND_PATH if BLEND_PATH.exists() else ALT_BLEND_PATH
    if not path.exists():
        raise FileNotFoundError(f"Blend weights not found at {BLEND_PATH} or {ALT_BLEND_PATH}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_store_history(store: str) -> pd.DataFrame:
    path = DATA_DIR / f"processed_{store}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = (
        NUM_SCALED
        + RAW_BASE_NUM
        + RAW_LAG_COLS
        + RAW_ROLL_MEAN
        + RAW_ROLL_STD
        + RAW_ROLL_MED
        + RAW_ROLL_MIN
        + RAW_ROLL_MAX
        + CYCLIC
        + BIN_COLS
        + CAT_COLS
        + [TARGET_COL, "d"]
    )
    usecols = [c for c in wanted if c in cols]
    dtypes = {c: "float32" for c in NUM_SCALED if c in usecols}
    dtypes.update({c: "float32" for c in RAW_BASE_NUM + RAW_LAG_COLS + RAW_ROLL_MEAN + RAW_ROLL_STD + RAW_ROLL_MED + RAW_ROLL_MIN + RAW_ROLL_MAX if c in usecols})
    dtypes.update({c: "int8" for c in BIN_COLS if c in usecols})
    dtypes.update({c: "category" for c in CAT_COLS if c in usecols})
    if TARGET_COL in usecols:
        dtypes[TARGET_COL] = "float32"
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes, engine="pyarrow")
    for col in wanted:
        if col not in df.columns:
            if col in BIN_COLS:
                df[col] = np.int8(0)
            elif col in CAT_COLS:
                df[col] = pd.Series(["Unknown"] * len(df), dtype="category")
            else:
                df[col] = np.float32(0.0)
    num_cols = [c for c in wanted if c not in CAT_COLS + BIN_COLS + ["d"]]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32").fillna(0)
    for col in BIN_COLS:
        df[col] = df[col].astype("int8")
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    df["d_int"] = df["d"].str.replace("d_", "", regex=False).astype(int)
    if any(c not in df.columns for c in ["item_freq_scaled", "id_freq_scaled", "item_te_scaled", "id_te_scaled"]):
        item_counts = df["item_id"].value_counts()
        id_counts = df["id"].value_counts()
        item_means = df.groupby("item_id")[TARGET_COL].mean()
        id_means = df.groupby("id")[TARGET_COL].mean()
        global_mean = df[TARGET_COL].mean()
        df["item_freq"] = df["item_id"].map(item_counts).fillna(0).astype("float32")
        df["id_freq"] = df["id"].map(id_counts).fillna(0).astype("float32")
        df["item_te"] = df["item_id"].map(item_means).fillna(global_mean).astype("float32")
        df["id_te"] = df["id"].map(id_means).fillna(global_mean).astype("float32")
        for col in ["item_freq", "id_freq", "item_te", "id_te"]:
            mn = df[col].min()
            mx = df[col].max()
            rng = mx - mn
            df[col + "_scaled"] = 0.0 if rng == 0 else (df[col] - mn) / rng
        for col in ["item_freq", "id_freq", "item_te", "id_te", "item_freq_scaled", "id_freq_scaled", "item_te_scaled", "id_te_scaled"]:
            df[col] = df[col].astype("float32")
    if HISTORY_WINDOW is not None:
        cutoff = df["d_int"].max() - HISTORY_WINDOW + 1
        df = df[df["d_int"] >= cutoff].reset_index(drop=True)
    return df

def fit_models(train_df: pd.DataFrame, feature_cols: List[str], params: dict):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train_df[feature_cols],
        train_df[TARGET_COL].astype("float32"),
        categorical_feature=CAT_COLS,
    )
    return model


def prepare_future_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    # create sin/cos for future rows
    cal = calendar.copy()
    cal["d_int"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    if "quarter" not in cal.columns:
        cal["quarter"] = ((cal["month"] - 1) // 3 + 1).astype(int)
    cal["wday_sin"] = np.sin(2 * np.pi * cal["wday"] / 7)
    cal["wday_cos"] = np.cos(2 * np.pi * cal["wday"] / 7)
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12)
    cal["quarter_sin"] = np.sin(2 * np.pi * cal["quarter"] / 4)
    cal["quarter_cos"] = np.cos(2 * np.pi * cal["quarter"] / 4)
    return cal


def iterative_forecast(
    store: str,
    blend_cfg: dict,
    params: dict,
    cal: pd.DataFrame,
    prices_store: pd.DataFrame,
) -> pd.DataFrame:
    df = load_store_history(store)
    last_d = df["d_int"].max()
    horizon = 28

    # Precompute encoders for freq/te
    item_counts = df["item_id"].value_counts()
    id_counts = df["id"].value_counts()
    global_mean = df[TARGET_COL].mean()
    item_means = df.groupby("item_id")[TARGET_COL].mean()
    id_means = df.groupby("id")[TARGET_COL].mean()
    # min-max stats for freq/te from train_df
    item_freq_min, item_freq_max = item_counts.min(), item_counts.max()
    id_freq_min, id_freq_max = id_counts.min(), id_counts.max()
    item_te_min, item_te_max = item_means.min(), item_means.max()
    id_te_min, id_te_max = id_means.min(), id_means.max()

    feature_cols = NUM_SCALED + CYCLIC + BIN_COLS + CAT_COLS
    train_df = df[df["d_int"] <= last_d].copy()
    # ensure raw lag/rolling cols exist for scaling stats
    raw_lags = [f"lag_{k}" for k in [1, 7, 14, 28, 30, 56, 84]]
    raw_roll_mean = [f"rolling_mean_{k}" for k in [7, 14, 28, 30, 56, 84]]
    raw_roll_std = [f"rolling_std_{k}" for k in [7, 14, 28, 30, 56, 84]]
    raw_roll_med = [f"rolling_median_{k}" for k in [7, 14, 28, 30, 56, 84]]
    raw_roll_min = [f"rolling_min_{k}" for k in [7, 14, 28, 30, 56, 84]]
    raw_roll_max = [f"rolling_max_{k}" for k in [7, 14, 28, 30, 56, 84]]
    for col in raw_lags + raw_roll_mean + raw_roll_std + raw_roll_med + raw_roll_min + raw_roll_max:
        if col not in train_df.columns:
            train_df[col] = np.float32(0.0)
    # set categorical dtypes based on train
    cat_types = {}
    for col in CAT_COLS:
        cats = pd.CategoricalDtype(categories=train_df[col].dropna().unique())
        train_df[col] = train_df[col].astype(cats)
        cat_types[col] = cats
    # binary to int
    for col in BIN_COLS:
        train_df[col] = train_df[col].astype("int8")
    # ensure freq/te scaled exist in train_df for min-max reference
    if "item_freq_scaled" not in train_df:
        train_df["item_freq_scaled"] = item_counts.reindex(train_df["item_id"]).values
        train_df["id_freq_scaled"] = id_counts.reindex(train_df["id"]).values
        train_df["item_te_scaled"] = item_means.reindex(train_df["item_id"]).fillna(global_mean).values
        train_df["id_te_scaled"] = id_means.reindex(train_df["id"]).fillna(global_mean).values

    # Fit models
    model_base = fit_models(train_df, feature_cols, params)
    models = [model_base]
    if store not in NO_BLEND_STORES:
        from train_lgbm_baseline import tweak_params_for_blend, tweak_params_for_blend2  # reuse helpers

        model_alt = fit_models(train_df, feature_cols, tweak_params_for_blend(params))
        model_alt2 = fit_models(train_df, feature_cols, tweak_params_for_blend2(params))
        models.extend([model_alt, model_alt2])

    preds_records = []
    cal_future = cal[cal["d_int"] > last_d].copy()

    # Build a dict for quick row updates
    df.set_index("d_int", inplace=True)

    for d_int in range(last_d + 1, last_d + horizon + 1):
        # get calendar row
        cal_row = cal_future[cal_future["d_int"] == d_int]
        if cal_row.empty:
            raise ValueError(f"Calendar missing d_{d_int}")
        cal_row = cal_row.iloc[0]
        # price info: merge on (item_id, wm_yr_wk)
        week = cal_row["wm_yr_wk"]
        price_map = prices_store[prices_store["wm_yr_wk"] == week].set_index("item_id")

        # build today's rows per id
        todays = []
        for id_val, hist_row in df[df.index == last_d].iterrows():
            item_id = hist_row["item_id"]
            row = hist_row.copy()
            row.name = d_int
            row["d"] = f"d_{d_int}"
            row["d_int"] = d_int
            # calendar fields
            for col in ["wday", "month", "year", "quarter", "snap_CA", "snap_TX", "snap_WI", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "wday_sin", "wday_cos", "month_sin", "month_cos", "quarter_sin", "quarter_cos"]:
                row[col] = cal_row.get(col, np.nan)
            # derive holiday/promo flags if not present in calendar
            row["IsHoliday"] = cal_row.get(
                "IsHoliday",
                1
                if (pd.notna(cal_row.get("event_name_1", np.nan)) or pd.notna(cal_row.get("event_name_2", np.nan)))
                else 0,
            )
            row["IsPromotion"] = cal_row.get("IsPromotion", 0)
            # price fields
            if item_id in price_map.index:
                row["sell_price"] = price_map.loc[item_id, "sell_price"]
                row["baseline_price"] = price_map.loc[item_id, "sell_price"]  # assume baseline ~ sell_price if missing
            else:
                row["sell_price"] = row.get("sell_price", 0)
                row["baseline_price"] = row.get("baseline_price", 0)
            row["discount"] = max(0.0, row["baseline_price"] - row["sell_price"])
            row["promo_intensity"] = row["discount"] * row["IsPromotion"]
            row["price_ratio"] = row["sell_price"] / row["baseline_price"] if row["baseline_price"] > 0 else 0
            row["discount_pct"] = row["discount"] / row["baseline_price"] if row["baseline_price"] > 0 else 0
            snap_state = row["snap_CA"] if row["state_id"] == "CA" else row["snap_TX"] if row["state_id"] == "TX" else row["snap_WI"]
            row["snap_wday"] = snap_state * row["wday"]
            row["promo_holiday"] = row["IsPromotion"] * row["IsHoliday"]
            row["promo_wday_sin"] = row["promo_intensity"] * row["wday_sin"]
            row["promo_wday_cos"] = row["promo_intensity"] * row["wday_cos"]
            row["discount_snap"] = row["discount_pct"] * snap_state
            todays.append(row)
        today_df = pd.DataFrame(todays)
        today_df.set_index("d_int", inplace=True)

        # compute lags/rollings using existing df (which includes past preds)
        df_full = pd.concat([df, today_df]).sort_index()
        g = df_full.groupby("id")[TARGET_COL]
        for lag in [1, 7, 14, 28, 30, 56, 84]:
            df_full[f"lag_{lag}"] = g.shift(lag)
        for win in [7, 14, 28, 30, 56, 84]:
            df_full[f"rolling_mean_{win}"] = g.transform(lambda s, w=win: s.shift(1).rolling(w, min_periods=1).mean())
            df_full[f"rolling_std_{win}"] = g.transform(lambda s, w=win: s.shift(1).rolling(w, min_periods=1).std())
            df_full[f"rolling_median_{win}"] = g.transform(lambda s, w=win: s.shift(1).rolling(w, min_periods=1).median())
            df_full[f"rolling_min_{win}"] = g.transform(lambda s, w=win: s.shift(1).rolling(w, min_periods=1).min())
            df_full[f"rolling_max_{win}"] = g.transform(lambda s, w=win: s.shift(1).rolling(w, min_periods=1).max())

        # take current day rows
        cur = df_full.loc[[d_int]].reset_index()

        # freq/te (use train stats)
        cur["item_freq"] = cur["item_id"].map(item_counts).fillna(0)
        cur["id_freq"] = cur["id"].map(id_counts).fillna(0)
        cur["item_te"] = cur["item_id"].map(item_means).fillna(global_mean)
        cur["id_te"] = cur["id"].map(id_means).fillna(global_mean)
        # min-max scaling using train stats
        cur["item_freq_scaled"] = 0.0 if item_freq_max == item_freq_min else (cur["item_freq"] - item_freq_min) / (item_freq_max - item_freq_min)
        cur["id_freq_scaled"] = 0.0 if id_freq_max == id_freq_min else (cur["id_freq"] - id_freq_min) / (id_freq_max - id_freq_min)
        cur["item_te_scaled"] = 0.0 if item_te_max == item_te_min else (cur["item_te"] - item_te_min) / (item_te_max - item_te_min)
        cur["id_te_scaled"] = 0.0 if id_te_max == id_te_min else (cur["id_te"] - id_te_min) / (id_te_max - id_te_min)

        # scale new numeric cols using train min-max
        base_scale_cols = [
            "sell_price",
            "baseline_price",
            "discount",
            "promo_intensity",
            "price_ratio",
            "discount_pct",
            "snap_wday",
            "promo_holiday",
            "promo_wday_sin",
            "promo_wday_cos",
            "discount_snap",
            "promo_streak",
            "holiday_streak",
            "days_since_holiday",
            "days_until_holiday",
            "sell_price_week_chg",
            "sell_price_month_chg",
            "sell_price_z28",
            "discount_week_chg",
            "discount_month_chg",
            "discount_z28",
        ]
        lag_cols = [f"lag_{k}" for k in [1, 7, 14, 28, 30, 56, 84]]
        roll_mean_cols = [f"rolling_mean_{k}" for k in [7, 14, 28, 30, 56, 84]]
        roll_std_cols = [f"rolling_std_{k}" for k in [7, 14, 28, 30, 56, 84]]
        roll_median_cols = [f"rolling_median_{k}" for k in [7, 14, 28, 30, 56, 84]]
        roll_min_cols = [f"rolling_min_{k}" for k in [7, 14, 28, 30, 56, 84]]
        roll_max_cols = [f"rolling_max_{k}" for k in [7, 14, 28, 30, 56, 84]]
        for col in base_scale_cols + lag_cols + roll_mean_cols + roll_std_cols + roll_median_cols + roll_min_cols + roll_max_cols:
            mn = train_df[col].min(skipna=True)
            mx = train_df[col].max(skipna=True)
            rng = mx - mn
            cur[f"{col}_scaled"] = 0.0 if pd.isna(rng) or rng == 0 else (cur[col].fillna(0) - mn) / rng

        # set dtypes consistent with train
        for col in CAT_COLS:
            cur[col] = cur[col].astype(cat_types[col])
        for col in BIN_COLS:
            cur[col] = cur[col].astype("int8")

        # predict
        cur_feats = cur[feature_cols]
        preds = []
        if store in NO_BLEND_STORES:
            preds = model_base.predict(cur_feats, num_iteration=model_base.best_iteration_)
        else:
            p0 = models[0].predict(cur_feats, num_iteration=models[0].best_iteration_)
            p1 = models[1].predict(cur_feats, num_iteration=models[1].best_iteration_)
            p2 = models[2].predict(cur_feats, num_iteration=models[2].best_iteration_)
            w = blend_cfg.get(store, {"w_base": 1.0, "w_alt": 0.0, "w_alt2": 0.0})
            w_base = float(w.get("w_base", 1.0) or 0.0)
            w_alt = float(w.get("w_alt", 0.0) or 0.0)
            w_alt2 = float(w.get("w_alt2", 0.0) or 0.0)
            preds = w_base * p0 + w_alt * p1 + w_alt2 * p2

        cur["sales"] = preds
        cur = cur.set_index("d_int")
        # keep full columns for next-step lag/rolling computation
        df = pd.concat([df, cur[df.columns]], sort=False)

        preds_records.append(cur.reset_index()[["id", "d", "sales"]].rename(columns={"sales": "pred"}))

    preds_df = pd.concat(preds_records, ignore_index=True)
    return preds_df


def forecast_store(args) -> pd.DataFrame:
    store, blend_cfg, params, cal, prices_store = args
    return iterative_forecast(store, blend_cfg, params, cal, prices_store)


def main():
    args = parse_args()
    blend_cfg_all = load_blend_weights()
    requested = [s.strip() for s in args.stores.split(",") if s.strip()]
    if not requested:
        raise ValueError("No stores provided for forecasting.")
    blend_cfg = {store: blend_cfg_all[store] for store in requested if store in blend_cfg_all}
    missing = [store for store in requested if store not in blend_cfg_all]
    if missing:
        print(f"Warning: blend config missing for {missing}; they will be skipped.")
    if not blend_cfg:
        raise ValueError("None of the requested stores have blend weights available.")
    for store in blend_cfg:
        blend_cfg[store] = {"type": "c_model", "w_base": 0.0, "w_alt": 1.0, "w_alt2": 0.0}

    calendar = pd.read_csv(CAL_PATH)
    calendar = prepare_future_calendar(calendar)
    prices = pd.read_csv(PRICE_PATH)
    # split prices by store to avoid repeated filtering
    prices_dict = {sid: df for sid, df in prices.groupby("store_id")}

    all_preds = []
    jobs = []
    from train_lgbm_baseline import STORE_PARAMS

    for store in blend_cfg.keys():
        params = STORE_PARAMS.get(store)
        if params is None:
            continue
        prices_store = prices_dict.get(store)
        jobs.append((store, blend_cfg, params, calendar, prices_store))

    def run_job(job):
        store, blend_cfg_i, params_i, cal_i, prices_i = job
        cache_path = CACHE_DIR / f"preds_{store}.parquet"
        if cache_path.exists():
            print(f"Loading cached predictions for {store}")
            return cache_path
        print(f"Forecasting {store} ...")
        preds_df = iterative_forecast(
            store,
            blend_cfg_i,
            params_i,
            cal_i,
            prices_i,
        )
        preds_df.to_parquet(cache_path, index=False)
        print(f"Done {store}")
        return cache_path

    # sequential to save memory
    cache_files = []
    for job in jobs:
        path = run_job(job)
        cache_files.append(path)
        gc.collect()

    sub = pd.concat((pd.read_parquet(p) for p in cache_files), ignore_index=True)
    # keep only necessary columns and downcast to save memory
    sub = sub[["id", "d", "pred"]]
    sub["pred"] = pd.to_numeric(sub["pred"], errors="coerce").astype("float32").fillna(0)
    sub["d_num"] = sub["d"].str.replace("d_", "", regex=False).astype("int16")
    # Pivot to submission wide format (evaluation rows only, float32 to save memory)
    pivot = (
        sub[sub["id"].str.endswith("evaluation")][["id", "d_num", "pred"]]
        .pivot_table(index="id", columns="d_num", values="pred", aggfunc="first")
        .astype("float32")
        .sort_index(axis=1)
    )
    max_d = pivot.columns.max()
    horizon = list(range(max_d - 27, max_d + 1))
    pivot = pivot[horizon]
    pivot.columns = [f"F{i}" for i in range(1, 29)]
    pivot.reset_index(inplace=True)

    sample = pd.read_csv("data/sample_submission.csv")
    value_cols = [c for c in sample.columns if c.startswith("F")]
    sample[value_cols] = sample[value_cols].astype("float32")
    sample = sample.set_index("id")
    preds_aligned = pivot.set_index("id")
    eval_ids = [idx for idx in sample.index if idx.endswith("evaluation") and idx in preds_aligned.index]
    sample.loc[eval_ids, value_cols] = preds_aligned.loc[eval_ids, value_cols].values
    sample.reset_index(inplace=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"Wrote submission to {args.out} (aligned to sample_submission)")


if __name__ == "__main__":
    main()
