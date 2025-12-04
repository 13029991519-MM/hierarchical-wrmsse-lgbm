"""
Generate enriched features for M5 data and save to finaldata/.

New features added:
- lag_1, lag_14, lag_28 on sales
- rolling_mean_14/28, rolling_std_14/28 on sales
- price_ratio = sell_price / baseline_price
- discount_pct = discount / baseline_price
- promo_holiday = IsPromotion * IsHoliday
- snap_wday = snap_state * wday  (snap for the row's state)

Scaled versions (_scaled) are computed via min-max per file.

Source directory: newdata_evaluation
Output directory: finaldata
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SRC_DIR = Path("newdata_evaluation")
OUT_DIR = Path("newfinaldata")

OUT_DIR.mkdir(exist_ok=True)

FILES = sorted(SRC_DIR.glob("processed_*.csv"))

# New numeric features to scale
# Base new numeric features to scale
NEW_NUM_COLS = [
    "lag_1",
    "lag_14",
    "lag_28",
    "lag_56",
    "lag_84",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_mean_84",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_std_56",
    "rolling_std_84",
    "price_ratio",
    "discount_pct",
    "snap_wday",
    "promo_holiday",
    "promo_wday_sin",
    "promo_wday_cos",
    "discount_snap",
    # rolling median/min/max
    "rolling_median_7",
    "rolling_median_14",
    "rolling_median_28",
    "rolling_median_30",
    "rolling_median_56",
    "rolling_median_84",
    "rolling_min_7",
    "rolling_min_14",
    "rolling_min_28",
    "rolling_min_30",
    "rolling_min_56",
    "rolling_min_84",
    "rolling_max_7",
    "rolling_max_14",
    "rolling_max_28",
    "rolling_max_30",
    "rolling_max_56",
    "rolling_max_84",
    # price momentum / z-score
    "sell_price_week_chg",
    "sell_price_month_chg",
    "sell_price_z28",
    "discount_week_chg",
    "discount_month_chg",
    "discount_z28",
    # promo/holiday streaks
    "promo_streak",
    "holiday_streak",
    # event distance
    "days_since_holiday",
    "days_until_holiday",
]

for path in FILES:
    print(f"Processing {path.name} ...")
    df = pd.read_csv(path, low_memory=False)

    # Ensure d integer for sorting/grouping (if needed)
    df["d_int"] = df["d"].str.replace("d_", "", regex=False).astype(int)
    df = df.sort_values(["id", "d_int"])

    g = df.groupby("id", sort=False)["sales"]
    df["lag_1"] = g.shift(1)
    df["lag_14"] = g.shift(14)
    df["lag_28"] = g.shift(28)
    df["lag_56"] = g.shift(56)
    df["lag_84"] = g.shift(84)
    df["rolling_mean_14"] = g.transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["rolling_std_14"] = g.transform(lambda s: s.rolling(14, min_periods=1).std())
    df["rolling_mean_28"] = g.transform(lambda s: s.rolling(28, min_periods=1).mean())
    df["rolling_std_28"] = g.transform(lambda s: s.rolling(28, min_periods=1).std())
    df["rolling_mean_56"] = g.transform(lambda s: s.rolling(56, min_periods=1).mean())
    df["rolling_std_56"] = g.transform(lambda s: s.rolling(56, min_periods=1).std())
    df["rolling_mean_84"] = g.transform(lambda s: s.rolling(84, min_periods=1).mean())
    df["rolling_std_84"] = g.transform(lambda s: s.rolling(84, min_periods=1).std())
    # rolling median/min/max for robustness
    for win in [7, 14, 28, 30, 56, 84]:
        df[f"rolling_median_{win}"] = g.transform(lambda s, w=win: s.rolling(w, min_periods=1).median())
        df[f"rolling_min_{win}"] = g.transform(lambda s, w=win: s.rolling(w, min_periods=1).min())
        df[f"rolling_max_{win}"] = g.transform(lambda s, w=win: s.rolling(w, min_periods=1).max())

    # Price ratios
    df["price_ratio"] = np.where(df["baseline_price"] > 0, df["sell_price"] / df["baseline_price"], np.nan)
    df["discount_pct"] = np.where(df["baseline_price"] > 0, df["discount"] / df["baseline_price"], np.nan)

    # Interactions
    df["promo_holiday"] = df["IsPromotion"] * df["IsHoliday"]
    snap_state = np.select(
        [
            df["state_id"] == "CA",
            df["state_id"] == "TX",
            df["state_id"] == "WI",
        ],
        [
            df["snap_CA"],
            df["snap_TX"],
            df["snap_WI"],
        ],
        default=0,
    )
    df["snap_wday"] = snap_state * df["wday"]
    # Additional interactions
    df["promo_wday_sin"] = df["promo_intensity"] * df.get("wday_sin", 0)
    df["promo_wday_cos"] = df["promo_intensity"] * df.get("wday_cos", 0)
    df["discount_snap"] = df["discount_pct"] * snap_state
    # price momentum & z-score (28d)
    for col in ["sell_price", "discount"]:
        df[f"{col}_week_chg"] = df.groupby("item_id")[col].pct_change(periods=7)
        df[f"{col}_month_chg"] = df.groupby("item_id")[col].pct_change(periods=28)
        roll_mean = df.groupby("item_id")[col].transform(lambda s: s.rolling(28, min_periods=2).mean())
        roll_std = df.groupby("item_id")[col].transform(lambda s: s.rolling(28, min_periods=2).std())
        df[f"{col}_z28"] = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
    # promo/holiday streaks (consecutive days)
    df["promo_streak"] = df.groupby("id")["IsPromotion"].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    ) * df["IsPromotion"]
    df["holiday_streak"] = df.groupby("id")["IsHoliday"].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    ) * df["IsHoliday"]
    # event distance (to prev/next holiday)
    holiday_days = df.loc[df["IsHoliday"] == 1, "d_int"].unique()
    if len(holiday_days) > 0:
        prev_map = pd.Series(holiday_days).sort_values()
        next_map = prev_map
        df["days_since_holiday"] = df["d_int"].apply(lambda x: x - prev_map[prev_map <= x].max() if any(prev_map <= x) else np.nan)
        df["days_until_holiday"] = df["d_int"].apply(lambda x: next_map[next_map >= x].min() - x if any(next_map >= x) else np.nan)
    else:
        df["days_since_holiday"] = np.nan
    df["days_until_holiday"] = df.get("days_until_holiday", np.nan)

    # Fill numeric NaNs to 0 to keep downstream models simple
    NUM_COLS_TO_FILL = NEW_NUM_COLS + [
        "sell_price",
        "baseline_price",
        "discount",
        "promo_intensity",
    ]
    for col in NUM_COLS_TO_FILL:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill categorical NaNs to 'Unknown'
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
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Scale new numeric columns
    for col in NEW_NUM_COLS:
        mn = df[col].min(skipna=True)
        mx = df[col].max(skipna=True)
        rng = mx - mn
        if pd.isna(mn) or pd.isna(mx) or rng == 0:
            df[col + "_scaled"] = 0.0
        else:
            df[col + "_scaled"] = (df[col] - mn) / rng

    out_path = OUT_DIR / path.name
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

print("All done.")
