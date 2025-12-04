"""
Quick ARIMA/SARIMAX baseline on aggregated state-level sales.

What it does:
- Reads processed store CSVs from newfinaldata.
-, aggregates by state per day, splits train d_1–1913 and val d_1914–1941.
- Fits a simple weekly seasonal SARIMAX per state and reports RMSE/SMAPE on val.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

DATA_DIR = Path("newfinaldata")
STATE_GROUPS: Dict[str, List[str]] = {
    "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
    "TX": ["TX_1", "TX_2", "TX_3"],
}

TRAIN_END = 1913
VAL_END = 1941


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(np.mean(2.0 * num / denom))


def load_state_series(state: str) -> pd.Series:
    stores = STATE_GROUPS[state]
    dfs = []
    for store in stores:
        path = DATA_DIR / f"processed_{store}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(path, usecols=["d", "sales"])
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["d_int"] = df_all["d"].str.replace("d_", "", regex=False).astype(int)
    agg = df_all.groupby("d_int")["sales"].sum().sort_index()
    # attach a daily DateIndex to avoid statsmodels warnings
    start_date = pd.Timestamp("2011-01-29")  # matches M5 day1
    agg.index = pd.date_range(start=start_date, periods=len(agg), freq="D")
    return agg


def fit_eval(state: str, orders: list[tuple], seasonal_orders: list[tuple]) -> None:
    series = load_state_series(state)
    start_date = series.index.min()
    train_end_date = start_date + pd.Timedelta(days=TRAIN_END - 1)
    val_end_date = start_date + pd.Timedelta(days=VAL_END - 1)
    train = series[series.index <= train_end_date]
    val = series[(series.index > train_end_date) & (series.index <= val_end_date)]
    if val.empty:
        raise ValueError(f"No val split for state {state}")

    best = None
    for order, seasonal_order in itertools.product(orders, seasonal_orders):
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            preds = res.forecast(steps=len(val))
            rmse = float(np.sqrt(mean_squared_error(val, preds)))
            smape_val = smape(val.values, preds.values)
            if (best is None) or (rmse < best["rmse"]):
                best = {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "rmse": rmse,
                    "smape": smape_val,
                    "aic": res.aic,
                }
        except Exception as e:  # pragma: no cover
            print(f"[{state}] Failed order={order} seasonal={seasonal_order}: {e}")
            continue

    print(f"[{state}] Best RMSE {best['rmse']:.4f} | SMAPE {best['smape']:.4f} | order {best['order']} | seasonal {best['seasonal_order']} | AIC {best['aic']:.2f}")


def main() -> None:
    # Simple search around weekly seasonality
    orders = [(1, 0, 1), (2, 0, 2)]
    seasonal_orders = [(0, 1, 1, 7), (1, 1, 1, 7)]
    for state in STATE_GROUPS:
        fit_eval(state, orders, seasonal_orders)


if __name__ == "__main__":
    main()
