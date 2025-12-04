import os

import pandas as pd


DATA_DIR = "data"
SALES_PATH_EVAL = os.path.join(DATA_DIR, "sales_train_evaluation.csv")
SALES_PATH_VAL = os.path.join(DATA_DIR, "sales_train_validation.csv")
SALES_PATH = SALES_PATH_EVAL if os.path.exists(SALES_PATH_EVAL) else SALES_PATH_VAL

if SALES_PATH == SALES_PATH_EVAL:
    print(f">>> using evaluation data ({SALES_PATH_EVAL})")
else:
    print(f">>> evaluation data missing; falling back to validation ({SALES_PATH_VAL})")

SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = os.path.join(DATA_DIR, "submission_naive_28shift.csv")

H = 28
VAL_START, VAL_END = 1914, 1941
TEST_START, TEST_END = 1942, 1969


def wide_to_long(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[["id"] + cols].melt(id_vars="id", value_name="sales", var_name="d")
    out["d"] = out["d"].str.replace("d_", "", regex=False).astype(int)
    return out


print(f">>> reading {SALES_PATH}")
sales = pd.read_csv(SALES_PATH)

hist_cols_val = [f"d_{d}" for d in range(VAL_START - H, VAL_END - H + 1)]
val_cols = [f"d_{d}" for d in range(VAL_START, VAL_END + 1)]
hist_cols_test = [f"d_{d}" for d in range(TEST_START - H, TEST_END - H + 1)]
future_cols_test = [f"d_{d}" for d in range(TEST_START, TEST_END + 1)]

print(">>> constructing val prediction via naive shift")
print("  val columns:", len(val_cols), "shifted columns:", len(hist_cols_val))

true_val_long = wide_to_long(sales, val_cols)
pred_val_long = wide_to_long(sales, hist_cols_val)

print(">>> building future predictions")
y_pred_test = sales[hist_cols_test].copy()
y_pred_test["id"] = sales["id"].values
y_pred_test = y_pred_test[["id"] + hist_cols_test]
y_pred_test.columns = ["id"] + future_cols_test

try:
    from wrmsse_official import WRMSSEEvaluator

    print(">>> computing WRMSSE")
    evaluator = WRMSSEEvaluator()
    wrmsse_val, _ = evaluator.compute_wrmsse(true_val_long, pred_val_long)
    print(f"  WRMSSE on d_{VAL_START}-{VAL_END}: {wrmsse_val:.6f}")
except ImportError:
    print(">>> wrmsse_official not available; skipping WRMSSE computation")

print(f">>> reading {SAMPLE_SUB_PATH}")
sub = pd.read_csv(SAMPLE_SUB_PATH)

val_pred = pd.DataFrame({"id": sales["id"].values})
for i, d in enumerate(range(TEST_START, TEST_END + 1), start=1):
    val_pred[f"F{i}"] = y_pred_test[f"d_{d}"]

val_pred["id"] = val_pred["id"].str.replace("_validation", "_evaluation", regex=False)
eval_mask = sub["id"].str.endswith("_evaluation")
sub_eval_ids = sub.loc[eval_mask, "id"]
sub.loc[eval_mask, "F1":"F28"] = val_pred.set_index("id").loc[sub_eval_ids].values

sub.to_csv(OUT_PATH, index=False)
print(f">>> saved naive submission to {OUT_PATH}")
