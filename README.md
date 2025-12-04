# hierarchical-wrmsse-lgbm

Code accompanying the project **"Store-specific LightGBM with WRMSSE-aware Blending for the M5 Forecasting–Accuracy Task"**.

The repository implements a hierarchical, WRMSSE-aligned forecasting pipeline for the Kaggle M5 competition.  
It reproduces the system described in the paper, including:

- per-store LightGBM models with **core** and **extra** feature regimes
- dual-window **WRMSSE** model selection (allow / neutral / ban)
- WRMSSE-guided per-store **blending**
- 28-day **rolling forecasting** with feature updates
- several **baseline** models (naïve, ARIMA)

The code is organised so that others can inspect or adapt individual stages (feature engineering, training, evaluation, submission generation).

---

## Repository structure

```text
hierarchical-wrmsse-lgbm/
├── README.md
├── LICENSE
│
├── data_prep/
│   ├── build_features_v2.py        # main feature engineering for all stores
│   ├── build_features_v3.py        # compact feature build for representative stores
│   └── generate_finaldata.py       # final processed CSV generation
│
├── training/
│   ├── train_lgbm_baseline.py      # main per-store LightGBM training (core + extra)
│   ├── train_lgbm_v2_long.py       # long-seasonality variant on v2 features
│   ├── train_lgbm_v3_long.py       # long-seasonality variant on v3 features
│   ├── run_chunked_store.py        # store-wise training helper
│   ├── split_and_run.py            # split big CSV into chunks and call train_lgbm_baseline
│   ├── search_group_params.py      # group-level hyperparameter search (dual-window WRMSSE)
│   └── run_tx3_full.py             # full training routine for a representative store (TX_3)
│
├── evaluation/
│   ├── wrmsse_official.py          # official-style WRMSSE evaluator (12-level hierarchy)
│   ├── evaluate_wrmsse_windows.py  # dual-window WRMSSE diagnostics (allow / neutral / ban)
│   ├── evaluate_wrmsse_blended.py  # WRMSSE of blended validation submission
│   ├── evaluate_mae_rmse.py        # MAE / RMSE metrics on the validation window
│   ├── evaluate_mae_rmse_levels.py # optional per-level error diagnostics
│   └── summary_weights_viz.py      # visualisation of auto-decisions and blend weights
│
├── submission/
│   ├── fill_validation_submission.py  # generate submission_with_val.csv (F1–F28)
│   ├── visualize_and_blend.py         # scatter plots + blended validation submission
│   └── predict_future.py              # 28-day rolling forecasting for the final submission
│
├── baselines/
│   ├── naive_baseline.py           # simple naïve baseline
│   └── run_arima_baseline.py       # ARIMA/SARIMAX-style baseline
│
└── experimental/
    ├── export_val_preds_A_C.py     # early A/C variant experiment (kept for reference)
    └── train_lgbm_baseline_tmp.py  # legacy version of the training script
```

---

## High-level pipeline

1. **Prepare features**

   - Build intermediate features by store:

     ```bash
     python data_prep/build_features_v2.py
     python data_prep/build_features_v3.py   # only for representative stores
     python data_prep/generate_finaldata.py
     ```

2. **Search group-level hyperparameters**

   - Run random search for groups (e.g. CA / TX / WI) and both regimes:

     ```bash
     python training/search_group_params.py
     ```

3. **Train per-store models (core & extra)**

   - Option A – directly per store  
   - Option B – chunked training when memory is tight:

     ```bash
     python training/run_chunked_store.py
     # or
     python training/split_and_run.py
     ```

   - This stage produces:
     - per-store LightGBM models
     - a summary of WRMSSE scores and auto-decisions
     - per-store blending weights (stored as JSON)

4. **Diagnostics and WRMSSE-aware blending**

   - Inspect dual-window WRMSSE behaviour:

     ```bash
     python evaluation/evaluate_wrmsse_windows.py
     python evaluation/summary_weights_viz.py
     ```

   - Generate a blended validation submission and evaluate:

     ```bash
     python submission/fill_validation_submission.py
     python submission/visualize_and_blend.py
     python evaluation/evaluate_wrmsse_blended.py
     ```

5. **Final 28-day forecasting and Kaggle submission**

   - Using the selected hyperparameters and per-store blend weights:

     ```bash
     python submission/predict_future.py
     ```

   - The script iteratively rolls forward from d_1942 to d_1969, updating lags and rolling-window
     features, and writes predictions to the M5 submission format.

6. **Baselines (optional)**

   - Naïve baseline:

     ```bash
     python baselines/naive_baseline.py
     ```

   - ARIMA baseline:

     ```bash
     python baselines/run_arima_baseline.py
     ```

---

## Dependencies

The code assumes a standard scientific Python stack, for example:

- Python >= 3.9
- numpy
- pandas
- lightgbm
- scikit-learn
- matplotlib (for plots)
- statsmodels (for ARIMA baseline)

You may install them via:

```bash
pip install -r requirements.txt
```

(You can create `requirements.txt` based on your local environment.)

---

## Data

The scripts expect the original **M5 Forecasting – Accuracy** data in a `data/` folder
with file names matching the Kaggle competition:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`
- `sample_submission.csv`

Please download the dataset from Kaggle and place it under `data/` before running any scripts.

---

## Citation

If you find this code useful in your own work, please consider citing the accompanying paper
or linking back to this repository.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
