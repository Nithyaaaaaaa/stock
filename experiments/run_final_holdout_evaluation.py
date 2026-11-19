"""
Phase 4: Final Evaluation on Untouched Holdout Test Set (597 Sessions).
Fits models on Development Data (2,325 sessions) and evaluates ONCE on the frozen holdout.
Generates statistical tests, conformal prediction coverage, and publication charts.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data, split_development_and_holdout
from src.features.technical import compute_all_candidate_features, evaluate_feature_stationarity
from src.models.baselines import PersistenceForecaster, DriftForecaster, MajorityClassForecaster
from src.models.trees import CatBoostForecaster, XGBoostForecaster
from src.models.deep_learning import LSTMForecaster
from src.evaluation.metrics import compute_regression_metrics, compute_directional_metrics
from src.evaluation.statistical_tests import diebold_mariano_test, paired_wilcoxon_test
from src.evaluation.conformal import ConformalPredictor


def run_holdout_evaluation():
    print("=" * 95)
    print(" PHASE 4: FINAL EVALUATION ON UNTOUCHED HOLDOUT DATASET (597 SESSIONS)")
    print("=" * 95)

    # 1. Fetch & Split Data
    raw_target, raw_bench = fetch_market_data(ticker="AAPL", benchmark_ticker="SPY", start="2015-01-01")
    df_clean = validate_and_clean_data(raw_target, raw_bench)
    dev_df, holdout_df, split_meta = split_development_and_holdout(df_clean, holdout_pct=20.0)

    # Compute features
    all_features = compute_all_candidate_features(df_clean)
    usable_features = all_features.iloc[65:-1]

    # Partition features
    dev_data = usable_features.loc[usable_features.index < holdout_df.index[0]]
    holdout_data = usable_features.loc[usable_features.index >= holdout_df.index[0]]

    print(f"Development Set: {len(dev_data)} sessions ({dev_data.index[0].strftime('%Y-%m-%d')} to {dev_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"Holdout Set:     {len(holdout_data)} sessions ({holdout_data.index[0].strftime('%Y-%m-%d')} to {holdout_data.index[-1].strftime('%Y-%m-%d')})")

    stationary_features = [
        "Return_Lag_1", "Return_Lag_5", "Log_Return_Lag_1", "Log_Return_Lag_5",
        "Ratio_Close_SMA_5", "Ratio_Close_SMA_20", "Ratio_Close_SMA_50", "Ratio_SMA_5_20",
        "Ratio_Close_EMA_12", "Ratio_Close_EMA_26", "MACD_Normalized",
        "RSI_14_Centered", "Stoch_K_14", "High_Low_Ratio", "Close_Open_Ratio",
        "Vol_10d", "Vol_20d", "Vol_60d", "NATR_14", "Bollinger_Pct_B",
        "Log_Volume_Change", "Volume_Ratio_SMA_20",
        "Benchmark_Return_Lag_1", "Excess_Return_Over_Benchmark", "Rolling_Beta_60"
    ]
    feature_cols = [c for c in stationary_features if c in usable_features.columns]

    # Feature Scaling: Fit strictly on Development Set
    scaler = StandardScaler()
    X_dev = scaler.fit_transform(dev_data[feature_cols].values)
    y_dev = dev_data["Target_Next_Log_Return"].values

    X_test = scaler.transform(holdout_data[feature_cols].values)
    y_test = holdout_data["Target_Next_Log_Return"].values

    test_prev_closes = holdout_data["Close"].values
    test_actual_closes = holdout_data["Target_Next_Close"].values
    test_dates = holdout_data.index

    # Internal Dev Split for Conformal Calibration (last 20% of Dev as Calibration)
    cal_idx = int(len(X_dev) * 0.8)
    X_dev_sub = X_dev[:cal_idx]
    y_dev_sub = y_dev[:cal_idx]
    X_cal = X_dev[cal_idx:]
    y_cal = y_dev[cal_idx:]
    cal_vol = dev_data["Vol_20d"].iloc[cal_idx:].values
    test_vol = holdout_data["Vol_20d"].values

    # Models to Evaluate
    models = {
        "CatBoost (Selected Best)": CatBoostForecaster(iterations=120, depth=4, learning_rate=0.02, random_seed=42),
        "XGBoost": XGBoostForecaster(n_estimators=100, max_depth=3, learning_rate=0.02, random_state=42),
        "Two-Layer LSTM": LSTMForecaster(seq_len=20, hidden_dim=32, num_layers=2, epochs=25, lr=0.003, seed=42),
        "Persistence Baseline": PersistenceForecaster(),
        "Drift Baseline": DriftForecaster(),
        "Majority-Class (Always UP)": MajorityClassForecaster()
    }

    results = {}
    predictions = {}

    print("\n" + "=" * 95)
    print(f"{'Model Name':<28} | {'RMSE (USD)':<10} | {'MAE (USD)':<10} | {'MAPE %':<8} | {'Dir Acc %':<10} | {'UP F1 %':<9} | {'DOWN F1 %'}")
    print("-" * 95)

    for m_name, model in models.items():
        # Fit on entire dev data
        model.fit(X_dev, y_dev)
        pred_returns = model.predict_log_returns(X_test)
        pred_prices = model.predict_prices(X_test, test_prev_closes)
        
        reg_metrics = compute_regression_metrics(test_actual_closes, pred_prices)
        dir_metrics = compute_directional_metrics(y_test, pred_returns, threshold=0.0)
        
        results[m_name] = {
            "regression": reg_metrics,
            "directional": dir_metrics
        }
        predictions[m_name] = {
            "pred_returns": pred_returns,
            "pred_prices": pred_prices,
            "residuals": test_actual_closes - pred_prices
        }
        
        print(f"{m_name:<28} | {reg_metrics['RMSE']:<10.4f} | {reg_metrics['MAE']:<10.4f} | {reg_metrics['MAPE']:<8.2f} | {dir_metrics['Directional_Accuracy']:<10.2f} | {dir_metrics['UP_F1']:<9.2f} | {dir_metrics['DOWN_F1']:<9.2f}")

    # Statistical Significance Tests against Persistence
    pers_res = predictions["Persistence Baseline"]["residuals"]
    cat_res = predictions["CatBoost (Selected Best)"]["residuals"]
    lstm_res = predictions["Two-Layer LSTM"]["residuals"]

    print("\n--- STATISTICAL SIGNIFICANCE TESTS (vs. PERSISTENCE BASELINE) ---")
    dm_cat = diebold_mariano_test(pers_res, cat_res, loss_type="squared", max_lag=1)
    wilc_cat = paired_wilcoxon_test(pers_res, cat_res)
    print(f"• CatBoost vs. Persistence: DM Stat = {dm_cat['DM_Statistic']:.4f}, p-value = {dm_cat['p_value']:.4f} -> {dm_cat['interpretation']}")
    print(f"  Wilcoxon Signed-Rank Test: Stat = {wilc_cat['statistic']:.1f}, p-value = {wilc_cat['p_value']:.4f}")

    dm_lstm = diebold_mariano_test(pers_res, lstm_res, loss_type="squared", max_lag=1)
    print(f"• Two-Layer LSTM vs. Persistence: DM Stat = {dm_lstm['DM_Statistic']:.4f}, p-value = {dm_lstm['p_value']:.4f} -> {dm_lstm['interpretation']}")

    # Conformal Prediction Calibration & Holdout Coverage
    print("\n--- CONFORMAL PREDICTION INTERVALS (CALIBRATED ON DEV DATA) ---")
    best_model = models["CatBoost (Selected Best)"]
    # Fit on sub-dev and predict on calibration set to get unbiased nonconformity scores
    best_model.fit(X_dev_sub, y_dev_sub)
    cal_pred_ret = best_model.predict_log_returns(X_cal)

    # Re-fit on full dev for final test prediction
    best_model.fit(X_dev, y_dev)
    test_pred_ret = predictions["CatBoost (Selected Best)"]["pred_returns"]

    cp_95 = ConformalPredictor(nominal_coverage=0.95)
    cp_95.calibrate(y_cal, cal_pred_ret, cal_vol)
    _, _, low_95, high_95 = cp_95.predict_intervals(test_pred_ret, test_prev_closes, test_vol)
    cov_95 = cp_95.evaluate_coverage(test_actual_closes, low_95, high_95)

    cp_90 = ConformalPredictor(nominal_coverage=0.90)
    cp_90.calibrate(y_cal, cal_pred_ret, cal_vol)
    _, _, low_90, high_90 = cp_90.predict_intervals(test_pred_ret, test_prev_closes, test_vol)
    cov_90 = cp_90.evaluate_coverage(test_actual_closes, low_90, high_90)

    print(f"• Nominal 95% Interval: Empirical Coverage = {cov_95['Empirical_Coverage']:.2f}% | Avg Width = ${cov_95['Average_Interval_Width']:.2f} | Gap = {cov_95['Coverage_Gap']:+.2f}%")
    print(f"• Nominal 90% Interval: Empirical Coverage = {cov_90['Empirical_Coverage']:.2f}% | Avg Width = ${cov_90['Average_Interval_Width']:.2f} | Gap = {cov_90['Coverage_Gap']:+.2f}%")

    # Generate Publication Figures
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Actual vs Predicted Chart
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, test_actual_closes, label="Actual Closing Price", color="#1E293B", linewidth=1.8)
    plt.plot(test_dates, predictions["CatBoost (Selected Best)"]["pred_prices"], label="CatBoost (Selected Model)", color="#2563EB", linewidth=1.5)
    plt.plot(test_dates, predictions["Persistence Baseline"]["pred_prices"], label="Persistence (Random Walk)", color="#94A3B8", linestyle=":", linewidth=1.2)
    plt.fill_between(test_dates, low_95, high_95, color="#2563EB", alpha=0.12, label=f"95% Conformal Interval ({cov_95['Empirical_Coverage']:.1f}% Empirical)")
    plt.title("Out-of-Sample Holdout Evaluation: CatBoost vs Persistence & 95% Conformal Interval", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price (USD)", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    chart_path1 = os.path.join(artifacts_dir, "holdout_actual_vs_predicted.png")
    plt.savefig(chart_path1, dpi=300)
    plt.close()
    print(f"\nSaved holdout forecast figure: {chart_path1}")

    # 2. Residual Distribution Chart
    plt.figure(figsize=(10, 5))
    sns.kdeplot(cat_res, label=f"CatBoost Residuals (Std=${np.std(cat_res):.2f})", color="#2563EB", fill=True, alpha=0.3)
    sns.kdeplot(pers_res, label=f"Persistence Residuals (Std=${np.std(pers_res):.2f})", color="#64748B", linestyle="--")
    plt.axvline(0, color="red", linestyle=":", alpha=0.7)
    plt.title("Error Residual Distributions on 597 Holdout Sessions", fontsize=13, fontweight="bold")
    plt.xlabel("Prediction Residual: Actual - Predicted (USD)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.tight_layout()
    chart_path2 = os.path.join(artifacts_dir, "holdout_residuals.png")
    plt.savefig(chart_path2, dpi=300)
    plt.close()
    print(f"Saved residual distribution figure: {chart_path2}")

    # 3. Model Benchmark Bar Chart
    plt.figure(figsize=(12, 6))
    model_names_plot = list(results.keys())
    rmse_vals = [results[m]["regression"]["RMSE"] for m in model_names_plot]
    dir_acc_vals = [results[m]["directional"]["Directional_Accuracy"] for m in model_names_plot]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    color = "tab:blue"
    ax1.set_xlabel("Model Architecture", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE in USD (Lower is Better)", color=color, fontsize=11, fontweight="bold")
    bars1 = ax1.bar(np.arange(len(model_names_plot)) - 0.2, rmse_vals, width=0.4, color="#3B82F6", label="RMSE (USD)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(range(len(model_names_plot)))
    ax1.set_xticklabels(model_names_plot, rotation=20, ha="right")

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Directional Accuracy % (Higher is Better)", color=color, fontsize=11, fontweight="bold")
    bars2 = ax2.bar(np.arange(len(model_names_plot)) + 0.2, dir_acc_vals, width=0.4, color="#10B981", label="Directional Acc (%)")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.axhline(53.79, color="orange", linestyle="--", label="Majority-Class Benchmark (53.79%)")
    ax2.set_ylim(40, 65)

    plt.title("Holdout Test Set Benchmark: Predictive Accuracy vs Directional Edge", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    chart_path3 = os.path.join(artifacts_dir, "holdout_benchmark_summary.png")
    plt.savefig(chart_path3, dpi=300)
    plt.close()
    print(f"Saved benchmark summary figure: {chart_path3}")


if __name__ == "__main__":
    run_holdout_evaluation()
