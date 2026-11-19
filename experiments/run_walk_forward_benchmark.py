"""
Walk-Forward Cross-Validation Multi-Model Benchmark:
Evaluates Baselines, Statistical Models, Tree Ensembles, Deep Learning Models,
and Hybrid Ensembles across 5 Expanding Folds on Development Data (2,325 sessions).
Holdout test set (597 sessions) is 100% FROZEN and untouched.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
try:
    torch.set_num_threads(1)
except Exception:
    pass

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data, split_development_and_holdout
from src.features.technical import compute_all_candidate_features, evaluate_feature_stationarity
from src.evaluation.walk_forward import WalkForwardValidator
from src.evaluation.statistical_tests import diebold_mariano_test

# Model imports
from src.models.baselines import (
    PersistenceForecaster,
    DriftForecaster,
    SMAForecaster,
    MajorityClassForecaster
)
from src.models.statistical import ARIMAForecaster, ExponentialSmoothingForecaster
from src.models.trees import (
    RandomForestForecaster,
    XGBoostForecaster,
    LightGBMForecaster,
    CatBoostForecaster
)
from src.models.deep_learning import (
    LSTMForecaster,
    GRUForecaster,
    CNNLSTMForecaster,
    TCNForecaster
)
from src.models.ensemble import BlendingForecaster, StackingForecaster


def run_benchmark():
    print("=" * 95)
    print(" PHASE 3: EXPANDING-WINDOW WALK-FORWARD CROSS-VALIDATION BENCHMARK")
    print(" (Evaluated Strictly on 2,325 Development Sessions | Holdout 597 Sessions FROZEN)")
    print("=" * 95)

    # 1. Fetch & Prepare Data
    raw_target, raw_bench = fetch_market_data(ticker="AAPL", benchmark_ticker="SPY", start="2015-01-01")
    df_clean = validate_and_clean_data(raw_target, raw_bench)
    dev_df, holdout_df, _ = split_development_and_holdout(df_clean, holdout_pct=20.0)

    print(f"\nTotal Clean Sessions: {len(df_clean)}")
    print(f"Development Set:      {len(dev_df)} sessions (2015-01-02 to 2024-07-03)")
    print(f"Holdout Set (FROZEN): {len(holdout_df)} sessions (2024-07-05 to 2026-11-18)")

    # Compute features on dev
    df_dev_features = compute_all_candidate_features(dev_df)
    usable_dev = df_dev_features.iloc[65:-1]
    print(f"Usable Development Sessions: {len(usable_dev)} (Indices 65 to 2389)\n")

    # Select stationary features
    candidate_cols = [
        "Return_Lag_1", "Return_Lag_5", "Log_Return_Lag_1", "Log_Return_Lag_5",
        "Ratio_Close_SMA_5", "Ratio_Close_SMA_20", "Ratio_Close_SMA_50", "Ratio_SMA_5_20",
        "Ratio_Close_EMA_12", "Ratio_Close_EMA_26", "MACD_Normalized",
        "RSI_14_Centered", "Stoch_K_14", "High_Low_Ratio", "Close_Open_Ratio",
        "Vol_10d", "Vol_20d", "Vol_60d", "NATR_14", "Bollinger_Pct_B",
        "Log_Volume_Change", "Volume_Ratio_SMA_20",
        "Benchmark_Return_Lag_1", "Excess_Return_Over_Benchmark", "Rolling_Beta_60"
    ]
    stat_df = evaluate_feature_stationarity(usable_dev, candidate_cols)
    stationary_features = stat_df[stat_df["Empirical_Verdict"].str.contains(r"I\(0\)")]["Feature"].tolist()
    print(f"Selected {len(stationary_features)} verified stationary features for benchmarking.")

    # Walk-forward setup
    wf = WalkForwardValidator(n_splits=5, min_train_ratio=0.50)
    folds = wf.generate_folds(usable_dev)
    print("\nWalk-Forward Expanding Folds Structure:")
    for f in folds:
        print(f"  Fold {f['fold']}: Train {f['train_count']:4d} sessions ({f['train_start_date']} to {f['train_end_date']}) -> "
              f"Val {f['val_count']:3d} sessions ({f['val_start_date']} to {f['val_end_date']})")

    # Define model factories
    models_to_benchmark = [
        ("Persistence Baseline", lambda: PersistenceForecaster()),
        ("Drift Baseline", lambda: DriftForecaster()),
        ("SMA(20) Baseline", lambda: SMAForecaster(window=20)),
        ("Majority-Class (Always UP)", lambda: MajorityClassForecaster()),
        ("ARIMA(1,0,0)", lambda: ARIMAForecaster(order=(1, 0, 0))),
        ("Exponential Smoothing", lambda: ExponentialSmoothingForecaster(smoothing_level=0.1)),
        ("Random Forest", lambda: RandomForestForecaster(n_estimators=100, max_depth=5, random_state=42)),
        ("XGBoost", lambda: XGBoostForecaster(n_estimators=80, max_depth=3, learning_rate=0.02, random_state=42)),
        ("LightGBM", lambda: LightGBMForecaster(n_estimators=80, max_depth=3, num_leaves=15, learning_rate=0.02, random_state=42)),
        ("CatBoost", lambda: CatBoostForecaster(iterations=100, depth=4, learning_rate=0.02, random_seed=42)),
        ("Two-Layer LSTM", lambda: LSTMForecaster(seq_len=20, hidden_dim=32, num_layers=2, epochs=25, lr=0.003, seed=42)),
        ("GRU Network", lambda: GRUForecaster(seq_len=20, hidden_dim=32, num_layers=2, epochs=25, lr=0.003, seed=42)),
        ("CNN-LSTM Hybrid", lambda: CNNLSTMForecaster(seq_len=20, conv_filters=32, lstm_hidden=32, epochs=25, lr=0.003, seed=42)),
        ("Temporal ConvNet (TCN)", lambda: TCNForecaster(seq_len=20, num_channels=(32, 32), epochs=25, lr=0.003, seed=42)),
        ("Blend (LGBM + CatBoost + LSTM)", lambda: BlendingForecaster([
            LightGBMForecaster(n_estimators=80, max_depth=3, learning_rate=0.02, random_state=42),
            CatBoostForecaster(iterations=100, depth=4, learning_rate=0.02, random_seed=42),
            LSTMForecaster(seq_len=20, hidden_dim=32, num_layers=2, epochs=25, lr=0.003, seed=42)
        ], weights=[0.4, 0.4, 0.2], name="Blend (LGBM + CatBoost + LSTM)")),
        ("Stacking (Ridge Meta-Learner)", lambda: StackingForecaster([
            LightGBMForecaster(n_estimators=80, max_depth=3, learning_rate=0.02, random_state=42),
            CatBoostForecaster(iterations=100, depth=4, learning_rate=0.02, random_seed=42),
            RandomForestForecaster(n_estimators=100, max_depth=5, random_state=42)
        ], alpha=1.0, name="Stacking (Ridge Meta-Learner)"))
    ]

    benchmark_records = []
    benchmark_details = {}

    print("\n" + "=" * 95)
    print(f"{'Model Architecture':<32} | {'RMSE (USD)':<10} | {'MAE (USD)':<10} | {'Dir Acc %':<10} | {'UP F1 %':<9} | {'Fit (s)':<7} | {'Lat (ms)'}")
    print("-" * 95)

    persistence_oos_error = None

    for name, factory in models_to_benchmark:
        print(f"Starting {name}...", flush=True)
        res = wf.evaluate_model(
            model_factory=factory,
            df=usable_dev,
            feature_cols=stationary_features
        )
        agg = res["aggregated"]
        ov_reg = agg["overall_regression"]
        ov_dir = agg["overall_directional"]
        
        # Save OOS residuals for DM test against persistence
        oos_preds = res["oos_predictions"]
        oos_err = oos_preds["actual_prices"] - oos_preds["pred_prices"]
        
        if name == "Persistence Baseline":
            persistence_oos_error = oos_err

        dm_p_val = np.nan
        dm_stat = np.nan
        if persistence_oos_error is not None and name != "Persistence Baseline":
            dm_res = diebold_mariano_test(persistence_oos_error, oos_err, loss_type="squared", max_lag=1)
            dm_p_val = dm_res["p_value"]
            dm_stat = dm_res["DM_Statistic"]

        record = {
            "Model": name,
            "Overall_RMSE": ov_reg["RMSE"],
            "Overall_MAE": ov_reg["MAE"],
            "Overall_MAPE": ov_reg["MAPE"],
            "Overall_R2": ov_reg["R2"],
            "Mean_Fold_RMSE": agg["mean_fold_rmse"],
            "Std_Fold_RMSE": agg["std_fold_rmse"],
            "Directional_Accuracy": ov_dir["Directional_Accuracy"],
            "Balanced_Accuracy": ov_dir["Balanced_Accuracy"],
            "UP_Precision": ov_dir["UP_Precision"],
            "UP_Recall": ov_dir["UP_Recall"],
            "UP_F1": ov_dir["UP_F1"],
            "DOWN_Precision": ov_dir["DOWN_Precision"],
            "DOWN_Recall": ov_dir["DOWN_Recall"],
            "DOWN_F1": ov_dir["DOWN_F1"],
            "DM_Stat_vs_Persistence": dm_stat,
            "DM_p_value_vs_Persistence": dm_p_val,
            "Total_Fit_Time_Sec": agg["total_fit_time_sec"],
            "Inference_Latency_ms": agg["avg_inference_latency_ms"]
        }
        benchmark_records.append(record)
        benchmark_details[name] = res

        print(f"{name:<32} | {ov_reg['RMSE']:<10.4f} | {ov_reg['MAE']:<10.4f} | {ov_dir['Directional_Accuracy']:<10.2f} | {ov_dir['UP_F1']:<9.2f} | {agg['total_fit_time_sec']:<7.2f} | {agg['avg_inference_latency_ms']:<6.3f}")

    df_results = pd.DataFrame(benchmark_records)
    
    # Save results to artifacts
    out_csv = os.path.join(PROJECT_ROOT, "artifacts", "walk_forward_benchmark_results.csv")
    df_results.to_csv(out_csv, index=False)
    print("\n" + "=" * 95)
    print(f"Benchmark completed successfully! Results saved to: {out_csv}")
    
    # Save per-fold details
    fold_records = []
    for m_name, res in benchmark_details.items():
        for f in res["folds"]:
            fold_records.append({
                "Model": m_name,
                "Fold": f["fold"],
                "Train_Dates": f["train_dates"],
                "Val_Dates": f["val_dates"],
                "Regime_Vol_20d": f["regime_vol_20d"],
                "RMSE": f["regression"]["RMSE"],
                "MAE": f["regression"]["MAE"],
                "Dir_Acc": f["directional"]["Directional_Accuracy"],
                "UP_F1": f["directional"]["UP_F1"],
                "DOWN_F1": f["directional"]["DOWN_F1"]
            })
    df_folds = pd.DataFrame(fold_records)
    out_folds_csv = os.path.join(PROJECT_ROOT, "artifacts", "walk_forward_fold_details.csv")
    df_folds.to_csv(out_folds_csv, index=False)
    print(f"Fold breakdown saved to: {out_folds_csv}")

    # Top-3 ranking analysis
    print("\n--- TOP RANKED MODELS BY VALIDATION ACCURACY & RMSE ---")
    df_sorted_dir = df_results.sort_values(by="Directional_Accuracy", ascending=False).reset_index(drop=True)
    for idx, row in df_sorted_dir.head(5).iterrows():
        print(f"{idx+1}. {row['Model']:<30} | Dir Acc: {row['Directional_Accuracy']:.2f}% | Balanced Acc: {row['Balanced_Accuracy']:.2f}% | RMSE: {row['Overall_RMSE']:.4f} USD | DM p: {row['DM_p_value_vs_Persistence']}")

    df_sorted_rmse = df_results.sort_values(by="Overall_RMSE", ascending=True).reset_index(drop=True)
    print("\n--- TOP RANKED MODELS BY OUT-OF-SAMPLE RMSE ---")
    for idx, row in df_sorted_rmse.head(5).iterrows():
        print(f"{idx+1}. {row['Model']:<30} | RMSE: {row['Overall_RMSE']:.4f} USD | MAE: {row['Overall_MAE']:.4f} USD | Dir Acc: {row['Directional_Accuracy']:.2f}% | DM p: {row['DM_p_value_vs_Persistence']}")


if __name__ == "__main__":
    run_benchmark()
