"""
Experiment Script: Stage 1 Research & Diagnostic Study
1. Feature Stationarity Evaluation (ADF & KPSS tests on dev set only).
2. Target Formulation Study (Price vs. Relative Return vs. Log Return vs. Direction).
3. Lookback Window Optimization (L in [5, 10, 15, 20, 30, 45, 60]).
4. Market Regime Robustness Analysis.

CRITICAL: The final 20% holdout test set remains completely UNTOUCHED.
All evaluations in this study use an internal chronological validation split of the development set.
"""

import os
import sys

# Ensure local src directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data, split_development_and_holdout
from src.features.technical import compute_all_candidate_features, evaluate_feature_stationarity


def run_stage_1_research(ticker: str = "AAPL", benchmark: str = "SPY"):
    print("=" * 80)
    print(f" STAGE 1: COMPREHENSIVE RESEARCH, STATIONARITY & TARGET SELECTION STUDY")
    print(f" Target Asset: {ticker} | Macro Benchmark: {benchmark}")
    print("=" * 80)

    # 1. Fetch data
    raw_target, raw_bench = fetch_market_data(ticker=ticker, benchmark_ticker=benchmark, start="2015-01-01")
    df_clean = validate_and_clean_data(raw_target, raw_bench)

    # 2. Strict chronological split: 80% Dev / 20% Final Untouched Holdout
    dev_df, holdout_df, split_meta = split_development_and_holdout(df_clean, holdout_pct=20.0)
    print(f"\n[DATA SPLIT AUDIT]")
    print(f"Total Dataset Sessions: {split_meta['total_sessions']} (Dates: {df_clean.index[0].strftime('%Y-%m-%d')} to {df_clean.index[-1].strftime('%Y-%m-%d')})")
    print(f"• Development Set    : {split_meta['dev_sessions']} sessions ({split_meta['dev_start']} to {split_meta['dev_end']}) [80%]")
    print(f"• Untouched Holdout  : {split_meta['holdout_sessions']} sessions ({split_meta['holdout_start']} to {split_meta['holdout_end']}) [20% FROZEN]")

    # 3. Compute Features strictly on Development Data
    df_dev_features = compute_all_candidate_features(dev_df)
    
    # Drop warm-up rows (first 65 rows due to 60-day rolling indicators) and terminal row (where Target_Next_Close is NaN)
    df_dev_clean = df_dev_features.iloc[65:-1].copy()
    print(f"\nUsable Development Sessions after feature warm-up: {len(df_dev_clean)}")

    # -------------------------------------------------------------
    # 4. FORMAL FEATURE STATIONARITY AUDIT (ADF & KPSS)
    # -------------------------------------------------------------
    candidate_cols = [
        "Return_Lag_1", "Return_Lag_5", "Log_Return_Lag_1", "Log_Return_Lag_5",
        "Ratio_Close_SMA_5", "Ratio_Close_SMA_20", "Ratio_Close_SMA_50", "Ratio_SMA_5_20",
        "Ratio_Close_EMA_12", "Ratio_Close_EMA_26", "MACD_Normalized",
        "RSI_14_Centered", "Stoch_K_14", "High_Low_Ratio", "Close_Open_Ratio",
        "Vol_10d", "Vol_20d", "Vol_60d", "NATR_14", "Bollinger_Pct_B",
        "Log_Volume_Change", "Volume_Ratio_SMA_20",
        "Benchmark_Return_Lag_1", "Excess_Return_Over_Benchmark", "Rolling_Beta_60",
        # Unnormalized levels for contrast:
        "Raw_SMA_20", "Raw_EMA_12", "Raw_ATR_14", "Raw_Volume", "Close"
    ]

    print("\n" + "=" * 80)
    print(" 1. FORMAL FEATURE STATIONARITY AUDIT (ON DEV SET ONLY)")
    print("=" * 80)
    stationarity_df = evaluate_feature_stationarity(df_dev_clean, candidate_cols)
    
    print(f"{'Feature Name':<30} | {'ADF p-val':<10} | {'KPSS p-val':<10} | {'Empirical Verdict':<32}")
    print("-" * 90)
    for _, row in stationarity_df.iterrows():
        print(f"{row['Feature']:<30} | {row['ADF_p_value']:<10.4e} | {row['KPSS_p_value']:<10.4f} | {row['Empirical_Verdict']:<32}")

    stationary_features = stationarity_df[stationarity_df["Empirical_Verdict"].str.contains("I\(0\)")]["Feature"].tolist()
    non_stationary_features = stationarity_df[stationarity_df["Empirical_Verdict"].str.contains("Non-Stationary")]["Feature"].tolist()
    print(f"\nSummary: {len(stationary_features)} features pass stationarity checks. {len(non_stationary_features)} features exhibit non-stationary unit roots (e.g., raw levels: {non_stationary_features}).")

    # -------------------------------------------------------------
    # 5. TARGET FORMULATION INVESTIGATION
    # -------------------------------------------------------------
    # We partition dev_df into Sub-Train (first 75% of dev) and Sub-Val (terminal 25% of dev)
    dev_split_idx = int(len(df_dev_clean) * 0.75)
    sub_train = df_dev_clean.iloc[:dev_split_idx].copy()
    sub_val = df_dev_clean.iloc[dev_split_idx:].copy()
    
    print("\n" + "=" * 80)
    print(" 2. TARGET FORMULATION STUDY (ON INTERNAL VALIDATION SPLIT)")
    print(f" Sub-Train: {len(sub_train)} sessions ({sub_train.index[0].strftime('%Y-%m-%d')} to {sub_train.index[-1].strftime('%Y-%m-%d')})")
    print(f" Sub-Val  : {len(sub_val)} sessions ({sub_val.index[0].strftime('%Y-%m-%d')} to {sub_val.index[-1].strftime('%Y-%m-%d')})")
    print("=" * 80)

    # Clean feature matrix using verified stationary features only
    X_train_raw = sub_train[stationary_features].values
    X_val_raw = sub_val[stationary_features].values
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_raw)
    X_vl = scaler.transform(X_val_raw)
    
    val_prev_closes = sub_val["Close"].values
    val_actual_closes = sub_val["Target_Next_Close"].values  # actual close at t+1

    # Target 1: Direct Price Level
    y_tr_price = sub_train["Target_Next_Close"].values
    reg_price = Ridge(alpha=100.0).fit(X_tr, y_tr_price)
    pred_price_direct = reg_price.predict(X_vl)
    rmse_price = np.sqrt(mean_squared_error(val_actual_closes, pred_price_direct))
    mae_price = mean_absolute_error(val_actual_closes, pred_price_direct)
    acc_price = accuracy_score((val_actual_closes > val_prev_closes).astype(int), (pred_price_direct > val_prev_closes).astype(int))

    # Target 2: Relative Daily Return r_{t+1} = (P_{t+1} - P_t)/P_t
    y_tr_rel = sub_train["Target_Next_Rel_Return"].values
    reg_rel = Ridge(alpha=100.0).fit(X_tr, y_tr_rel)
    pred_rel_return = reg_rel.predict(X_vl)
    pred_price_rel = val_prev_closes * (1.0 + pred_rel_return)
    rmse_rel = np.sqrt(mean_squared_error(val_actual_closes, pred_price_rel))
    mae_rel = mean_absolute_error(val_actual_closes, pred_price_rel)
    acc_rel = accuracy_score((val_actual_closes > val_prev_closes).astype(int), (pred_price_rel > val_prev_closes).astype(int))

    # Target 3: Continuous Log Return y_{t+1} = ln(P_{t+1} / P_t)
    y_tr_log = sub_train["Target_Next_Log_Return"].values
    y_vl_log = sub_val["Target_Next_Log_Return"].values
    reg_log = Ridge(alpha=100.0).fit(X_tr, y_tr_log)
    pred_log_return = reg_log.predict(X_vl)
    pred_price_log = val_prev_closes * np.exp(pred_log_return)
    rmse_log = np.sqrt(mean_squared_error(val_actual_closes, pred_price_log))
    mae_log = mean_absolute_error(val_actual_closes, pred_price_log)
    acc_log = accuracy_score((val_actual_closes > val_prev_closes).astype(int), (pred_price_log > val_prev_closes).astype(int))

    # Target 4: Persistence Benchmark on same validation set
    pred_price_persist = val_prev_closes
    rmse_persist = np.sqrt(mean_squared_error(val_actual_closes, pred_price_persist))
    mae_persist = mean_absolute_error(val_actual_closes, pred_price_persist)
    acc_persist = accuracy_score((val_actual_closes > val_prev_closes).astype(int), np.zeros_like(val_prev_closes))

    print(f"{'Target Formulation':<30} | {'RMSE (USD)':<12} | {'MAE (USD)':<12} | {'Directional Acc':<16} | {'Stability Verdict'}")
    print("-" * 90)
    print(f"{'Direct Price Level P_{t+1}':<30} | {rmse_price:<12.4f} | {mae_price:<12.4f} | {acc_price*100:<15.2f}% | Prone to drift / out-of-bounds")
    print(f"{'Relative Return r_{t+1}':<30} | {rmse_rel:<12.4f} | {mae_rel:<12.4f} | {acc_rel*100:<15.2f}% | Stationary, exact price conversion")
    print(f"{'Continuous Log Return y_{t+1}':<30} | {rmse_log:<12.4f} | {mae_log:<12.4f} | {acc_log*100:<15.2f}% | Statistically symmetric, additive")
    print(f"{'Persistence Benchmark':<30} | {rmse_persist:<12.4f} | {mae_persist:<12.4f} | {acc_persist*100:<15.2f}% | Naive all-DOWN baseline")

    # -------------------------------------------------------------
    # 6. SYSTEMATIC LOOKBACK WINDOW OPTIMIZATION STUDY
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 3. LOOKBACK WINDOW OPTIMIZATION (VALIDATION SWEEP: L in [5, 10, 15, 20, 30, 45, 60])")
    print("=" * 80)
    
    lookback_candidates = [5, 10, 15, 20, 30, 45, 60]
    lookback_results = []

    for L in lookback_candidates:
        # Create sequences of length L from stationary features
        # Sequence input: [t - L + 1 : t] -> Target: Target_Next_Log_Return at row t
        X_seq_tr, y_seq_tr = [], []
        for i in range(L, len(sub_train)):
            # Flatten or mean-aggregate across window to evaluate linear & tabular lookback capacity
            window = X_tr[i - L:i]
            # Capture both the latest state (t) and temporal trajectory (mean, slope, min, max)
            feat_vector = np.concatenate([
                window[-1],                         # Current state at t
                window.mean(axis=0),                # Average momentum across window
                window[-1] - window[0],             # Net change across window
            ])
            X_seq_tr.append(feat_vector)
            y_seq_tr.append(y_tr_log[i])
            
        X_seq_vl, y_seq_vl = [], []
        for i in range(L, len(sub_val)):
            window = X_vl[i - L:i]
            feat_vector = np.concatenate([
                window[-1],
                window.mean(axis=0),
                window[-1] - window[0],
            ])
            X_seq_vl.append(feat_vector)
            y_seq_vl.append(y_vl_log[i])
            
        X_seq_tr = np.array(X_seq_tr)
        y_seq_tr = np.array(y_seq_tr)
        X_seq_vl = np.array(X_seq_vl)
        
        # Test targets for this window length
        val_actuals = val_actual_closes[L:]
        val_prevs = val_prev_closes[L:]
        
        model_sweep = Ridge(alpha=50.0).fit(X_seq_tr, y_seq_tr)
        pred_logs = model_sweep.predict(X_seq_vl)
        pred_prices = val_prevs * np.exp(pred_logs)
        
        l_rmse = np.sqrt(mean_squared_error(val_actuals, pred_prices))
        l_mae = mean_absolute_error(val_actuals, pred_prices)
        l_acc = accuracy_score((val_actuals > val_prevs).astype(int), (pred_prices > val_prevs).astype(int))
        
        lookback_results.append({
            "Lookback_L": L,
            "Validation_RMSE": l_rmse,
            "Validation_MAE": l_mae,
            "Directional_Acc": l_acc,
        })
        print(f"Window L={L:<2} days | Val RMSE: {l_rmse:.4f} USD | Val MAE: {l_mae:.4f} USD | Dir Acc: {l_acc*100:.2f}%")

    best_l = min(lookback_results, key=lambda x: x["Validation_RMSE"])
    print(f"\nEmpirical Optimal Lookback Window on Validation Split: L* = {best_l['Lookback_L']} days (RMSE: {best_l['Validation_RMSE']:.4f})")

    # -------------------------------------------------------------
    # 7. REGIME ROBUSTNESS AUDIT (ON DEV SET)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 4. MARKET REGIME ROBUSTNESS AUDIT (WITHIN DEV SET)")
    print("=" * 80)
    
    # Split Dev Set into 3 historical regimes:
    # Regime 1: 2015-2019 (Pre-COVID Bull)
    # Regime 2: 2020-2021 (COVID Crisis & Recovery)
    # Regime 3: 2022-2024 (Fed Inflation & Tightening)
    r1 = df_dev_clean.loc[:"2019-12-31"]
    r2 = df_dev_clean.loc["2020-01-01":"2021-12-31"]
    r3 = df_dev_clean.loc["2022-01-01":]
    
    for r_name, r_df in [("Regime 1: Pre-COVID (2015-2019)", r1), ("Regime 2: COVID Crisis (2020-2021)", r2), ("Regime 3: Fed Tightening (2022-2024)", r3)]:
        r_ret = r_df["Target_Log_Return"].dropna()
        print(f"{r_name:<38} | N={len(r_df):<5} | Mean Return: {r_ret.mean()*100:+.3f}% | Volatility: {r_ret.std()*100:.2f}% | Kurtosis: {r_ret.kurtosis():.2f}")

    print("\n" + "=" * 80)
    print(" STAGE 1 EXPERIMENTAL EXECUTION COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    run_stage_1_research()
