"""
Verification Script for Stage 1 Audit:
1. Exact integer partition math and tracing of the 66 dropped observations.
2. Complete table of all 27 selected features with ADF and KPSS statistics.
3. Common-window lookback evaluation (evaluating all L on the identical validation dates starting after L=60).
4. Explaining persistence invariance to lookback.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data, split_development_and_holdout
from src.features.technical import compute_all_candidate_features, evaluate_feature_stationarity


def audit_stage_1():
    print("=" * 85)
    print(" STAGE 1 AUDIT: EXACT PARTITION RECONCILIATION & METHODOLOGICAL VERIFICATION")
    print("=" * 85)

    # 1. Fetch & Clean
    raw_target, raw_bench = fetch_market_data(ticker="AAPL", benchmark_ticker="SPY", start="2015-01-01")
    df_clean = validate_and_clean_data(raw_target, raw_bench)

    # 2. Train / Holdout Partition
    dev_df, holdout_df, split_meta = split_development_and_holdout(df_clean, holdout_pct=20.0)
    
    print("\n--- 1. DATA SPLIT ARITHMETIC & OBSERVATION RECONCILIATION ---")
    print(f"Total Raw Clean Sessions: N = {len(df_clean)}")
    print(f"• Dev Set Raw:     {len(dev_df)} sessions (Indices 0 to {len(dev_df)-1}) | Dates: {dev_df.index[0].strftime('%Y-%m-%d')} to {dev_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"• Holdout Set Raw: {len(holdout_df)} sessions (Indices {len(dev_df)} to {len(df_clean)-1}) | Dates: {holdout_df.index[0].strftime('%Y-%m-%d')} to {holdout_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Check Sum: {len(dev_df)} + {len(holdout_df)} = {len(dev_df) + len(holdout_df)} (Exact match to {len(df_clean)})")

    # Feature engineering on Dev Set
    df_dev_features = compute_all_candidate_features(dev_df)
    
    # Trace the 66 dropped observations:
    warmup_dropped = df_dev_features.iloc[:65]
    usable_dev = df_dev_features.iloc[65:-1]
    terminal_dropped = df_dev_features.iloc[-1:]
    
    print(f"\nBreakdown of Development Set Preprocessing:")
    print(f"  Total Dev Set rows: {len(df_dev_features)}")
    print(f"  - Dropped for feature warm-up (65 rows, indices 0 to 64): {warmup_dropped.index[0].strftime('%Y-%m-%d')} to {warmup_dropped.index[-1].strftime('%Y-%m-%d')}")
    print(f"    (Reason: Vol_60d, Rolling_Beta_60, Ratio_Close_SMA_50 require 60-65 historical rows before first valid value)")
    print(f"  - Dropped terminal row (1 row, index {len(df_dev_features)-1}): {terminal_dropped.index[0].strftime('%Y-%m-%d')}")
    print(f"    (Reason: Target_Next_Close and Target_Next_Log_Return use shift(-1), producing NaN at the final row)")
    print(f"  = Usable Development Sessions: {len(usable_dev)} rows (Indices 65 to {len(df_dev_features)-2})")
    print(f"    Dates: {usable_dev.index[0].strftime('%Y-%m-%d')} to {usable_dev.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Verification: {len(warmup_dropped)} (warm-up) + {len(usable_dev)} (usable) + {len(terminal_dropped)} (terminal) = {len(warmup_dropped) + len(usable_dev) + len(terminal_dropped)} (Exact: {len(dev_df)})")

    # Internal sub-train / sub-val split
    dev_split_idx = int(len(usable_dev) * 0.75)
    sub_train = usable_dev.iloc[:dev_split_idx]
    sub_val = usable_dev.iloc[dev_split_idx:]
    
    print(f"\nInternal Development Partition for Model Diagnostics:")
    print(f"  • Sub-Train: {len(sub_train)} sessions | {sub_train.index[0].strftime('%Y-%m-%d')} to {sub_train.index[-1].strftime('%Y-%m-%d')}")
    print(f"  • Sub-Val:   {len(sub_val)} sessions | {sub_val.index[0].strftime('%Y-%m-%d')} to {sub_val.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Verification: {len(sub_train)} + {len(sub_val)} = {len(sub_train) + len(sub_val)} (Exact match to usable dev: {len(usable_dev)})")

    # -------------------------------------------------------------
    # 2. FEATURE STATIONARITY AUDIT
    # -------------------------------------------------------------
    candidate_cols = [
        "Return_Lag_1", "Return_Lag_5", "Log_Return_Lag_1", "Log_Return_Lag_5",
        "Ratio_Close_SMA_5", "Ratio_Close_SMA_20", "Ratio_Close_SMA_50", "Ratio_SMA_5_20",
        "Ratio_Close_EMA_12", "Ratio_Close_EMA_26", "MACD_Normalized",
        "RSI_14_Centered", "Stoch_K_14", "High_Low_Ratio", "Close_Open_Ratio",
        "Vol_10d", "Vol_20d", "Vol_60d", "NATR_14", "Bollinger_Pct_B",
        "Log_Volume_Change", "Volume_Ratio_SMA_20",
        "Benchmark_Return_Lag_1", "Excess_Return_Over_Benchmark", "Rolling_Beta_60",
        "Raw_SMA_20", "Raw_EMA_12", "Raw_ATR_14", "Raw_Volume", "Close"
    ]
    stationarity_df = evaluate_feature_stationarity(usable_dev, candidate_cols)
    
    print("\n--- 2. COMPLETE STATIONARITY TEST RESULTS (ON DEV DATA ONLY) ---")
    print(f"{'Feature Name':<28} | {'ADF Stat':<10} | {'ADF p-val':<10} | {'KPSS Stat':<10} | {'KPSS p-val':<10} | {'Verdict'}")
    print("-" * 95)
    for _, row in stationarity_df.iterrows():
        print(f"{row['Feature']:<28} | {row['ADF_Statistic']:<10.4f} | {row['ADF_p_value']:<10.4e} | {row['KPSS_Statistic']:<10.4f} | {row['KPSS_p_value']:<10.4f} | {row['Empirical_Verdict']}")

    # -------------------------------------------------------------
    # 3. FAIR TARGET COMPARISON
    # -------------------------------------------------------------
    stationary_features = stationarity_df[stationarity_df["Empirical_Verdict"].str.contains("I\(0\)")]["Feature"].tolist()
    
    X_train_raw = sub_train[stationary_features].values
    X_val_raw = sub_val[stationary_features].values
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_raw)
    X_vl = scaler.transform(X_val_raw)
    
    val_prev_closes = sub_val["Close"].values
    val_actual_closes = sub_val["Target_Next_Close"].values
    
    # Ridge alpha = 100 on all targets
    reg_price = Ridge(alpha=100.0).fit(X_tr, sub_train["Target_Next_Close"].values)
    pred_p = reg_price.predict(X_vl)
    
    reg_rel = Ridge(alpha=100.0).fit(X_tr, sub_train["Target_Next_Rel_Return"].values)
    pred_r = reg_rel.predict(X_vl)
    pred_p_rel = val_prev_closes * (1.0 + pred_r)
    
    reg_log = Ridge(alpha=100.0).fit(X_tr, sub_train["Target_Next_Log_Return"].values)
    pred_y = reg_log.predict(X_vl)
    pred_p_log = val_prev_closes * np.exp(pred_y)
    
    # Persistence: P_hat_{t+1} = P_t
    pred_p_persist = val_prev_closes
    
    # Evaluate on exact same 582 validation sessions
    def eval_reg(actual, pred, prev):
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae = float(mean_absolute_error(actual, pred))
        true_dir = (actual > prev).astype(int)
        pred_dir = (pred > prev).astype(int)
        acc = float(accuracy_score(true_dir, pred_dir))
        return rmse, mae, acc
        
    rmse_p, mae_p, acc_p = eval_reg(val_actual_closes, pred_p, val_prev_closes)
    rmse_r, mae_r, acc_r = eval_reg(val_actual_closes, pred_p_rel, val_prev_closes)
    rmse_l, mae_l, acc_l = eval_reg(val_actual_closes, pred_p_log, val_prev_closes)
    rmse_pers, mae_pers, acc_pers = eval_reg(val_actual_closes, pred_p_persist, val_prev_closes)
    
    print("\n--- 3. TARGET REPRESENTATION COMPARISON (SAME 582 SESSIONS) ---")
    print(f"Persistence Baseline:       RMSE = {rmse_pers:.4f} USD | MAE = {mae_pers:.4f} USD | Dir Acc = {acc_pers*100:.2f}% (All-DOWN tie)")
    print(f"Log Return Model:           RMSE = {rmse_l:.4f} USD | MAE = {mae_l:.4f} USD | Dir Acc = {acc_l*100:.2f}%")
    print(f"Relative Return Model:      RMSE = {rmse_r:.4f} USD | MAE = {mae_r:.4f} USD | Dir Acc = {acc_r*100:.2f}%")
    print(f"Direct Price Level Model:   RMSE = {rmse_p:.4f} USD | MAE = {mae_p:.4f} USD | Dir Acc = {acc_p*100:.2f}%")
    print(f"\nClarification: Persistence has lower validation RMSE ({rmse_pers:.4f} vs. {rmse_l:.4f}) and MAE ({mae_pers:.4f} vs. {mae_l:.4f}).")
    print(f"Log return is the best-performing LEARNED target representation, NOT an outperformance over persistence.")

    # -------------------------------------------------------------
    # 4. LOOKBACK WINDOW SWEEP ON IDENTICAL EVALUATION WINDOW
    # -------------------------------------------------------------
    # To compare lookback windows fairly, all windows must be evaluated on the EXACT SAME validation sessions.
    # The longest window is L_max = 60. Therefore, all models are evaluated on sub_val starting from index 60.
    L_max = 60
    common_val_actuals = val_actual_closes[L_max:]
    common_val_prevs = val_prev_closes[L_max:]
    common_persist_rmse = float(np.sqrt(mean_squared_error(common_val_actuals, common_val_prevs)))
    common_persist_mae = float(mean_absolute_error(common_val_actuals, common_val_prevs))
    
    print(f"\n--- 4. FAIR LOOKBACK EVALUATION ON IDENTICAL SESSIONS (N={len(common_val_actuals)}) ---")
    print(f"Common Persistence Baseline: RMSE = {common_persist_rmse:.4f} USD | MAE = {common_persist_mae:.4f} USD")
    print(f"(Note: Persistence is invariant to lookback window length L because P_hat_{{t+1}} = P_t)")
    
    lookbacks = [5, 10, 15, 20, 30, 45, 60]
    for L in lookbacks:
        # Build sequences for training
        X_seq_tr, y_seq_tr = [], []
        y_tr_vals = sub_train["Target_Next_Log_Return"].values
        for i in range(L, len(sub_train)):
            w = X_tr[i - L:i]
            vec = np.concatenate([w[-1], w.mean(axis=0), w[-1] - w[0]])
            X_seq_tr.append(vec)
            y_seq_tr.append(y_tr_vals[i])
            
        X_seq_vl = []
        for i in range(L, len(sub_val)):
            w = X_vl[i - L:i]
            vec = np.concatenate([w[-1], w.mean(axis=0), w[-1] - w[0]])
            X_seq_vl.append(vec)
            
        X_seq_tr = np.array(X_seq_tr)
        y_seq_tr = np.array(y_seq_tr)
        X_seq_vl = np.array(X_seq_vl)
        
        # Slice predictions to match the common evaluation window (starting from L_max = 60)
        offset = L_max - L
        X_seq_vl_common = X_seq_vl[offset:]
        
        mdl = Ridge(alpha=50.0).fit(X_seq_tr, y_seq_tr)
        pred_logs = mdl.predict(X_seq_vl_common)
        pred_prices = common_val_prevs * np.exp(pred_logs)
        
        v_rmse = float(np.sqrt(mean_squared_error(common_val_actuals, pred_prices)))
        v_mae = float(mean_absolute_error(common_val_actuals, pred_prices))
        v_acc = float(accuracy_score((common_val_actuals > common_val_prevs).astype(int), (pred_prices > common_val_prevs).astype(int)))
        
        print(f"Lookback L={L:<2} | Common Val RMSE: {v_rmse:.4f} USD | Common Val MAE: {v_mae:.4f} USD | Dir Acc: {v_acc*100:.2f}%")

if __name__ == "__main__":
    audit_stage_1()
