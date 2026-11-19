import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
sys.path.insert(0, "/Users/nithyashreevsuresh/Downloads/st/predictive_insights")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostRegressor

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data, split_development_and_holdout
from src.features.technical import compute_all_candidate_features

raw_target, raw_bench = fetch_market_data(ticker="AAPL", benchmark_ticker="SPY", start="2015-01-01")
df_clean = validate_and_clean_data(raw_target, raw_bench)
dev_df, holdout_df, _ = split_development_and_holdout(df_clean, holdout_pct=20.0)

all_feats = compute_all_candidate_features(df_clean)
usable = all_feats.iloc[65:-1]
dev_usable = usable.loc[usable.index < holdout_df.index[0]]
dev_purged = dev_usable.iloc[:-1]  # 2,326 sessions

stationary_cols = [
    "Return_Lag_1", "Return_Lag_5", "Log_Return_Lag_1", "Log_Return_Lag_5",
    "Ratio_Close_SMA_5", "Ratio_Close_SMA_20", "Ratio_Close_SMA_50", "Ratio_SMA_5_20",
    "Ratio_Close_EMA_12", "Ratio_Close_EMA_26", "MACD_Normalized",
    "RSI_14_Centered", "Stoch_K_14", "High_Low_Ratio", "Close_Open_Ratio",
    "Vol_10d", "Vol_20d", "Vol_60d", "NATR_14", "Bollinger_Pct_B",
    "Log_Volume_Change", "Volume_Ratio_SMA_20",
    "Benchmark_Return_Lag_1", "Excess_Return_Over_Benchmark", "Rolling_Beta_60"
]

scaler = StandardScaler()
X_dev = scaler.fit_transform(dev_purged[stationary_cols].values)
y_dev = dev_purged["Target_Next_Log_Return"].values

inner_split = int(len(X_dev) * 0.8)
X_tr, X_val = X_dev[:inner_split], X_dev[inner_split:]
y_tr, y_val = y_dev[:inner_split], y_dev[inner_split:]
y_val_dir = (y_val > 0).astype(int)

inner_model = CatBoostRegressor(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, random_seed=42, verbose=0)
inner_model.fit(X_tr, y_tr)
pred_val = inner_model.predict(X_val)

best_tau = 0.0
best_bal = -1.0
best_up_rec = 0.0
best_dn_rec = 0.0

for tau in np.linspace(-0.004, 0.004, 81):
    p_dir = (pred_val > tau).astype(int)
    rec_up = np.sum((p_dir == 1) & (y_val_dir == 1)) / (np.sum(y_val_dir == 1) + 1e-9)
    if rec_up >= 0.50:
        bal = balanced_accuracy_score(y_val_dir, p_dir)
        if bal > best_bal:
            best_bal = bal
            best_tau = tau
            best_up_rec = rec_up
            best_dn_rec = np.sum((p_dir == 0) & (y_val_dir == 0)) / (np.sum(y_val_dir == 0) + 1e-9)

tr_start_d = dev_purged.index[0].strftime("%Y-%m-%d")
tr_end_d = dev_purged.index[inner_split-1].strftime("%Y-%m-%d")
val_start_d = dev_purged.index[inner_split].strftime("%Y-%m-%d")
val_end_d = dev_purged.index[-1].strftime("%Y-%m-%d")

print(f"Full Dev Set (N={len(dev_purged)}) Inner Split:")
print(f"  Inner Train N = {len(X_tr)} ({tr_start_d} to {tr_end_d})")
print(f"  Inner Val N   = {len(X_val)} ({val_start_d} to {val_end_d})")
print(f"Learned Deployed Threshold tau*: {best_tau:.6f}")
print(f"Inner Val Balanced Accuracy: {best_bal*100:.2f}% | UP Recall: {best_up_rec*100:.2f}% | DOWN Recall: {best_dn_rec*100:.2f}%")
