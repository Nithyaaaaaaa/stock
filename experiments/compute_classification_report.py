import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data, split_development_and_holdout
from src.features.technical import compute_all_candidate_features
from src.models.trees import CatBoostForecaster

raw_target, raw_bench = fetch_market_data(ticker="AAPL", benchmark_ticker="SPY", start="2015-01-01")
df_clean = validate_and_clean_data(raw_target, raw_bench)
dev_df, holdout_df, split_meta = split_development_and_holdout(df_clean, holdout_pct=20.0)

all_features = compute_all_candidate_features(df_clean)
usable_features = all_features.iloc[65:-1]

dev_usable = usable_features.loc[usable_features.index < holdout_df.index[0]]
holdout_usable = usable_features.loc[usable_features.index >= holdout_df.index[0]]

# Purge boundary row from development set (1-observation embargo)
dev_purged = dev_usable.iloc[:-1]

d_start = dev_purged.index[0].strftime("%Y-%m-%d")
d_end = dev_purged.index[-1].strftime("%Y-%m-%d")
h_start = holdout_usable.index[0].strftime("%Y-%m-%d")
h_end = holdout_usable.index[-1].strftime("%Y-%m-%d")

print(f"Dev Purged Sessions: {len(dev_purged)} ({d_start} to {d_end})")
print(f"Holdout Usable Sessions: {len(holdout_usable)} ({h_start} to {h_end})")

stationary_features = [
    "Return_Lag_1", "Return_Lag_5", "Log_Return_Lag_1", "Log_Return_Lag_5",
    "Ratio_Close_SMA_5", "Ratio_Close_SMA_20", "Ratio_Close_SMA_50", "Ratio_SMA_5_20",
    "Ratio_Close_EMA_12", "Ratio_Close_EMA_26", "MACD_Normalized",
    "RSI_14_Centered", "Stoch_K_14", "High_Low_Ratio", "Close_Open_Ratio",
    "Vol_10d", "Vol_20d", "Vol_60d", "NATR_14", "Bollinger_Pct_B",
    "Log_Volume_Change", "Volume_Ratio_SMA_20",
    "Benchmark_Return_Lag_1", "Excess_Return_Over_Benchmark", "Rolling_Beta_60"
]

scaler = StandardScaler()
X_dev = scaler.fit_transform(dev_purged[stationary_features].values)
y_dev = dev_purged["Target_Next_Log_Return"].values

X_test = scaler.transform(holdout_usable[stationary_features].values)
y_test = holdout_usable["Target_Next_Log_Return"].values

cat = CatBoostForecaster(iterations=120, depth=4, learning_rate=0.02, random_seed=42)
cat.fit(X_dev, y_dev)
cat_pred_ret = cat.predict_log_returns(X_test)

y_true_dir = (y_test > 0).astype(int)
cat_pred_dir = (cat_pred_ret > 0).astype(int)
maj_pred_dir = np.ones(len(y_test), dtype=int)

print("=" * 80)
print("CATBOOST CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_true_dir, cat_pred_dir, target_names=["DOWN", "UP"], digits=6))

print("=" * 80)
print("CATBOOST CONFUSION MATRIX")
print("=" * 80)
cm_cat = confusion_matrix(y_true_dir, cat_pred_dir, labels=[0, 1])
print(cm_cat)
print(f"True Negative (DOWN -> DOWN): {cm_cat[0, 0]}")
print(f"False Positive (DOWN -> UP):   {cm_cat[0, 1]}")
print(f"False Negative (UP -> DOWN):   {cm_cat[1, 0]}")
print(f"True Positive (UP -> UP):      {cm_cat[1, 1]}")

print("\n" + "=" * 80)
print("ALWAYS-UP BASELINE CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_true_dir, maj_pred_dir, target_names=["DOWN", "UP"], digits=6, zero_division=0))

print("=" * 80)
print("ALWAYS-UP BASELINE CONFUSION MATRIX")
print("=" * 80)
cm_maj = confusion_matrix(y_true_dir, maj_pred_dir, labels=[0, 1])
print(cm_maj)
print(f"True Negative (DOWN -> DOWN): {cm_maj[0, 0]}")
print(f"False Positive (DOWN -> UP):   {cm_maj[0, 1]}")
print(f"False Negative (UP -> DOWN):   {cm_maj[1, 0]}")
print(f"True Positive (UP -> UP):      {cm_maj[1, 1]}")

# Also get dict format for exact values
report_cat_dict = classification_report(y_true_dir, cat_pred_dir, target_names=["DOWN", "UP"], output_dict=True)
report_maj_dict = classification_report(y_true_dir, maj_pred_dir, target_names=["DOWN", "UP"], output_dict=True, zero_division=0)

import json
print("\nJSON FORMAT (EXACT FLOATS):")
print(json.dumps({"CatBoost": report_cat_dict, "Always_UP": report_maj_dict}, indent=2))
