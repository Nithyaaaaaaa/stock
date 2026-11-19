"""
Comprehensive Directional Improvement Experimentation Script:
Executes rigorous 5-fold expanding walk-forward cross-validation on dev_purged (N=2,326).
Compares:
- Baselines: Always-UP, Existing CatBoost Regressor (tau=0.0)
- CatBoost Classifier with class weights (1.0, 1.2, 1.5, 2.0, Balanced)
- Probability threshold tuning on training data (inner-CV or train-cal split)
- Model comparisons: CatBoost, XGBoost, Random Forest
- Feature sets: Standard 25 vs Standard 25 + 6 Downside features
- Comprehensive metrics: Overall Acc, Balanced Acc, Macro F1, Weighted F1,
  UP Precision, Recall, F1, Support; DOWN Precision, Recall, F1, Support; Confusion Matrix.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, f1_score
)

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data, split_development_and_holdout
from src.features.technical import compute_all_candidate_features
from src.evaluation.walk_forward import WalkForwardValidator


def engineer_downside_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ret = out["Target_Relative_Return"]
    close = out["Close"]
    vol = out["Volume"]
    
    # 1. Downside Semivariance / Semi-volatility (20-day rolling)
    down_ret = ret.clip(upper=0.0)
    out["Downside_Vol_20d"] = np.sqrt((down_ret ** 2).rolling(20).mean())
    
    # 2. Drawdown from 20-day Rolling High (distance below recent peak)
    high_20 = close.rolling(20).max()
    out["Drawdown_20d"] = (close - high_20) / (high_20 + 1e-9)
    
    # 3. Down-to-Total Volume Ratio (10-day rolling)
    is_down = (ret < 0).astype(float)
    down_vol = is_down * vol
    out["Down_Volume_Ratio_10d"] = down_vol.rolling(10).sum() / (vol.rolling(10).sum() + 1e-9)
    
    # 4. Negative Return Streak (past 3 days down indicator)
    out["Down_Streak_3d"] = ((ret < 0).astype(int) + (ret.shift(1) < 0).astype(int) + (ret.shift(2) < 0).astype(int)) / 3.0
    
    # 5. Volatility Shock: Ratio of 10-day volatility to 60-day volatility
    vol_10 = out["Vol_10d"]
    vol_60 = out["Vol_60d"]
    out["Vol_Shock_10_60"] = (vol_10 / (vol_60 + 1e-9)) - 1.0
    
    # 6. Return Momentum Acceleration (3-day return difference)
    ret_3d = close.pct_change(3)
    out["Return_Acceleration_3d"] = ret_3d - ret_3d.shift(3)
    
    return out


def evaluate_directional_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    macro_prec = float(np.mean(prec) * 100.0)
    macro_rec = float(np.mean(rec) * 100.0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    return {
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Macro_Precision": macro_prec,
        "Macro_Recall": macro_rec,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "UP_Precision": prec[1] * 100.0,
        "UP_Recall": rec[1] * 100.0,
        "UP_F1": f1[1] * 100.0,
        "UP_Support": int(support[1]),
        "DOWN_Precision": prec[0] * 100.0,
        "DOWN_Recall": rec[0] * 100.0,
        "DOWN_F1": f1[0] * 100.0,
        "DOWN_Support": int(support[0]),
        "Confusion_Matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }


def run_experiments():
    print("Fetching and preparing data...")
    raw_target, raw_bench = fetch_market_data(ticker="AAPL", benchmark_ticker="SPY", start="2015-01-01")
    df_clean = validate_and_clean_data(raw_target, raw_bench)
    dev_df, holdout_df, _ = split_development_and_holdout(df_clean, holdout_pct=20.0)

    base_features = compute_all_candidate_features(df_clean)
    all_features = engineer_downside_features(base_features)

    usable = all_features.iloc[65:-1]
    dev_usable = usable.loc[usable.index < holdout_df.index[0]]
    dev_purged = dev_usable.iloc[:-1]  # 2,326 sessions

    standard_features = [
        "Return_Lag_1", "Return_Lag_5", "Log_Return_Lag_1", "Log_Return_Lag_5",
        "Ratio_Close_SMA_5", "Ratio_Close_SMA_20", "Ratio_Close_SMA_50", "Ratio_SMA_5_20",
        "Ratio_Close_EMA_12", "Ratio_Close_EMA_26", "MACD_Normalized",
        "RSI_14_Centered", "Stoch_K_14", "High_Low_Ratio", "Close_Open_Ratio",
        "Vol_10d", "Vol_20d", "Vol_60d", "NATR_14", "Bollinger_Pct_B",
        "Log_Volume_Change", "Volume_Ratio_SMA_20",
        "Benchmark_Return_Lag_1", "Excess_Return_Over_Benchmark", "Rolling_Beta_60"
    ]
    
    downside_features = [
        "Downside_Vol_20d", "Drawdown_20d", "Down_Volume_Ratio_10d",
        "Down_Streak_3d", "Vol_Shock_10_60", "Return_Acceleration_3d"
    ]
    
    expanded_features = standard_features + downside_features

    # Setup walk-forward validator
    wf = WalkForwardValidator(n_splits=5, min_train_ratio=0.50, purge_boundary=True)
    folds = wf.generate_folds(dev_purged)

    print(f"Generated {len(folds)} expanding folds across {len(dev_purged)} sessions.")

    # Define experimental models
    # Each experiment has: name, model_type ('reg' or 'clf'), model_fn, features, threshold_strategy
    experiments = [
        {
            "id": "EXP0_ALWAYS_UP",
            "name": "Always-UP Baseline",
            "type": "baseline",
            "features": standard_features,
        },
        {
            "id": "EXP1_CATBOOST_REG_DEFAULT",
            "name": "CatBoost Regressor (Baseline, tau=0.0)",
            "type": "reg",
            "model_fn": lambda: CatBoostRegressor(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, random_seed=42, verbose=0),
            "features": standard_features,
            "threshold": 0.0
        },
        {
            "id": "EXP2_CATBOOST_REG_TUNED_TAU",
            "name": "CatBoost Regressor + Train-Tuned Return Threshold",
            "type": "reg_tuned",
            "model_fn": lambda: CatBoostRegressor(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, random_seed=42, verbose=0),
            "features": standard_features,
        },
        {
            "id": "EXP3_CATBOOST_CLF_UNWEIGHTED",
            "name": "CatBoost Classifier (Unweighted, p>0.5)",
            "type": "clf",
            "model_fn": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, random_seed=42, verbose=0),
            "features": standard_features,
            "threshold": 0.50
        },
        {
            "id": "EXP4_CATBOOST_CLF_BALANCED",
            "name": "CatBoost Classifier (auto_class_weights='Balanced')",
            "type": "clf",
            "model_fn": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, auto_class_weights="Balanced", random_seed=42, verbose=0),
            "features": standard_features,
            "threshold": 0.50
        },
        {
            "id": "EXP5_CATBOOST_CLF_WEIGHTED_1_5",
            "name": "CatBoost Classifier (Class Weight DOWN:UP = 1.5:1.0)",
            "type": "clf",
            "model_fn": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, class_weights=[1.5, 1.0], random_seed=42, verbose=0),
            "features": standard_features,
            "threshold": 0.50
        },
        {
            "id": "EXP6_CATBOOST_CLF_WEIGHTED_2_0",
            "name": "CatBoost Classifier (Class Weight DOWN:UP = 2.0:1.0)",
            "type": "clf",
            "model_fn": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, class_weights=[2.0, 1.0], random_seed=42, verbose=0),
            "features": standard_features,
            "threshold": 0.50
        },
        {
            "id": "EXP7_CATBOOST_CLF_TUNED_PROB",
            "name": "CatBoost Classifier + Train-Tuned Probability Threshold",
            "type": "clf_tuned",
            "model_fn": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, random_seed=42, verbose=0),
            "features": standard_features,
        },
        {
            "id": "EXP8_XGBOOST_CLF_UNWEIGHTED",
            "name": "XGBoost Classifier (Unweighted, p>0.5)",
            "type": "clf",
            "model_fn": lambda: XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.02, random_state=42, n_jobs=1, verbosity=0),
            "features": standard_features,
            "threshold": 0.50
        },
        {
            "id": "EXP9_XGBOOST_CLF_BALANCED",
            "name": "XGBoost Classifier (scale_pos_weight=0.85 to favor DOWN)",
            "type": "clf",
            "model_fn": lambda: XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.02, scale_pos_weight=0.85, random_state=42, n_jobs=1, verbosity=0),
            "features": standard_features,
            "threshold": 0.50
        },
        {
            "id": "EXP10_RANDOM_FOREST_CLF_UNWEIGHTED",
            "name": "Random Forest Classifier (Unweighted, p>0.5)",
            "type": "clf",
            "model_fn": lambda: RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=1),
            "features": standard_features,
            "threshold": 0.50
        },
        {
            "id": "EXP11_RANDOM_FOREST_CLF_BALANCED",
            "name": "Random Forest Classifier (class_weight='balanced')",
            "type": "clf",
            "model_fn": lambda: RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=1),
            "features": standard_features,
            "threshold": 0.50
        },
        {
            "id": "EXP12_CATBOOST_CLF_EXPANDED_FEATURES",
            "name": "CatBoost Classifier + Downside Features (Unweighted)",
            "type": "clf",
            "model_fn": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, random_seed=42, verbose=0),
            "features": expanded_features,
            "threshold": 0.50
        },
        {
            "id": "EXP13_CATBOOST_CLF_EXPANDED_BALANCED",
            "name": "CatBoost Classifier + Downside Features (Balanced)",
            "type": "clf",
            "model_fn": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, auto_class_weights="Balanced", random_seed=42, verbose=0),
            "features": expanded_features,
            "threshold": 0.50
        },
        {
            "id": "EXP14_CATBOOST_CLF_EXPANDED_TUNED_PROB",
            "name": "CatBoost Classifier + Downside Features + Train-Tuned Prob Threshold",
            "type": "clf_tuned",
            "model_fn": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.02, l2_leaf_reg=3.0, random_seed=42, verbose=0),
            "features": expanded_features,
        }
    ]

    all_exp_results = []

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        print(f"\nRunning {exp_id}: {exp_name}...")
        
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        tuned_thresholds = []

        for f in folds:
            tr_s, tr_e = f["train_indices"]
            vl_s, vl_e = f["val_indices"]

            df_tr = dev_purged.iloc[tr_s:tr_e]
            df_vl = dev_purged.iloc[vl_s:vl_e]

            feat_cols = exp["features"]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(df_tr[feat_cols].values)
            X_vl = scaler.transform(df_vl[feat_cols].values)

            y_tr_ret = df_tr["Target_Next_Log_Return"].values
            y_vl_ret = df_vl["Target_Next_Log_Return"].values

            y_tr_dir = (y_tr_ret > 0).astype(int)
            y_vl_dir = (y_vl_ret > 0).astype(int)

            if exp["type"] == "baseline":
                # Always-UP
                y_pred_vl = np.ones(len(y_vl_dir), dtype=int)
                thresh_used = 0.5

            elif exp["type"] == "reg":
                model = exp["model_fn"]()
                model.fit(X_tr, y_tr_ret)
                pred_ret = model.predict(X_vl)
                thresh_used = exp["threshold"]
                y_pred_vl = (pred_ret > thresh_used).astype(int)

            elif exp["type"] == "reg_tuned":
                # Tune return threshold on inner 80/20 train split
                inner_split = int(len(X_tr) * 0.8)
                X_tr_in, X_val_in = X_tr[:inner_split], X_tr[inner_split:]
                y_tr_in, y_val_in = y_tr_ret[:inner_split], y_tr_ret[inner_split:]
                y_val_in_dir = (y_val_in > 0).astype(int)

                model = exp["model_fn"]()
                model.fit(X_tr_in, y_tr_in)
                pred_in = model.predict(X_val_in)

                # Search threshold in [-0.005, +0.005] that maximizes balanced accuracy with UP recall >= 0.50
                best_tau = 0.0
                best_score = -1.0
                for tau in np.linspace(-0.004, 0.004, 81):
                    p_dir = (pred_in > tau).astype(int)
                    rec_up = np.sum((p_dir == 1) & (y_val_in_dir == 1)) / (np.sum(y_val_in_dir == 1) + 1e-9)
                    if rec_up >= 0.50:  # protect UP performance
                        bal = balanced_accuracy_score(y_val_in_dir, p_dir)
                        if bal > best_score:
                            best_score = bal
                            best_tau = tau

                # Refit on full train
                model_full = exp["model_fn"]()
                model_full.fit(X_tr, y_tr_ret)
                pred_ret = model_full.predict(X_vl)
                thresh_used = best_tau
                tuned_thresholds.append(best_tau)
                y_pred_vl = (pred_ret > best_tau).astype(int)

            elif exp["type"] == "clf":
                model = exp["model_fn"]()
                model.fit(X_tr, y_tr_dir)
                probs = model.predict_proba(X_vl)[:, 1]
                thresh_used = exp["threshold"]
                y_pred_vl = (probs > thresh_used).astype(int)

            elif exp["type"] == "clf_tuned":
                # Tune prob threshold on inner 80/20 train split
                inner_split = int(len(X_tr) * 0.8)
                X_tr_in, X_val_in = X_tr[:inner_split], X_tr[inner_split:]
                y_tr_in, y_val_in_dir = y_tr_dir[:inner_split], y_tr_dir[inner_split:]

                model = exp["model_fn"]()
                model.fit(X_tr_in, y_tr_in)
                probs_in = model.predict_proba(X_val_in)[:, 1]

                best_p = 0.50
                best_score = -1.0
                for p_th in np.linspace(0.40, 0.60, 41):
                    p_dir = (probs_in > p_th).astype(int)
                    rec_up = np.sum((p_dir == 1) & (y_val_in_dir == 1)) / (np.sum(y_val_in_dir == 1) + 1e-9)
                    if rec_up >= 0.50:  # protect UP recall from collapsing
                        bal = balanced_accuracy_score(y_val_in_dir, p_dir)
                        if bal > best_score:
                            best_score = bal
                            best_p = p_th

                model_full = exp["model_fn"]()
                model_full.fit(X_tr, y_tr_dir)
                probs = model_full.predict_proba(X_vl)[:, 1]
                thresh_used = best_p
                tuned_thresholds.append(best_p)
                y_pred_vl = (probs > best_p).astype(int)

            m = evaluate_directional_predictions(y_vl_dir, y_pred_vl)
            m["fold"] = f["fold"]
            m["threshold"] = thresh_used
            fold_metrics.append(m)

            all_y_true.extend(y_vl_dir)
            all_y_pred.extend(y_pred_vl)

        # Pooled metrics across all 1,164 out-of-sample predictions
        pooled_m = evaluate_directional_predictions(np.array(all_y_true), np.array(all_y_pred))
        
        # Mean across folds
        mean_acc = np.mean([f["Accuracy"] for f in fold_metrics])
        mean_bal_acc = np.mean([f["Balanced_Accuracy"] for f in fold_metrics])
        mean_macro_f1 = np.mean([f["Macro_F1"] for f in fold_metrics])
        mean_up_f1 = np.mean([f["UP_F1"] for f in fold_metrics])
        mean_down_f1 = np.mean([f["DOWN_F1"] for f in fold_metrics])
        mean_down_rec = np.mean([f["DOWN_Recall"] for f in fold_metrics])
        mean_up_rec = np.mean([f["UP_Recall"] for f in fold_metrics])

        summary = {
            "id": exp_id,
            "name": exp_name,
            "pooled": pooled_m,
            "fold_metrics": fold_metrics,
            "mean_fold_acc": mean_acc,
            "mean_fold_bal_acc": mean_bal_acc,
            "mean_fold_macro_f1": mean_macro_f1,
            "mean_fold_up_f1": mean_up_f1,
            "mean_fold_down_f1": mean_down_f1,
            "mean_fold_up_rec": mean_up_rec,
            "mean_fold_down_rec": mean_down_rec,
            "tuned_thresholds": tuned_thresholds
        }
        all_exp_results.append(summary)

        print(f"  -> Pooled Acc: {pooled_m['Accuracy']:.2f}% | Bal Acc: {pooled_m['Balanced_Accuracy']:.2f}% | Macro F1: {pooled_m['Macro_F1']:.2f}%")
        print(f"     UP:   Prec={pooled_m['UP_Precision']:.2f}%, Rec={pooled_m['UP_Recall']:.2f}%, F1={pooled_m['UP_F1']:.2f}%")
        print(f"     DOWN: Prec={pooled_m['DOWN_Precision']:.2f}%, Rec={pooled_m['DOWN_Recall']:.2f}%, F1={pooled_m['DOWN_F1']:.2f}%")
        print(f"     CM: TN={pooled_m['Confusion_Matrix']['TN']}, FP={pooled_m['Confusion_Matrix']['FP']}, FN={pooled_m['Confusion_Matrix']['FN']}, TP={pooled_m['Confusion_Matrix']['TP']}")

    return all_exp_results


if __name__ == "__main__":
    results = run_experiments()
    import pickle
    with open("/Users/nithyashreevsuresh/Downloads/st/predictive_insights/experiments/directional_study_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nAll experiments successfully completed and saved to directional_study_results.pkl.")
