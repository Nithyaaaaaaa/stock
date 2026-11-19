"""
Walk-Forward Cross-Validation:
5-Fold Expanding Window Cross-Validation across development data.
Evaluates regression, directional metrics, regime robustness, and latency.
"""

import time
from typing import Dict, List, Any, Type
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseForecaster
from src.evaluation.metrics import compute_regression_metrics, compute_directional_metrics


class WalkForwardValidator:
    """
    Expanding-Window Walk-Forward Cross-Validation with boundary embargo / purging.
    """
    def __init__(self, n_splits: int = 5, min_train_ratio: float = 0.50, purge_boundary: bool = True):
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.purge_boundary = purge_boundary
        
    def generate_folds(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates expanding-window train/val index splits and metadata.
        When purge_boundary is True, drops the final training session (tr_end - 1)
        whose forward target realized on the first validation session (vl_start).
        """
        N = len(df)
        min_train_size = int(N * self.min_train_ratio)
        test_size_total = N - min_train_size
        fold_test_size = test_size_total // self.n_splits
        
        folds = []
        for i in range(self.n_splits):
            train_end = min_train_size + i * fold_test_size
            test_start = train_end
            test_end = test_start + fold_test_size if i < self.n_splits - 1 else N
            
            # Apply 1-observation embargo/purge if requested
            actual_tr_end = train_end - 1 if self.purge_boundary else train_end
            
            folds.append({
                "fold": i + 1,
                "train_indices": (0, actual_tr_end),
                "val_indices": (test_start, test_end),
                "train_start_date": df.index[0].strftime("%Y-%m-%d"),
                "train_end_date": df.index[actual_tr_end - 1].strftime("%Y-%m-%d"),
                "val_start_date": df.index[test_start].strftime("%Y-%m-%d"),
                "val_end_date": df.index[test_end - 1].strftime("%Y-%m-%d"),
                "train_count": actual_tr_end,
                "val_count": test_end - test_start
            })
        return folds
        
    def evaluate_model(
        self,
        model_factory,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "Target_Next_Log_Return",
        price_col: str = "Close",
        actual_price_col: str = "Target_Next_Close"
    ) -> Dict[str, Any]:
        """
        Runs walk-forward evaluation across all folds for a given model factory.
        """
        folds = self.generate_folds(df)
        fold_results = []
        total_fit_time = 0.0
        total_infer_time = 0.0
        total_val_samples = 0
        
        all_val_actual_prices = []
        all_val_pred_prices = []
        all_val_actual_returns = []
        all_val_pred_returns = []
        all_val_pred_directions = []
        
        for f in folds:
            tr_start, tr_end = f["train_indices"]
            vl_start, vl_end = f["val_indices"]
            
            df_train = df.iloc[tr_start:tr_end]
            df_val = df.iloc[vl_start:vl_end]
            
            X_tr_raw = df_train[feature_cols].values
            y_tr = df_train[target_col].values
            
            X_vl_raw = df_val[feature_cols].values
            y_vl = df_val[target_col].values
            
            val_prev_closes = df_val[price_col].values
            val_actual_closes = df_val[actual_price_col].values
            
            # Feature scaling strictly within fold (no leakage)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_vl = scaler.transform(X_vl_raw)
            
            # Instantiate fresh model
            model: BaseForecaster = model_factory()
            
            # Timing fit
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_time = time.perf_counter() - t0
            total_fit_time += fit_time
            
            # Timing inference
            t1 = time.perf_counter()
            pred_returns = model.predict_log_returns(X_vl)
            infer_time = time.perf_counter() - t1
            total_infer_time += infer_time
            total_val_samples += len(X_vl)
            
            # Reconstruct prices
            pred_prices = model.predict_prices(X_vl, val_prev_closes)
            
            # Metrics
            fold_thresh = getattr(model, "learned_threshold", 0.0)
            pred_directions = model.predict_direction(X_vl)
            reg_metrics = compute_regression_metrics(val_actual_closes, pred_prices)
            dir_metrics = compute_directional_metrics(
                y_vl, pred_returns, threshold=fold_thresh, pred_directions=pred_directions
            )
            dir_metrics["Learned_Threshold"] = fold_thresh
            
            # Market regime: Volatility of fold
            vol_col = "Vol_20d" if "Vol_20d" in df_val.columns else None
            mean_vol = float(df_val[vol_col].mean()) if vol_col else 0.0
            
            fold_results.append({
                "fold": f["fold"],
                "train_dates": f"{f['train_start_date']} to {f['train_end_date']}",
                "val_dates": f"{f['val_start_date']} to {f['val_end_date']}",
                "val_count": f["val_count"],
                "learned_threshold": fold_thresh,
                "regime_vol_20d": mean_vol,
                "fit_time_sec": fit_time,
                "infer_latency_ms_per_sample": (infer_time / len(X_vl)) * 1000.0,
                "regression": reg_metrics,
                "directional": dir_metrics
            })
            
            all_val_actual_prices.extend(val_actual_closes)
            all_val_pred_prices.extend(pred_prices)
            all_val_actual_returns.extend(y_vl)
            all_val_pred_returns.extend(pred_returns)
            all_val_pred_directions.extend(pred_directions)
            
        all_actual_p = np.array(all_val_actual_prices)
        all_pred_p = np.array(all_val_pred_prices)
        all_actual_r = np.array(all_val_actual_returns)
        all_pred_r = np.array(all_val_pred_returns)
        all_pred_d = np.array(all_val_pred_directions)
        
        overall_reg = compute_regression_metrics(all_actual_p, all_pred_p)
        overall_dir = compute_directional_metrics(all_actual_r, all_pred_r, pred_directions=all_pred_d)
        
        avg_rmse = float(np.mean([fr["regression"]["RMSE"] for fr in fold_results]))
        avg_mae = float(np.mean([fr["regression"]["MAE"] for fr in fold_results]))
        avg_dir_acc = float(np.mean([fr["directional"]["Directional_Accuracy"] for fr in fold_results]))
        avg_bal_acc = float(np.mean([fr["directional"]["Balanced_Accuracy"] for fr in fold_results]))
        avg_macro_f1 = float(np.mean([fr["directional"]["Macro_F1"] for fr in fold_results]))
        avg_up_f1 = float(np.mean([fr["directional"]["UP_F1"] for fr in fold_results]))
        avg_down_f1 = float(np.mean([fr["directional"]["DOWN_F1"] for fr in fold_results]))
        
        return {
            "model_name": model.name,
            "folds": fold_results,
            "aggregated": {
                "overall_regression": overall_reg,
                "overall_directional": overall_dir,
                "mean_fold_rmse": avg_rmse,
                "std_fold_rmse": float(np.std([fr["regression"]["RMSE"] for fr in fold_results])),
                "mean_fold_mae": avg_mae,
                "mean_fold_dir_acc": avg_dir_acc,
                "mean_fold_bal_acc": avg_bal_acc,
                "mean_fold_macro_f1": avg_macro_f1,
                "std_fold_dir_acc": float(np.std([fr["directional"]["Directional_Accuracy"] for fr in fold_results])),
                "mean_fold_up_f1": avg_up_f1,
                "mean_fold_down_f1": avg_down_f1,
                "total_fit_time_sec": total_fit_time,
                "avg_inference_latency_ms": (total_infer_time / total_val_samples) * 1000.0
            },
            "oos_predictions": {
                "actual_prices": all_actual_p,
                "pred_prices": all_pred_p,
                "actual_returns": all_actual_r,
                "pred_returns": all_pred_r
            }
        }
