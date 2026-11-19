"""
Evaluation Metrics:
Unified evaluation module for regression and directional classification metrics.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score
)


def compute_regression_metrics(actual_prices: np.ndarray, pred_prices: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics on reconstructed closing prices.
    """
    actual = np.asarray(actual_prices, dtype=np.float64)
    pred = np.asarray(pred_prices, dtype=np.float64)
    
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    medae = float(median_absolute_error(actual, pred))
    
    # Avoid division by zero
    nonzero_mask = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero_mask] - pred[nonzero_mask]) / actual[nonzero_mask])) * 100.0)
    
    # R2 score
    r2 = float(r2_score(actual, pred))
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "MAPE": mape,
        "R2": r2
    }


def compute_directional_metrics(
    actual_returns: np.ndarray,
    pred_returns: np.ndarray,
    threshold: float = 0.0,
    positive_label_name: str = "UP",
    pred_directions: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive directional classification metrics.
    Actual UP: actual_return > 0
    Predicted UP: pred_directions == 1, or pred_return > threshold if pred_directions is None.
    
    Returns accuracy, balanced accuracy, precision, recall, F1 (for UP and DOWN),
    macro/weighted F1, and 2x2 confusion matrix [[TN, FP], [FN, TP]].
    """
    actual_ret = np.asarray(actual_returns, dtype=np.float64)
    pred_ret = np.asarray(pred_returns, dtype=np.float64)
    
    y_true = (actual_ret > 0).astype(int)
    if pred_directions is not None:
        y_pred = np.asarray(pred_directions, dtype=int)
    else:
        y_pred = (pred_ret > threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    # Classification metrics for UP (label=1) and DOWN (label=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0)
    macro_prec = float(np.mean(prec) * 100.0)
    macro_rec = float(np.mean(rec) * 100.0)
    
    return {
        "Directional_Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Macro_Precision": macro_prec,
        "Macro_Recall": macro_rec,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "UP_Precision": float(prec[1] * 100.0),
        "UP_Recall": float(rec[1] * 100.0),
        "UP_F1": float(f1[1] * 100.0),
        "DOWN_Precision": float(prec[0] * 100.0),
        "DOWN_Recall": float(rec[0] * 100.0),
        "DOWN_F1": float(f1[0] * 100.0),
        "Confusion_Matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Matrix": cm.tolist()
        },
        "Support_DOWN": int(support[0]),
        "Support_UP": int(support[1]),
        "Threshold": threshold
    }
