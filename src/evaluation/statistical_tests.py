"""
Statistical Significance Tests:
Diebold-Mariano test with Newey-West HAC variance estimator and Wilcoxon signed-rank test.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    loss_type: str = "squared",
    max_lag: int = 1
) -> Dict[str, Any]:
    """
    Diebold-Mariano test for predictive accuracy equivalence.
    H0: E[loss(e1) - loss(e2)] = 0 (Both models have identical predictive accuracy)
    H1: E[loss(e1) - loss(e2)] != 0
    
    Uses Newey-West Bartlett kernel for Heteroscedasticity and Autocorrelation Consistent (HAC) standard error.
    """
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    T = len(e1)
    
    if loss_type == "squared":
        d = (e1 ** 2) - (e2 ** 2)
    elif loss_type == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    d_mean = np.mean(d)
    
    # Auto-covariances of d
    d_centered = d - d_mean
    gamma = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            gamma[0] = np.mean(d_centered ** 2)
        else:
            gamma[k] = np.mean(d_centered[k:] * d_centered[:-k])
            
    # Newey-West variance with Bartlett kernel weights: w_k = 1 - k / (max_lag + 1)
    var_d = gamma[0]
    for k in range(1, max_lag + 1):
        weight = 1.0 - (k / (max_lag + 1))
        var_d += 2.0 * weight * gamma[k]
        
    var_d_mean = var_d / T
    if var_d_mean <= 0:
        dm_stat = 0.0
        p_val = 1.0
    else:
        dm_stat = float(d_mean / np.sqrt(var_d_mean))
        p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat))))
        
    return {
        "DM_Statistic": dm_stat,
        "p_value": p_val,
        "mean_loss_differential": float(d_mean),
        "loss_type": loss_type,
        "hac_variance": float(var_d),
        "is_significant_5pct": bool(p_val < 0.05),
        "interpretation": (
            "Statistically significant difference (p < 0.05)"
            if p_val < 0.05
            else "No statistically significant difference (fail to reject H0)"
        )
    }


def paired_wilcoxon_test(e1: np.ndarray, e2: np.ndarray) -> Dict[str, Any]:
    """
    Non-parametric Wilcoxon signed-rank test on absolute errors.
    """
    abs_e1 = np.abs(e1)
    abs_e2 = np.abs(e2)
    diff = abs_e1 - abs_e2
    
    # Exclude exact zeros
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) < 10:
        return {"statistic": 0.0, "p_value": 1.0, "interpretation": "Insufficient distinct pairs"}
        
    res = stats.wilcoxon(diff_nonzero, alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "is_significant_5pct": bool(res.pvalue < 0.05),
        "median_abs_diff": float(np.median(diff))
    }
