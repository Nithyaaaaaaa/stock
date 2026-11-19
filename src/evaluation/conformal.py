"""
Conformal Prediction Intervals:
Provides distribution-free prediction intervals calibrated on validation data.
Explicitly distinguishes nominal coverage (e.g., 95%) from empirical holdout coverage.
"""

from typing import Dict, Any, Tuple
import numpy as np


class ConformalPredictor:
    """
    Split Conformal Calibrator using return-scaled nonconformity scores:
    s_i = |y_i - y_hat_i| / sigma_i
    where sigma_i is historical return volatility (or default 1.0).
    """
    def __init__(self, nominal_coverage: float = 0.95):
        self.nominal_coverage = nominal_coverage
        self.alpha = 1.0 - nominal_coverage
        self.q_val = None
        self.is_calibrated = False
        
    def calibrate(self, actual_returns: np.ndarray, pred_returns: np.ndarray, volatilities: np.ndarray = None) -> "ConformalPredictor":
        """
        Calibrate quantile on a distinct calibration dataset (never on test data).
        """
        actual = np.asarray(actual_returns, dtype=np.float64)
        pred = np.asarray(pred_returns, dtype=np.float64)
        
        if volatilities is None or np.all(volatilities <= 0):
            scale = np.ones_like(actual)
        else:
            scale = np.asarray(volatilities, dtype=np.float64)
            scale = np.where(scale > 1e-4, scale, np.median(scale[scale > 1e-4]))
            
        scores = np.abs(actual - pred) / scale
        n = len(scores)
        
        # Conformal quantile level with finite-sample adjustment (n+1)(1-alpha)/n
        p_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_val = float(np.quantile(scores, p_level))
        self.is_calibrated = True
        return self
        
    def predict_intervals(
        self,
        pred_returns: np.ndarray,
        prev_closes: np.ndarray,
        volatilities: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (lower_returns, upper_returns, lower_prices, upper_prices).
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor must be calibrated before generating intervals.")
            
        pred = np.asarray(pred_returns, dtype=np.float64)
        closes = np.asarray(prev_closes, dtype=np.float64)
        
        if volatilities is None or np.all(volatilities <= 0):
            scale = np.ones_like(pred)
        else:
            scale = np.asarray(volatilities, dtype=np.float64)
            scale = np.where(scale > 1e-4, scale, np.median(scale[scale > 1e-4]))
            
        margin = self.q_val * scale
        lower_ret = pred - margin
        upper_ret = pred + margin
        
        lower_price = closes * np.exp(lower_ret)
        upper_price = closes * np.exp(upper_ret)
        
        return lower_ret, upper_ret, lower_price, upper_price
        
    def evaluate_coverage(self, actual_prices: np.ndarray, lower_prices: np.ndarray, upper_prices: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates empirical coverage and interval widths on test observations.
        """
        actual = np.asarray(actual_prices, dtype=np.float64)
        lower = np.asarray(lower_prices, dtype=np.float64)
        upper = np.asarray(upper_prices, dtype=np.float64)
        
        covered = (actual >= lower) & (actual <= upper)
        empirical_cov = float(np.mean(covered) * 100.0)
        avg_width = float(np.mean(upper - lower))
        median_width = float(np.median(upper - lower))
        
        return {
            "Nominal_Coverage": float(self.nominal_coverage * 100.0),
            "Empirical_Coverage": empirical_cov,
            "Average_Interval_Width": avg_width,
            "Median_Interval_Width": median_width,
            "Total_Sessions": len(actual),
            "Covered_Sessions": int(np.sum(covered)),
            "Coverage_Gap": float(empirical_cov - (self.nominal_coverage * 100.0))
        }
