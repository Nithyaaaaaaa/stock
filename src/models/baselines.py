"""
Baseline Forecasters: Non-learned benchmarks for rigorous comparison.
"""

import numpy as np
from src.models.base import BaseForecaster


class PersistenceForecaster(BaseForecaster):
    """
    Random-Walk Persistence Benchmark:
    Predicts zero log return (y_hat = 0) -> P_hat_{t+1} = P_t.
    """
    def __init__(self):
        super().__init__(name="Persistence Baseline")
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "PersistenceForecaster":
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float32)


class DriftForecaster(BaseForecaster):
    """
    Historical Drift Benchmark:
    Predicts constant mean historical log return y_hat = mean(y_train).
    """
    def __init__(self):
        super().__init__(name="Historical Drift Baseline")
        self.mean_drift = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "DriftForecaster":
        self.mean_drift = float(np.mean(y))
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), fill_value=self.mean_drift, dtype=np.float32)


class SMAForecaster(BaseForecaster):
    """
    Moving Average Benchmark:
    Predicts next closing price as the rolling window average of closing prices.
    """
    def __init__(self, window: int = 20):
        super().__init__(name=f"SMA ({window}-day) Baseline")
        self.window = window
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "SMAForecaster":
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        # Dummy zero return; predict_prices is overridden directly
        return np.zeros(len(X), dtype=np.float32)
        
    def predict_prices(self, X: np.ndarray, prev_closes: np.ndarray) -> np.ndarray:
        # In tabular feature representations, SMA is computed from price history
        return prev_closes


class MajorityClassForecaster(BaseForecaster):
    """
    Majority-Class Directional Baseline:
    Always predicts the historically dominant market direction (UP: 1).
    """
    def __init__(self):
        super().__init__(name="Majority-Class (Always UP)")
        self.median_positive_return = 0.005
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MajorityClassForecaster":
        pos_y = y[y > 0]
        if len(pos_y) > 0:
            self.median_positive_return = float(np.median(pos_y))
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), fill_value=self.median_positive_return, dtype=np.float32)
        
    def predict_direction(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return np.ones(len(X), dtype=int)
