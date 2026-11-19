"""
Base Forecaster Interface: Standardizes fit, predict, and price reconstruction.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseForecaster":
        """Fit model on feature matrix X and target y (log returns)."""
        pass
        
    @abstractmethod
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous log returns (y_hat)."""
        pass
        
    def predict_prices(self, X: np.ndarray, prev_closes: np.ndarray) -> np.ndarray:
        """
        Reconstruct predicted dollar closing prices using exact continuous compound formula:
        P_hat_{t+1} = P_t * exp(y_hat_{t+1})
        """
        log_returns = self.predict_log_returns(X)
        return prev_closes * np.exp(log_returns)
        
    def predict_direction(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary market direction:
        1 (UP) if predicted log return > threshold else 0 (DOWN).
        Uses self.learned_threshold if available and threshold is None.
        """
        thresh = getattr(self, "learned_threshold", 0.0) if threshold is None else threshold
        log_returns = self.predict_log_returns(X)
        return (log_returns > thresh).astype(int)
