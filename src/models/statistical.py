"""
Statistical Time Series Forecasters:
ARIMA and Exponential Smoothing for return and price forecasting.
"""

from typing import Optional
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing

from src.models.base import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA(1,0,0) / AR(1) Forecaster on stationary log returns:
    y_hat_{t+1} = c + phi * y_t
    """
    def __init__(self, order: tuple = (1, 0, 0)):
        super().__init__(name=f"ARIMA{order}")
        self.order = order
        self.const_ = 0.0
        self.phi_ = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "ARIMAForecaster":
        try:
            model = ARIMA(y, order=self.order)
            res = model.fit()
            params = res.params
            self.const_ = float(params[0]) if len(params) > 0 else 0.0
            self.phi_ = float(params[1]) if len(params) > 1 else 0.0
        except Exception:
            self.const_ = float(np.mean(y))
            self.phi_ = 0.0
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        # If feature 0 is Log_Return_Lag_1, y_hat = const + phi * lag1
        # Else forecast unconditional mean
        if X.ndim == 2 and X.shape[1] > 0:
            last_returns = X[:, 0]
            preds = self.const_ + self.phi_ * last_returns
        else:
            preds = np.full(len(X), fill_value=self.const_, dtype=np.float32)
        return preds.astype(np.float32)


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Simple Exponential Smoothing (SES) on log returns:
    y_hat_{t+1} = alpha * y_t + (1 - alpha) * y_hat_t
    """
    def __init__(self, smoothing_level: Optional[float] = 0.1):
        super().__init__(name="Exponential Smoothing")
        self.smoothing_level = smoothing_level
        self.fitted_level_ = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "ExponentialSmoothingForecaster":
        try:
            ses = SimpleExpSmoothing(y, initialization_method="estimated")
            fit_res = ses.fit(smoothing_level=self.smoothing_level, optimized=(self.smoothing_level is None))
            self.fitted_level_ = float(fit_res.forecast(1)[0])
        except Exception:
            self.fitted_level_ = float(np.mean(y))
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), fill_value=self.fitted_level_, dtype=np.float32)
