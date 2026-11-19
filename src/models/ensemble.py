"""
Ensemble Forecasters:
BlendingForecaster and StackingForecaster combining predictions from diverse model families.
"""

from typing import List, Optional, Dict
import numpy as np
from sklearn.linear_model import Ridge

from src.models.base import BaseForecaster


class BlendingForecaster(BaseForecaster):
    """
    Weighted Average Blending Ensemble of multiple forecasters.
    y_hat = sum(w_i * y_hat_i)
    """
    def __init__(self, models: List[BaseForecaster], weights: Optional[List[float]] = None, name: str = "Blend Ensemble"):
        super().__init__(name=name)
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]
            
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BlendingForecaster":
        for model in self.models:
            model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        preds = np.zeros(len(X), dtype=np.float32)
        for model, w in zip(self.models, self.weights):
            preds += w * model.predict_log_returns(X)
        return preds


class StackingForecaster(BaseForecaster):
    """
    Stacking Ensemble with Ridge meta-learner on out-of-fold (OOF) base model predictions.
    Uses 3-fold temporal cross-validation to generate unbiased meta-features.
    """
    def __init__(self, base_models: List[BaseForecaster], alpha: float = 1.0, name: str = "Stacking (Ridge Meta-Learner)"):
        super().__init__(name=name)
        self.base_models = base_models
        self.meta_learner = Ridge(alpha=alpha, fit_intercept=True)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "StackingForecaster":
        from sklearn.model_selection import KFold
        
        N = len(X)
        kf = KFold(n_splits=3, shuffle=False)
        oof_meta_features = np.zeros((N, len(self.base_models)), dtype=np.float32)
        
        # 1. Generate out-of-fold predictions for meta-learner training
        for m_idx, model in enumerate(self.base_models):
            for tr_idx, val_idx in kf.split(X):
                model.fit(X[tr_idx], y[tr_idx], **kwargs)
                oof_meta_features[val_idx, m_idx] = model.predict_log_returns(X[val_idx])
                
        # 2. Fit Ridge meta-learner on unbiased OOF meta-features
        self.meta_learner.fit(oof_meta_features, y)
        
        # 3. Re-fit base models on full training data for future test inference
        for model in self.base_models:
            model.fit(X, y, **kwargs)
            
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        meta_features = []
        for model in self.base_models:
            m_pred = model.predict_log_returns(X)
            meta_features.append(m_pred.reshape(-1, 1))
        M = np.hstack(meta_features)
        return self.meta_learner.predict(M).astype(np.float32)
