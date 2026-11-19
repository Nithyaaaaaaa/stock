"""
Tree-Based Ensemble Forecasters:
Random Forest, XGBoost, LightGBM, and CatBoost.
All models predict continuous log returns.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
from catboost import CatBoostRegressor, CatBoostClassifier

from src.models.base import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    """Random Forest regressor on stationary tabular features."""
    def __init__(self, n_estimators: int = 150, max_depth: int = 5, min_samples_leaf: int = 5, random_state: int = 42):
        super().__init__(name="Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=1
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RandomForestForecaster":
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32)


class XGBoostForecaster(BaseForecaster):
    """Gradient boosted decision trees via XGBoost."""
    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 3,
        learning_rate: float = 0.02,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        super().__init__(name="XGBoost")
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for XGBoostForecaster. Install it via pip install xgboost.")
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=1,
            verbosity=0
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "XGBoostForecaster":
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32)


class LightGBMForecaster(BaseForecaster):
    """Highly efficient gradient boosted trees via LightGBM."""
    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 3,
        num_leaves: int = 15,
        learning_rate: float = 0.02,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        super().__init__(name="LightGBM")
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required for LightGBMForecaster. Install it via pip install lightgbm.")
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=1,
            verbose=-1
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LightGBMForecaster":
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32)


class CatBoostForecaster(BaseForecaster):
    """
    Symmetric tree gradient boosting via CatBoost with robust regularization.
    Supports EXP2 (CatBoost Regressor + Train-Tuned Return Threshold):
    - Preserves continuous log return regression for next-day price forecasting.
    - Learns an optimal return threshold tau* on historical training data to optimize
      balanced accuracy for UP/DOWN directional classification without sacrificing UP recall.
    """
    def __init__(
        self,
        iterations: int = 120,
        depth: int = 4,
        learning_rate: float = 0.02,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        tune_threshold: bool = True,
        threshold: float = 0.0,
        inner_val_ratio: float = 0.20,
        up_recall_floor: float = 0.50
    ):
        name = "CatBoost (EXP2 Tuned)" if tune_threshold else "CatBoost (Baseline)"
        super().__init__(name=name)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.tune_threshold = tune_threshold
        self.learned_threshold = float(threshold)
        self.inner_val_ratio = inner_val_ratio
        self.up_recall_floor = up_recall_floor
        
        self.model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            verbose=0
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "CatBoostForecaster":
        """
        Fit CatBoost regressor on feature matrix X and target log returns y.
        When tune_threshold is True, uses an inner chronological train/val split
        of the training fold to learn the return threshold tau* that maximizes
        balanced directional accuracy while preserving an UP recall floor (>= up_recall_floor).
        """
        N = len(X)
        if self.tune_threshold and N >= 100:
            inner_split = int(N * (1.0 - self.inner_val_ratio))
            X_tr_in, X_val_in = X[:inner_split], X[inner_split:]
            y_tr_in, y_val_in = y[:inner_split], y[inner_split:]
            y_val_dir = (y_val_in > 0).astype(int)
            
            # Fit inner model to predict on inner validation slice
            inner_model = CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                random_seed=self.random_seed,
                verbose=0
            )
            inner_model.fit(X_tr_in, y_tr_in)
            pred_val_ret = inner_model.predict(X_val_in)
            
            # Grid search candidate return thresholds in [-0.004, +0.004] (81 points)
            best_tau = 0.0
            best_score = -1.0
            candidate_taus = np.linspace(-0.004, 0.004, 81)
            for tau in candidate_taus:
                p_dir = (pred_val_ret > tau).astype(int)
                rec_up = np.sum((p_dir == 1) & (y_val_dir == 1)) / (np.sum(y_val_dir == 1) + 1e-9)
                if rec_up >= self.up_recall_floor:
                    bal = balanced_accuracy_score(y_val_dir, p_dir)
                    if bal > best_score:
                        best_score = bal
                        best_tau = float(tau)
            self.learned_threshold = best_tau
            
        elif not self.tune_threshold:
            # Baseline configuration: zero return threshold
            self.learned_threshold = 0.0

        # Fit final model on full training set (X, y)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32)

    def predict_direction(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict market direction using learned threshold tau* (or specified threshold).
        1 (UP) if predicted log return > threshold else 0 (DOWN).
        """
        thresh = self.learned_threshold if threshold is None else float(threshold)
        log_ret = self.predict_log_returns(X)
        return (log_ret > thresh).astype(int)


class CatBoostEXP2Forecaster(CatBoostForecaster):
    """Explicit alias for EXP2 Champion Configuration (CatBoost + Train-Tuned Threshold)."""
    def __init__(self, **kwargs):
        kwargs["tune_threshold"] = True
        super().__init__(**kwargs)


class CatBoostBaselineForecaster(CatBoostForecaster):
    """Documented baseline comparison: Unweighted CatBoost with fixed zero threshold (tau=0.0)."""
    def __init__(self, **kwargs):
        kwargs["tune_threshold"] = False
        kwargs["threshold"] = 0.0
        super().__init__(**kwargs)


class CatBoostDownsideSpecialist(BaseForecaster):
    """
    Experimental Downside-Specialist Model (EXP13):
    Direct CatBoost Classifier with auto_class_weights='Balanced' on expanded downside features.
    Documented as an experimental downside hedge/risk model. Must NOT be the default.
    """
    def __init__(
        self,
        iterations: int = 120,
        depth: int = 4,
        learning_rate: float = 0.02,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42
    ):
        super().__init__(name="CatBoost Downside Specialist (EXP13, Experimental)")
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            auto_class_weights="Balanced",
            random_seed=random_seed,
            verbose=0
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "CatBoostDownsideSpecialist":
        # Target y should be binary directional label (1 if return > 0 else 0)
        y_dir = (y > 0).astype(int) if y.dtype != int else y
        self.model.fit(X, y_dir)
        self.is_fitted = True
        return self
        
    def predict_log_returns(self, X: np.ndarray) -> np.ndarray:
        # Not a natural continuous price regressor; returns pseudo log returns from probabilities
        probs = self.model.predict_proba(X)[:, 1]
        return (probs - 0.5).astype(np.float32) * 0.02
        
    def predict_direction(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        probs = self.model.predict_proba(X)[:, 1]
        p_thresh = 0.50 if threshold is None else float(threshold)
        return (probs > p_thresh).astype(int)
