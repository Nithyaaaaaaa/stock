"""
Unit Tests: EXP2 CatBoost Regressor with Train-Tuned Return Threshold Integration.
Verifies threshold learning, continuous price preservation, independent fold tuning,
and comparison baseline availability.
"""

import pytest
import numpy as np

from src.models.trees import (
    CatBoostForecaster,
    CatBoostEXP2Forecaster,
    CatBoostBaselineForecaster,
    CatBoostDownsideSpecialist
)
from src.evaluation.walk_forward import WalkForwardValidator
import pandas as pd


@pytest.fixture
def synthetic_timeseries_data():
    np.random.seed(42)
    N, D = 250, 8
    X = np.random.randn(N, D).astype(np.float32)
    # Slight positive drift with some noise
    y = (np.random.randn(N) * 0.015 + 0.001).astype(np.float32)
    prev_closes = np.linspace(150, 180, N).astype(np.float32)
    return X, y, prev_closes


def test_catboost_exp2_threshold_tuning(synthetic_timeseries_data):
    X, y, prev_closes = synthetic_timeseries_data
    model = CatBoostEXP2Forecaster(iterations=20, depth=3, random_seed=42)
    
    # Fit model on training slice
    model.fit(X[:180], y[:180])
    
    assert model.is_fitted
    assert hasattr(model, "learned_threshold")
    # Learned threshold should be a finite float learned from inner split
    assert np.isfinite(model.learned_threshold)
    
    # 1. Continuous return predictions must be preserved
    pred_returns = model.predict_log_returns(X[180:])
    assert len(pred_returns) == 70
    assert not np.isnan(pred_returns).any()
    
    # 2. Next-day price reconstruction must remain functional
    pred_prices = model.predict_prices(X[180:], prev_closes[180:])
    assert len(pred_prices) == 70
    assert np.all(pred_prices > 0)
    assert np.allclose(pred_prices, prev_closes[180:] * np.exp(pred_returns))
    
    # 3. Directional classification uses learned threshold by default
    pred_dir = model.predict_direction(X[180:])
    manual_dir = (pred_returns > model.learned_threshold).astype(int)
    assert np.array_equal(pred_dir, manual_dir)


def test_catboost_baseline_zero_threshold(synthetic_timeseries_data):
    X, y, prev_closes = synthetic_timeseries_data
    baseline = CatBoostBaselineForecaster(iterations=20, depth=3, random_seed=42)
    baseline.fit(X[:180], y[:180])
    
    assert baseline.learned_threshold == 0.0
    pred_dir = baseline.predict_direction(X[180:])
    pred_returns = baseline.predict_log_returns(X[180:])
    assert np.array_equal(pred_dir, (pred_returns > 0.0).astype(int))


def test_walk_forward_evaluates_exp2_independently():
    np.random.seed(42)
    N = 300
    dates = pd.date_range("2020-01-01", periods=N, freq="B")
    feature_cols = [f"F_{i}" for i in range(5)]
    data = {f: np.random.randn(N) for f in feature_cols}
    data["Close"] = np.linspace(100, 150, N) + np.random.randn(N)
    data["Target_Next_Close"] = np.roll(data["Close"], -1)
    data["Target_Next_Log_Return"] = np.random.randn(N) * 0.01 + 0.0005
    data["Vol_20d"] = np.full(N, 0.015)
    
    df = pd.DataFrame(data, index=dates).iloc[:-1]
    
    wf = WalkForwardValidator(n_splits=3, min_train_ratio=0.50, purge_boundary=True)
    res = wf.evaluate_model(
        lambda: CatBoostEXP2Forecaster(iterations=15, depth=2, random_seed=42),
        df=df,
        feature_cols=feature_cols,
        target_col="Target_Next_Log_Return",
        price_col="Close",
        actual_price_col="Target_Next_Close"
    )
    
    assert "folds" in res
    assert len(res["folds"]) == 3
    # Check that each fold has its own independent learned threshold
    for f in res["folds"]:
        assert "learned_threshold" in f
        assert np.isfinite(f["learned_threshold"])


def test_exp13_downside_specialist_non_default(synthetic_timeseries_data):
    X, y, prev_closes = synthetic_timeseries_data
    specialist = CatBoostDownsideSpecialist(iterations=15, depth=2, random_seed=42)
    specialist.fit(X[:180], y[:180])
    
    assert specialist.is_fitted
    pred_dir = specialist.predict_direction(X[180:])
    assert len(pred_dir) == 70
    assert set(np.unique(pred_dir)).issubset({0, 1})
