"""
Unit Tests: Model Forecaster Interface & Prediction Invariance.
"""

import pytest
import numpy as np

from src.models.baselines import PersistenceForecaster, DriftForecaster, MajorityClassForecaster
from src.models.trees import RandomForestForecaster, XGBoostForecaster, LightGBMForecaster, CatBoostForecaster
from src.models.deep_learning import LSTMForecaster, GRUForecaster


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    N, D = 100, 10
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randn(N).astype(np.float32) * 0.015
    prev_closes = np.linspace(150, 180, N).astype(np.float32)
    return X, y, prev_closes


def test_persistence_price_reconstruction(synthetic_data):
    X, y, prev_closes = synthetic_data
    forecaster = PersistenceForecaster()
    forecaster.fit(X, y)
    
    log_ret = forecaster.predict_log_returns(X)
    assert np.allclose(log_ret, 0.0)
    
    pred_p = forecaster.predict_prices(X, prev_closes)
    assert np.allclose(pred_p, prev_closes), "Persistence must predict exact previous close"


def test_tree_forecasters_produce_valid_predictions(synthetic_data):
    X, y, prev_closes = synthetic_data
    models = [
        RandomForestForecaster(n_estimators=10, max_depth=3, random_state=42),
        XGBoostForecaster(n_estimators=10, max_depth=2, random_state=42),
        LightGBMForecaster(n_estimators=10, max_depth=2, random_state=42),
        CatBoostForecaster(iterations=10, depth=2, random_seed=42)
    ]
    
    for m in models:
        m.fit(X[:80], y[:80])
        assert m.is_fitted
        
        preds = m.predict_log_returns(X[80:])
        assert len(preds) == 20
        assert not np.isnan(preds).any()
        assert not np.isinf(preds).any()
        
        prices = m.predict_prices(X[80:], prev_closes[80:])
        assert len(prices) == 20
        assert np.all(prices > 0)


def test_deep_learning_models_handle_sequence_padding(synthetic_data):
    pytest.importorskip("torch")
    X, y, prev_closes = synthetic_data
    lstm = LSTMForecaster(seq_len=10, hidden_dim=16, num_layers=1, epochs=2, lr=0.01, seed=42)
    lstm.fit(X[:80], y[:80])
    assert lstm.is_fitted
    
    preds = lstm.predict_log_returns(X[80:])
    assert len(preds) == 20
    assert not np.isnan(preds).any()
