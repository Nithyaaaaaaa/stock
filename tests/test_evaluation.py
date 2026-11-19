"""
Unit Tests: Evaluation Metrics, Conformal Calibration, and Statistical Tests.
"""

import pytest
import numpy as np

from src.evaluation.metrics import compute_regression_metrics, compute_directional_metrics
from src.evaluation.statistical_tests import diebold_mariano_test
from src.evaluation.conformal import ConformalPredictor


def test_regression_metrics_exact_values():
    actual = np.array([100.0, 102.0, 101.0, 105.0])
    pred = np.array([100.0, 101.0, 102.0, 105.0])
    
    metrics = compute_regression_metrics(actual, pred)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(2.0 / 4.0), rel=1e-3)
    assert metrics["MAE"] == pytest.approx(0.5, rel=1e-3)
    assert metrics["MAPE"] > 0


def test_directional_metrics_confusion_matrix():
    actual_returns = np.array([0.01, -0.02, 0.03, -0.01])  # UP, DOWN, UP, DOWN
    pred_returns = np.array([0.005, -0.01, -0.002, -0.015])  # UP, DOWN, DOWN, DOWN
    # Predictions: 1, 0, 0, 0
    # True: 1, 0, 1, 0
    # Matches: [1==1, 0==0, 0!=1, 0==0] -> 3/4 = 75%
    
    dir_metrics = compute_directional_metrics(actual_returns, pred_returns, threshold=0.0)
    assert dir_metrics["Directional_Accuracy"] == 75.0
    assert dir_metrics["Confusion_Matrix"]["TP"] == 1
    assert dir_metrics["Confusion_Matrix"]["TN"] == 2
    assert dir_metrics["Confusion_Matrix"]["FN"] == 1
    assert dir_metrics["Confusion_Matrix"]["FP"] == 0


def test_diebold_mariano_zero_loss_differential():
    e1 = np.random.randn(100) * 0.1
    # If e2 == e1, DM test should produce DM = 0 and p = 1.0
    res = diebold_mariano_test(e1, e1, loss_type="squared")
    assert res["DM_Statistic"] == 0.0
    assert res["p_value"] == 1.0
    assert not res["is_significant_5pct"]


def test_conformal_calibrator_coverage():
    np.random.seed(42)
    cal_actual = np.random.randn(200) * 0.02
    cal_pred = np.zeros(200)
    
    test_actual = np.random.randn(100) * 0.02
    test_pred = np.zeros(100)
    prev_closes = np.full(100, 150.0)
    
    cp = ConformalPredictor(nominal_coverage=0.90)
    cp.calibrate(cal_actual, cal_pred)
    assert cp.is_calibrated
    
    lower_r, upper_r, lower_p, upper_p = cp.predict_intervals(test_pred, prev_closes)
    cov_res = cp.evaluate_coverage(prev_closes * np.exp(test_actual), lower_p, upper_p)
    
    assert cov_res["Nominal_Coverage"] == 90.0
    # Should be close to 90% (e.g. 80% to 98% empirically)
    assert 75.0 <= cov_res["Empirical_Coverage"] <= 99.0
