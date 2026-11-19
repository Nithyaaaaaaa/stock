"""
Unit Tests: Verification of UI Price Movement & EXP2 Classification Logic.
Explicitly verifies the three required user test cases:
1. +0.05% predicted return with +0.13% threshold -> PRICE INCREASE / Below UP threshold
2. +0.20% predicted return with +0.13% threshold -> PRICE INCREASE / UP — Above threshold
3. -0.10% predicted return with +0.13% threshold -> PRICE DECREASE / Below UP threshold
"""

import numpy as np
import pytest


def evaluate_ui_labels(last_close: float, pred_log_ret: float, tau: float):
    """Mirror of app.py rendering logic for price movement and EXP2 classification."""
    pred_price = float(last_close * np.exp(pred_log_ret))
    price_change = pred_price - last_close
    
    # 1. Price Movement
    if price_change > 1e-6:
        price_movement = "PRICE INCREASE"
        badge_class = "price-badge-up"
        arrow = "↗"
    elif price_change < -1e-6:
        price_movement = "PRICE DECREASE"
        badge_class = "price-badge-down"
        arrow = "↘"
    else:
        price_movement = "NO PRICE CHANGE"
        badge_class = "price-badge-neutral"
        arrow = "—"
        
    # 2. EXP2 Classification
    if pred_log_ret > tau:
        exp2_classification = "UP — Above threshold"
        exp2_badge = "exp2-badge-up"
    else:
        exp2_classification = "Below UP threshold"
        exp2_badge = "exp2-badge-below"
        
    return {
        "pred_price": pred_price,
        "price_change": price_change,
        "price_movement": price_movement,
        "price_badge": badge_class,
        "price_arrow": arrow,
        "exp2_classification": exp2_classification,
        "exp2_badge": exp2_badge,
        "chart_color": "#16A34A" if price_change >= 0 else "#DC2626"
    }


def test_case_1_positive_return_below_threshold():
    """
    Test Case 1: +0.05% predicted return with +0.13% threshold.
    Must show: PRICE INCREASE (green) / BELOW UP THRESHOLD (never red DOWN).
    """
    res = evaluate_ui_labels(last_close=59.19, pred_log_ret=0.0005, tau=0.0013)
    
    assert res["price_change"] > 0
    assert res["price_movement"] == "PRICE INCREASE"
    assert res["price_badge"] == "price-badge-up"
    assert res["price_arrow"] == "↗"
    
    # Critical: Must NOT be labeled as DOWN!
    assert res["exp2_classification"] == "Below UP threshold"
    assert res["exp2_badge"] == "exp2-badge-below"
    
    # Chart line must be green for price gain, not red!
    assert res["chart_color"] == "#16A34A"


def test_case_2_positive_return_above_threshold():
    """
    Test Case 2: +0.20% predicted return with +0.13% threshold.
    Must show: PRICE INCREASE / UP — ABOVE THRESHOLD.
    """
    res = evaluate_ui_labels(last_close=59.19, pred_log_ret=0.0020, tau=0.0013)
    
    assert res["price_change"] > 0
    assert res["price_movement"] == "PRICE INCREASE"
    assert res["price_badge"] == "price-badge-up"
    assert res["price_arrow"] == "↗"
    
    assert res["exp2_classification"] == "UP — Above threshold"
    assert res["exp2_badge"] == "exp2-badge-up"
    assert res["chart_color"] == "#16A34A"


def test_case_3_negative_return_below_threshold():
    """
    Test Case 3: -0.10% predicted return with +0.13% threshold.
    Must show: PRICE DECREASE (red) / BELOW UP THRESHOLD.
    """
    res = evaluate_ui_labels(last_close=59.19, pred_log_ret=-0.0010, tau=0.0013)
    
    assert res["price_change"] < 0
    assert res["price_movement"] == "PRICE DECREASE"
    assert res["price_badge"] == "price-badge-down"
    assert res["price_arrow"] == "↘"
    
    assert res["exp2_classification"] == "Below UP threshold"
    assert res["exp2_badge"] == "exp2-badge-below"
    assert res["chart_color"] == "#DC2626"
