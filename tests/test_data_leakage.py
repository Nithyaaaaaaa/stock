"""
Unit & Integration Tests: Data Leakage and Temporal Integrity.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.validator import validate_and_clean_data, split_development_and_holdout
from src.features.technical import compute_all_candidate_features
from src.evaluation.walk_forward import WalkForwardValidator


def test_no_temporal_overlap_in_development_holdout_split():
    """Verify that development set and holdout set have zero temporal overlap."""
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    df = pd.DataFrame({
        "Open": np.linspace(100, 150, 200),
        "High": np.linspace(101, 151, 200),
        "Low": np.linspace(99, 149, 200),
        "Close": np.linspace(100, 150, 200),
        "Volume": 1000000,
        "SPY_Close": np.linspace(300, 400, 200)
    }, index=dates)
    
    dev, holdout, meta = split_development_and_holdout(df, holdout_pct=20.0)
    
    assert dev.index[-1] < holdout.index[0], "Temporal leakage: dev set must strictly precede holdout set"
    assert len(dev) + len(holdout) == len(df), "Sample count mismatch in split"
    assert len(set(dev.index).intersection(set(holdout.index))) == 0, "Shared timestamps between dev and holdout"


def test_walk_forward_expanding_folds_chronological_ordering():
    """Verify that every walk-forward fold tests strictly on future unseen data with proper embargo."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    df = pd.DataFrame({"Close": np.random.randn(500)}, index=dates)
    
    wf = WalkForwardValidator(n_splits=5, min_train_ratio=0.50, purge_boundary=True)
    folds = wf.generate_folds(df)
    
    assert len(folds) == 5
    for i, f in enumerate(folds):
        tr_end = f["train_indices"][1]
        vl_start = f["val_indices"][0]
        vl_end = f["val_indices"][1]
        
        # With 1-observation boundary purge to prevent forward target overlap:
        assert tr_end <= vl_start, f"Fold {i+1}: training must end before or at validation start"
        assert vl_start < vl_end, f"Fold {i+1}: validation slice must have positive length"
        assert f["train_end_date"] < f["val_start_date"], f"Fold {i+1}: dates must be strictly monotonic"
