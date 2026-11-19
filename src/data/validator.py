"""
Data Validator Module: Ensures trading continuity, corporate action handling,
and strictly enforces the chronological 80:20 partition with an untouched holdout set.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def validate_and_clean_data(
    target_df: pd.DataFrame,
    bench_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Synchronize dates, handle missing records, and merge target with benchmark context.
    
    All operations are strictly point-in-time: benchmark values are aligned on matching trade dates.
    """
    # Inner join on common trading dates
    common_dates = target_df.index.intersection(bench_df.index)
    if len(common_dates) < 250:
        raise ValueError(f"Insufficient overlapping trading dates between target and benchmark: {len(common_dates)} days.")
        
    df = target_df.loc[common_dates].copy()
    b_df = bench_df.loc[common_dates].copy()
    
    # Check for NaNs and handle forward fill (max 2 consecutive days for holiday discrepancies)
    if df.isna().sum().sum() > 0:
        logger.warning("Found NaNs in target data; applying forward fill (limit=2)")
        df = df.ffill(limit=2).dropna()
        
    # Add benchmark reference columns with explicit prefix
    df["Benchmark_Close"] = b_df["Close"]
    df["Benchmark_Volume"] = b_df["Volume"]
    
    # Check for zero volumes and replace with small positive value to avoid division by zero
    df["Volume"] = df["Volume"].replace(0, np.nan).ffill().fillna(1.0)
    df["Benchmark_Volume"] = df["Benchmark_Volume"].replace(0, np.nan).ffill().fillna(1.0)
    
    return df


def split_development_and_holdout(
    df: pd.DataFrame,
    holdout_pct: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partition the dataset chronologically into:
    1. Development Set (first (100 - holdout_pct)%) -> Used for feature design, target study, lookback tuning.
    2. Untouched Holdout Test Set (terminal holdout_pct%) -> FROZEN until final model verification.
    
    Parameters
    ----------
    df : pd.DataFrame
        Complete cleaned dataset.
    holdout_pct : float
        Percentage allocated to the untouched final test set (default: 20.0).
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        dev_df, holdout_df, split_metadata
    """
    total_n = len(df)
    train_pct = 100.0 - holdout_pct
    dev_len = int(np.ceil(total_n * (train_pct / 100.0)))
    holdout_len = total_n - dev_len
    
    dev_df = df.iloc[:dev_len].copy()
    holdout_df = df.iloc[dev_len:].copy()
    
    metadata = {
        "total_sessions": total_n,
        "dev_sessions": dev_len,
        "holdout_sessions": holdout_len,
        "dev_pct": (dev_len / total_n) * 100.0,
        "holdout_pct": (holdout_len / total_n) * 100.0,
        "dev_start": dev_df.index[0].strftime("%Y-%m-%d"),
        "dev_end": dev_df.index[-1].strftime("%Y-%m-%d"),
        "holdout_start": holdout_df.index[0].strftime("%Y-%m-%d"),
        "holdout_end": holdout_df.index[-1].strftime("%Y-%m-%d"),
    }
    
    logger.info(
        "Chronological Split: Dev Set=%d sessions (%s to %s) | Untouched Holdout=%d sessions (%s to %s)",
        dev_len, metadata["dev_start"], metadata["dev_end"],
        holdout_len, metadata["holdout_start"], metadata["holdout_end"]
    )
    return dev_df, holdout_df, metadata
