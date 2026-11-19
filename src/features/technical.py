"""
Feature Engineering Module: Point-in-time technical and macro indicators.
All calculations are strictly backward-looking.
Includes raw levels, ratios, oscillators, and volatility metrics to allow empirical stationarity testing.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import warnings
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
import logging

logger = logging.getLogger(__name__)


def compute_all_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute candidate predictive features strictly point-in-time using past data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Open, High, Low, Close, Volume, Benchmark_Close, Benchmark_Volume.
        
    Returns
    -------
    pd.DataFrame
        DataFrame augmented with engineered candidate features and target representations.
    """
    out = df.copy()
    
    # -------------------------------------------------------------
    # 1. Target Formulations (To be shifted backward by 1 for point-in-time feature usage,
    #    or evaluated at t+1 for forecasting objectives)
    # -------------------------------------------------------------
    # Daily relative return: r_t = (P_t - P_{t-1}) / P_{t-1}
    out["Target_Relative_Return"] = out["Close"].pct_change()
    
    # Daily log return: y_t = ln(P_t / P_{t-1})
    out["Target_Log_Return"] = np.log(out["Close"] / out["Close"].shift(1))
    
    # Directional binary target: 1 if return > 0 else 0
    out["Target_Direction"] = (out["Target_Relative_Return"] > 0).astype(int)
    
    # Raw price level target (t+1)
    out["Target_Next_Close"] = out["Close"].shift(-1)
    
    # Next-period targets (shifted so that features at row t predict target at row t)
    out["Target_Next_Rel_Return"] = out["Target_Relative_Return"].shift(-1)
    out["Target_Next_Log_Return"] = out["Target_Log_Return"].shift(-1)
    out["Target_Next_Direction"] = out["Target_Direction"].shift(-1)

    # -------------------------------------------------------------
    # 2. Historical Returns (Lags strictly from t-1, t-2, ...)
    # -------------------------------------------------------------
    out["Return_Lag_1"] = out["Target_Relative_Return"]
    out["Return_Lag_2"] = out["Target_Relative_Return"].shift(1)
    out["Return_Lag_5"] = out["Close"].pct_change(5)
    out["Return_Lag_10"] = out["Close"].pct_change(10)
    out["Log_Return_Lag_1"] = out["Target_Log_Return"]
    out["Log_Return_Lag_5"] = np.log(out["Close"] / out["Close"].shift(5))

    # -------------------------------------------------------------
    # 3. Trend & Moving Average Ratios (Normalized to avoid unit root)
    # -------------------------------------------------------------
    # Ratios vs Rolling Means
    sma_5 = out["Close"].rolling(window=5).mean()
    sma_20 = out["Close"].rolling(window=20).mean()
    sma_50 = out["Close"].rolling(window=50).mean()
    
    out["Ratio_Close_SMA_5"] = (out["Close"] / sma_5) - 1.0
    out["Ratio_Close_SMA_20"] = (out["Close"] / sma_20) - 1.0
    out["Ratio_Close_SMA_50"] = (out["Close"] / sma_50) - 1.0
    out["Ratio_SMA_5_20"] = (sma_5 / sma_20) - 1.0

    # Exponential Moving Averages
    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["Ratio_Close_EMA_12"] = (out["Close"] / ema_12) - 1.0
    out["Ratio_Close_EMA_26"] = (out["Close"] / ema_26) - 1.0
    out["MACD_Normalized"] = (ema_12 - ema_26) / out["Close"]
    
    # Raw indicator levels (included intentionally for stationarity verification)
    out["Raw_SMA_20"] = sma_20
    out["Raw_EMA_12"] = ema_12

    # -------------------------------------------------------------
    # 4. Momentum Oscillators
    # -------------------------------------------------------------
    # RSI (Relative Strength Index - 14 day)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0.0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    out["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
    out["RSI_14_Centered"] = (out["RSI_14"] / 100.0) - 0.5  # centered in [-0.5, 0.5]

    # Stochastic Oscillator %K (14-day)
    low_14 = out["Low"].rolling(window=14).min()
    high_14 = out["High"].rolling(window=14).max()
    out["Stoch_K_14"] = ((out["Close"] - low_14) / (high_14 - low_14 + 1e-9)) - 0.5

    # Intraday Price Spread Ratios
    out["High_Low_Ratio"] = (out["High"] - out["Low"]) / out["Close"]
    out["Close_Open_Ratio"] = (out["Close"] - out["Open"]) / out["Open"]

    # -------------------------------------------------------------
    # 5. Volatility & Risk Metrics
    # -------------------------------------------------------------
    # Rolling standard deviations of daily returns
    out["Vol_10d"] = out["Target_Relative_Return"].rolling(window=10).std()
    out["Vol_20d"] = out["Target_Relative_Return"].rolling(window=20).std()
    out["Vol_60d"] = out["Target_Relative_Return"].rolling(window=60).std()
    
    # Normalized Average True Range (NATR)
    prev_close = out["Close"].shift(1)
    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - prev_close).abs()
    tr3 = (out["Low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(window=14).mean()
    out["NATR_14"] = atr_14 / out["Close"]
    out["Raw_ATR_14"] = atr_14  # Unnormalized for stationarity check

    # Bollinger Bands %B
    rolling_std_20 = out["Close"].rolling(window=20).std()
    upper_band = sma_20 + 2.0 * rolling_std_20
    lower_band = sma_20 - 2.0 * rolling_std_20
    out["Bollinger_Pct_B"] = (out["Close"] - lower_band) / (upper_band - lower_band + 1e-9)

    # -------------------------------------------------------------
    # 6. Volume Indicators
    # -------------------------------------------------------------
    out["Log_Volume_Change"] = np.log(out["Volume"] / out["Volume"].shift(1))
    vol_sma_20 = out["Volume"].rolling(window=20).mean()
    out["Volume_Ratio_SMA_20"] = (out["Volume"] / (vol_sma_20 + 1e-9)) - 1.0
    out["Raw_Volume"] = out["Volume"]

    # -------------------------------------------------------------
    # 7. Macroeconomic / Benchmark Context (SPY)
    # -------------------------------------------------------------
    out["Benchmark_Return"] = out["Benchmark_Close"].pct_change()
    out["Benchmark_Return_Lag_1"] = out["Benchmark_Return"]
    out["Excess_Return_Over_Benchmark"] = out["Target_Relative_Return"] - out["Benchmark_Return"]
    
    # Rolling Beta against Benchmark (60-day window)
    cov_60 = out["Target_Relative_Return"].rolling(60).cov(out["Benchmark_Return"])
    var_60 = out["Benchmark_Return"].rolling(60).var()
    out["Rolling_Beta_60"] = cov_60 / (var_60 + 1e-9)

    return out


def evaluate_feature_stationarity(
    dev_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Formally evaluate stationarity using both Augmented Dickey-Fuller (ADF)
    and KPSS tests on the DEVELOPMENT SET ONLY.
    
    Never runs on the final holdout set.
    
    ADF: H0: Unit root present (non-stationary). Rejection (p < 0.05) supports stationarity.
    KPSS: H0: Level stationary. Rejection (p < 0.05) indicates non-stationarity.
    """
    warnings.filterwarnings("ignore")
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required for stationarity auditing. Install via pip install statsmodels.")
    results = []
    
    for feat in feature_cols:
        series = dev_df[feat].dropna()
        if len(series) < 100:
            continue
            
        # 1. ADF Test
        try:
            adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series, maxlag=10, autolag="AIC")
        except Exception:
            adf_stat, adf_p = np.nan, np.nan
            
        # 2. KPSS Test
        try:
            kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression="c", nlags="auto")
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan
            
        # Determine empirical verdict
        is_adf_stationary = (adf_p < 0.05) if not np.isnan(adf_p) else False
        is_kpss_stationary = (kpss_p >= 0.05) if not np.isnan(kpss_p) else False
        
        if is_adf_stationary and is_kpss_stationary:
            verdict = "Strictly I(0) (ADF & KPSS Agree)"
        elif is_adf_stationary and not is_kpss_stationary:
            verdict = "Trend / Heteroscedastic I(0)"
        else:
            verdict = "Non-Stationary I(1) (Unit Root)"
            
        results.append({
            "Feature": feat,
            "ADF_Statistic": adf_stat,
            "ADF_p_value": adf_p,
            "KPSS_Statistic": kpss_stat,
            "KPSS_p_value": kpss_p,
            "Empirical_Verdict": verdict,
            "Mean": float(series.mean()),
            "Std": float(series.std()),
            "Skewness": float(series.skew()),
            "Kurtosis": float(series.kurtosis()),
        })
        
    return pd.DataFrame(results).sort_values("ADF_p_value")
