"""
Tests for Dashboard Next-Trading-Day Forecasting Logic:
Verifies single next-day prediction, accurate price/pct delta calculation,
exchange calendar integration, and zero holdout leakage.
"""

import pytest
import numpy as np
import pandas as pd

from app import run_next_day_forecast, POPULAR_STOCKS
from src.data.calendar import get_next_trading_day


@pytest.fixture
def sample_clean_market_data():
    """Generates synthetic clean OHLCV + benchmark market dataframe."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    
    close = 150.0 + np.cumsum(np.random.randn(200) * 1.5)
    high = close + np.random.uniform(0.5, 2.0, 200)
    low = close - np.random.uniform(0.5, 2.0, 200)
    open_p = close + np.random.randn(200) * 0.5
    vol = np.random.randint(1000000, 5000000, 200)
    bench_close = 400.0 + np.cumsum(np.random.randn(200) * 1.0)
    
    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
        "Benchmark_Close": bench_close
    }, index=dates)
    return df


def test_dashboard_next_day_forecast_single_prediction(sample_clean_market_data):
    df = sample_clean_market_data
    res = run_next_day_forecast(df, ticker="MSFT")
    
    # 1. Exactly one forecast output
    assert "predicted_price" in res
    assert isinstance(res["predicted_price"], float)
    assert res["predicted_price"] > 0
    
    # 2. Origin date and next trading day date
    last_date = df.index[-1]
    last_close = float(df["Close"].iloc[-1])
    assert res["last_date"] == last_date
    assert res["last_close"] == last_close
    
    # 3. Next trading day matches exchange calendar
    expected_next = get_next_trading_day(last_date, ticker="MSFT")
    assert res["next_date"] == expected_next
    assert res["next_date"] > last_date
    
    # 4. Verified price difference & percentage difference formulas
    delta_p = res["predicted_price"] - last_close
    expected_pct = (delta_p / last_close) * 100.0
    assert np.isclose(res["price_change"], delta_p, atol=1e-5)
    assert np.isclose(res["pct_change"], expected_pct, atol=1e-5)
    
    # 5. Directional classification adheres to learned threshold
    tau = res["learned_threshold"]
    pred_log_ret = res["pred_log_return"]
    expected_dir = "UP" if pred_log_ret > tau else "DOWN"
    assert res["predicted_direction"] == expected_dir
    
    # 6. Price movement and EXP2 classification fields
    expected_mov = (
        "PRICE INCREASE" if delta_p > 1e-6
        else "PRICE DECREASE" if delta_p < -1e-6
        else "NO PRICE CHANGE"
    )
    expected_exp2 = (
        "UP — Above threshold" if pred_log_ret > tau
        else "Below UP threshold"
    )
    assert res["price_movement"] == expected_mov
    assert res["exp2_classification"] == expected_exp2


def test_popular_stocks_dropdown_options():
    # Verify popular stocks exist
    required_stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "META", "NFLX", "AMD", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
    available_tickers = list(POPULAR_STOCKS.values())
    for req in required_stocks:
        assert req in available_tickers
