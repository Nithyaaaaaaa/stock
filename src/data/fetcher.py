"""
Data Fetcher Module: Retrieves and validates raw market data with benchmark context.
Ensures point-in-time integrity with no forward lookahead.
"""

from typing import Optional, Tuple
import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_market_data(
    ticker: str = "AAPL",
    benchmark_ticker: str = "SPY",
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch historical daily OHLCV data for target asset and benchmark index.
    
    Parameters
    ----------
    ticker : str
        Primary equity ticker (e.g. 'AAPL', 'MSFT').
    benchmark_ticker : str
        Macro market benchmark ticker (e.g. 'SPY').
    start : str
        Historical start date (YYYY-MM-DD).
    end : Optional[str]
        Historical end date (exclusive). None fetches up to the latest trading day.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Target DataFrame and Benchmark DataFrame with standardized columns:
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'].
    """
    ticker = (ticker or "").strip().upper()
    benchmark_ticker = (benchmark_ticker or "").strip().upper() or "SPY"
    
    if not ticker:
        raise ValueError("A valid non-empty stock ticker must be specified (e.g. 'AAPL', 'MSFT').")
        
    logger.info("Fetching market data for target='%s', benchmark='%s' from %s to %s", ticker, benchmark_ticker, start, end)
    
    # Download target asset safely (threads=False avoids ThreadPoolExecutor(0) runtime errors)
    target_raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
    if isinstance(target_raw.columns, pd.MultiIndex):
        target_raw.columns = target_raw.columns.get_level_values(0)
    
    if target_raw.empty:
        raise ValueError(f"No market data returned for target ticker '{ticker}'. Please verify the ticker symbol.")
        
    # Download benchmark asset
    bench_raw = yf.download(benchmark_ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
    if isinstance(bench_raw.columns, pd.MultiIndex):
        bench_raw.columns = bench_raw.columns.get_level_values(0)
        
    if bench_raw.empty:
        raise ValueError(f"No market data returned for benchmark ticker '{benchmark_ticker}'.")

    # Standardize column names
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in target_raw.columns:
            raise ValueError(f"Missing required price column '{col}' in target data.")
            
    if "Adj Close" not in target_raw.columns:
        target_raw["Adj Close"] = target_raw["Close"]
    if "Adj Close" not in bench_raw.columns:
        bench_raw["Adj Close"] = bench_raw["Close"]

    target_df = target_raw[expected_cols].copy()
    bench_df = bench_raw[expected_cols].copy()

    # Index handling: ensure tz-naive DatetimeIndex sorted chronologically
    target_df.index = pd.to_datetime(target_df.index).tz_localize(None)
    bench_df.index = pd.to_datetime(bench_df.index).tz_localize(None)
    
    target_df = target_df.sort_index().drop_duplicates()
    bench_df = bench_df.sort_index().drop_duplicates()

    # Drop zero/negative price anomalies
    target_df = target_df[target_df["Close"] > 0]
    bench_df = bench_df[bench_df["Close"] > 0]

    logger.info("Successfully fetched %d sessions for %s, %d sessions for %s", len(target_df), ticker, len(bench_df), benchmark_ticker)
    return target_df, bench_df
