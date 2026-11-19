"""
Unit Tests: Exchange Trading Calendar (NYSE and NSE) with Weekend & Holiday Handling.
"""

import pytest
import pandas as pd
from src.data.calendar import get_next_trading_day, NYSEExchangeCalendar, NSEExchangeCalendar


def test_nyse_skips_weekends():
    # Normal Friday (Nov 15, 2024) -> Should map to Monday (Nov 18, 2024)
    last_close = pd.Timestamp("2024-11-15")
    next_day = get_next_trading_day(last_close, ticker="AAPL")
    assert next_day == pd.Timestamp("2024-11-18")
    assert next_day.weekday() == 0  # Monday


def test_nyse_skips_thanksgiving():
    # Wednesday before Thanksgiving 2024 (Nov 27, 2024) -> Should skip Thursday (Nov 28) and return Friday (Nov 29)
    last_close = pd.Timestamp("2024-11-27")
    next_day = get_next_trading_day(last_close, ticker="AAPL")
    assert next_day == pd.Timestamp("2024-11-29")
    assert next_day.weekday() == 4  # Friday


def test_nyse_skips_good_friday():
    # Thursday before Good Friday 2024 (Mar 28, 2024) -> Should skip Good Friday (Mar 29), Sat, Sun -> Monday (Apr 1)
    last_close = pd.Timestamp("2024-03-28")
    next_day = get_next_trading_day(last_close, ticker="AAPL")
    assert next_day == pd.Timestamp("2024-04-01")
    assert next_day.weekday() == 0  # Monday


def test_nyse_skips_christmas():
    # Tuesday Dec 24, 2024 -> Should skip Wednesday Dec 25 (Christmas) -> Thursday Dec 26
    last_close = pd.Timestamp("2024-12-24")
    next_day = get_next_trading_day(last_close, ticker="AAPL")
    assert next_day == pd.Timestamp("2024-12-26")
    assert next_day.weekday() == 3  # Thursday


def test_nse_skips_republic_day():
    # Thursday Jan 25, 2024 (NSE Indian market) -> Jan 26, 2024 is Republic Day (Friday) -> Monday Jan 29
    last_close = pd.Timestamp("2024-01-25")
    next_day = get_next_trading_day(last_close, ticker="RELIANCE.NS")
    assert next_day == pd.Timestamp("2024-01-29")
    assert next_day.weekday() == 0  # Monday


def test_calendar_safe_fallback():
    # Safe fallback test when given an unusual input
    last_close = pd.Timestamp("2024-06-14")  # Friday
    next_day = get_next_trading_day(last_close, ticker="UNKNOWN_TICKER")
    assert next_day == pd.Timestamp("2024-06-17")  # Monday
