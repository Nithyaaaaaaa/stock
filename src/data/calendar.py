"""
Exchange Calendar Module:
Determines authentic market trading days accounting for weekends and exchange holidays.
Supports US markets (NYSE / NASDAQ) and Indian markets (NSE / BSE),
with a safe fallback to business days if calendar data is unavailable.
"""

from typing import Optional, Set
from datetime import date
import logging
import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, GoodFriday, nearest_workday
)

logger = logging.getLogger(__name__)


class NYSEExchangeCalendar(AbstractHolidayCalendar):
    """Official market holidays for New York Stock Exchange and NASDAQ."""
    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        Holiday("Martin Luther King Jr. Day", month=1, day=1, offset=pd.DateOffset(weekday=MO(3))),
        Holiday("Presidents Day", month=2, day=1, offset=pd.DateOffset(weekday=MO(3))),
        GoodFriday,
        Holiday("Memorial Day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))),
        Holiday("Juneteenth National Independence Day", month=6, day=19, observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving Day", month=11, day=1, offset=pd.DateOffset(weekday=TH(4))),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


class NSEExchangeCalendar(AbstractHolidayCalendar):
    """National Stock Exchange of India (NSE) primary fixed national and market holidays."""
    rules = [
        Holiday("Republic Day", month=1, day=26, observance=nearest_workday),
        GoodFriday,
        Holiday("Independence Day India", month=8, day=15, observance=nearest_workday),
        Holiday("Mahatma Gandhi Jayanti", month=10, day=2, observance=nearest_workday),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


# Module-level cached singletons
_NYSE_CALENDAR = NYSEExchangeCalendar()
_NSE_CALENDAR = NSEExchangeCalendar()


def get_next_trading_day(last_date: pd.Timestamp, ticker: str = "AAPL") -> pd.Timestamp:
    """
    Computes the authentic next trading session date following last_date.
    Properly skips weekends (Saturday/Sunday) and official exchange holidays.
    
    Parameters
    ----------
    last_date : pd.Timestamp
        Timestamp of the latest observed market session.
    ticker : str
        Ticker symbol used to select the correct exchange calendar (e.g. NYSE vs NSE).
        
    Returns
    -------
    pd.Timestamp
        Timestamp representing the next open trading session.
    """
    try:
        ticker_upper = str(ticker).strip().upper()
        if ticker_upper.endswith((".NS", ".BO")):
            cal = _NSE_CALENDAR
        else:
            cal = _NYSE_CALENDAR
            
        # Precompute holidays over a 20-day forward window
        window_end = last_date + pd.Timedelta(days=20)
        hols_series = cal.holidays(start=last_date, end=window_end)
        holiday_dates: Set[date] = set(hols_series.date)
        
        curr = last_date + pd.Timedelta(days=1)
        # Advance through weekends and exchange holidays
        while curr.weekday() >= 5 or curr.date() in holiday_dates:
            curr += pd.Timedelta(days=1)
            
        return pd.Timestamp(curr.date())
        
    except Exception as e:
        logger.warning(f"Exchange calendar lookup failed for {ticker}: {e}. Using safe business-day fallback.")
        # Safe fallback: business day range (skips weekends)
        fallback = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)[0]
        return pd.Timestamp(fallback.date())
