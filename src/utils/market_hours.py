"""Market-hours utilities."""
import datetime as dt
from typing import Tuple

import pytz

ET = pytz.timezone("US/Eastern")

def is_market_hours() -> bool:
    now = dt.datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time

def minutes_to_close() -> int:
    now = dt.datetime.now(ET)
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return max(0, int((close - now).total_seconds() / 60))