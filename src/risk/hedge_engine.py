"""Beta-adjusted SPY hedge for single-stock position."""
import datetime as dt

import pandas as pd
from ib_insync import IB, Stock

from utils.config import load_config

cfg = load_config()

class HedgeEngine:
    def __init__(self, ib: IB) -> None:
        self.ib = ib
        self.spy_contract = Stock("SPY", "ARCA", "USD")   # FIXED: was INTC/EUR
        self.beta = cfg["hedge"]["beta"]  # could be live-updated

    def hedge_qty(self, symbol: str, qty: int) -> int:
        """Return SPY shares to trade to neutralize beta."""
        return int(-qty * self.beta)