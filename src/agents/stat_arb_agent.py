"""Statistical arbitrage (pairs) agent."""
import numpy as np
import pandas as pd

class StatArbAgent:
    def __init__(self, window: int = 100) -> None:
        self.window = window

    def decide(self, series1: pd.Series, series2: pd.Series) -> str:
        spread = series1 - series2
        zscore = (spread - spread.rolling(self.window).mean()) / spread.rolling(
            self.window
        ).std()
        z = zscore.iloc[-1]
        if z > 2.0:
            return "SELL"
        if z < -2.0:
            return "BUY"
        return "HOLD"