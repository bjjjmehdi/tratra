"""Rule-based labeling: BUY if next 5-min return > +0.5 % else HOLD."""
import numpy as np
import pandas as pd
from typing import Tuple

from data_pipeline.label_collector import LabelCollector

class FutureReturnLabeler:
    def __init__(self, horizon_min: int = 5, threshold: float = 0.005) -> None:
        self.horizon_min = horizon_min
        self.threshold = threshold

    def label(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Return (action, reward)."""
        if len(df) < self.horizon_min:
            return "HOLD", 0.0
        future_ret = (df["close"].iloc[-1] / df["close"].iloc[-self.horizon_min]) - 1
        if future_ret > self.threshold:
            return "BUY", future_ret
        elif future_ret < -self.threshold:
            return "SELL", future_ret
        return "HOLD", 0.0