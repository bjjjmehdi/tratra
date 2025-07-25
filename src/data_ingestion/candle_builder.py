"""Lock-free 1-minute OHLCV builder."""
import io
from collections import deque
from datetime import datetime
from typing import List

import mplfinance as mpf
import pandas as pd
from ib_insync import Contract, Ticker

from utils.logger import get_logger

log = get_logger("CANDLE_BUILDER")

class CandleBuilder:
    def __init__(self, lookback: int = 60) -> None:
        self.lookback = lookback
        self._ticks: deque = deque(maxlen=10_000)

    def add_tick(self, contract: Contract, tick: Ticker) -> bool:
        now = datetime.utcnow()
        self._ticks.append(
            {"t": now, "p": float(tick.last or tick.close or 0), "v": tick.volume or 0}
        )
        return now.second == 0

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self._ticks).set_index("t")
        return df["p"].resample("1min").ohlc().dropna()

    def render_png(self, df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        mpf.plot(
            df.tail(self.lookback),
            type="candle",
            style="charles",
            figsize=(10, 6),
            mav=(20, 50),
            volume=False,
            savefig=dict(fname=buf, format="png", dpi=120),
        )
        buf.seek(0)
        return buf.read()

    def reset(self) -> None:
        self._ticks.clear()