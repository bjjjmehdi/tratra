"""
Vectorised back-test using historical 1-min bars + synthetic LOB.
"""
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from agents.technical_agent import TechnicalAgent
from encoders.multimodal import MultiModalEncoder
from execution.impact_model import ImpactModel
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("BACKTEST")


class BackTestEngine:
    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol
        self.agent = TechnicalAgent()
        self.encoder = MultiModalEncoder()
        self.impact = ImpactModel()

    def run(self, bars: pd.DataFrame) -> pd.DataFrame:
        trades = []
        for ts, row in tqdm(bars.iterrows(), total=len(bars)):
            # mock LOB
            lob = FakeLob(row["close"])
            png = self._render_png(bars.loc[:ts].tail(60))
            vec = self.encoder.encode_live(png, lob, "")
            action, conf = self.agent.decide(png, bars.loc[:ts].tail(60))
            if action != "HOLD":
                impact = self.impact.estimate(100, action, lob)
                trades.append(
                    {
                        "ts": ts,
                        "action": action,
                        "price": impact.expected_price,
                        "slippage_bps": impact.slippage_bps,
                    }
                )
        return pd.DataFrame(trades)

    def _render_png(self, df: pd.DataFrame) -> bytes:
        from data_ingestion.candle_builder import CandleBuilder
        return CandleBuilder().render_png(df)


class FakeLob:
    def __init__(self, mid: float):
        self.bid = [(mid - 0.01 * i, 100) for i in range(1, 6)]
        self.ask = [(mid + 0.01 * i, 100) for i in range(1, 6)]