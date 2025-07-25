"""
Almgren-Chriss style market-impact + passive-fill probability.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from data_ingestion.lob_stream import LobTick

@dataclass
class ImpactEstimate:
    expected_price: float
    expected_participation: float  # % of qty that will fill passively
    slippage_bps: float

class ImpactModel:
    def __init__(self, gamma: float = 0.5, eta: float = 0.1):
        self.gamma = gamma  # temporary impact coefficient
        self.eta = eta      # permanent impact coefficient

    def estimate(self, qty: int, side: str, lob: LobTick) -> ImpactEstimate:
        book = lob.ask if side == "BUY" else lob.bid
        if not book:
            return ImpactEstimate(lob.ask[0][0] if side == "BUY" else lob.bid[0][0], 0.0, 0.0)

        price_levels = np.array([p for p, _ in book])
        sizes = np.array([s for _, s in book])
        cum_size = np.cumsum(sizes)
        idx = np.searchsorted(cum_size, qty)
        if idx >= len(price_levels):
            # sweep book
            expected_price = price_levels[-1]
            participation = 0.0
        else:
            expected_price = price_levels[idx]
            participation = (cum_size[idx] - qty) / cum_size[idx]

        temporary_impact = self.gamma * qty / cum_size[-1]
        permanent_impact = self.eta * qty / cum_size[-1]
        slippage = 1e4 * (expected_price - price_levels[0]) / price_levels[0]

        return ImpactEstimate(expected_price, participation, slippage)