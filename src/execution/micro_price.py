"""Queue-aware sizing + adverse-selection cost estimate."""
from dataclasses import dataclass
from typing import Dict

from data_ingestion.lob_stream import LobTick
from utils.config import load_config

cfg = load_config()

@dataclass
class SizedOrder:
    action: str          # "BUY" | "SELL" | "HOLD"
    qty: int
    limit: float
    stop: float
    take: float

@dataclass
class MicroPrice:
    fair_px: float
    cost_bps: float
    queue_ahead: int

class MicroPriceEngine:
    def __init__(self) -> None:
        self.alpha = cfg["micro"]["adverse_alpha"]   # half-life 30 s

    def compute(self, lob: LobTick, desired_qty: int) -> MicroPrice:
        bid_vol = sum(v for _, v in lob.bid)
        ask_vol = sum(v for _, v in lob.ask)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        cost_bps = self.alpha * abs(imbalance)
        queue_ahead = lob.ask[0][1] if desired_qty > 0 else lob.bid[0][1]
        fair_px = lob.ask[0][0] if desired_qty > 0 else lob.bid[0][0]
        return MicroPrice(fair_px, cost_bps, queue_ahead)