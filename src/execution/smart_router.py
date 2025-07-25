"""Decides passive vs aggressive, venue, and expected fee."""
from dataclasses import dataclass
from typing import List, Literal

from data_ingestion.lob_stream import LobTick

@dataclass
class Route:
    action: Literal["PASSIVE", "AGGRESSIVE"]
    venue: str
    limit_price: float
    expected_fee_bps: float
    queue_ahead: int        # contracts ahead of us on level-1

class SmartRouter:
    def __init__(self, venue_fees_bps: dict[str, float]) -> None:
        self.venue_fees = venue_fees_bps  # {"SMART": 0.3, "ARCA": 0.2, ...}

    def route(self, lob: LobTick, side: str, qty: int) -> Route:
        if side == "BUY":
            level = lob.ask[0] if lob.ask else (0, 0)
            fee = self.venue_fees.get("SMART", 0.3)
            queue_ahead = level[1] if level else 0
            if qty < level[1] * 0.5:  # cheap fill
                return Route("PASSIVE", "SMART", level[0], fee, queue_ahead)
            else:
                return Route("AGGRESSIVE", "SMART", level[0], fee, 0)
        else:  # SELL
            level = lob.bid[0] if lob.bid else (0, 0)
            fee = self.venue_fees.get("SMART", 0.3)
            queue_ahead = level[1] if level else 0
            if qty < level[1] * 0.5:
                return Route("PASSIVE", "SMART", level[0], fee, queue_ahead)
            else:
                return Route("AGGRESSIVE", "SMART", level[0], fee, 0)