"""Real-time alternative alpha: LOB imbalance, options flow, dark-pool prints."""
import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator

import httpx
from ib_insync import Contract

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("ALT_DATA")

@dataclass
class AltBar:
    contract: Contract
    ts: float
    lob_imb: float           # (bid_vol - ask_vol) / total_vol
    gamma_flip: float        # 0-DTE gamma wall flip
    dp_notional: float       # Dark-pool block notional last 30 s

class AltDataFeed:
    """WebSocket streams from Polygon & CBOE."""
    def __init__(self) -> None:
        self.poly_key = cfg["alt_data"]["polygon_key"]
        self.cboe_key = cfg["alt_data"]["cboe_key"]

    async def stream(self, contract: Contract) -> AsyncGenerator[AltBar, None]:
        url = f"wss://socket.polygon.io/options"
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, headers={"Authorization": f"Bearer {self.poly_key}"}) as ws:
                async for msg in ws:
                    data = msg.json()
                    yield AltBar(
                        contract=contract,
                        ts=data["t"],
                        lob_imb=data["lob_imbalance"],
                        gamma_flip=data["gamma_flip"],
                        dp_notional=data["dark_pool_notional"],
                    )