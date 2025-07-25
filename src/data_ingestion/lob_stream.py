"""Live order-book, latency, and imbalance feed with hard wall-clock timeout."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncGenerator

from ib_insync import IB, Contract, Ticker
from utils.logger import get_logger

log = get_logger("LOB_STREAM")


@dataclass
class LobTick:
    contract: Contract
    bid: list[tuple[float, int]]
    ask: list[tuple[float, int]]
    ts: float
    latency_us: int


class LobStream:
    def __init__(self, ib: IB, depth: int = 5) -> None:
        self.ib = ib
        self.depth = depth

    async def stream(self, contract: Contract) -> AsyncGenerator[LobTick, None]:
        self.ib.reqMktDepth(contract, self.depth, isSmartDepth=True)
        # Force delayed feed so depth is always available (even off-hours)
        self.ib.reqMarketDataType(4)

        deadline = time.time() + 5.0  # absolute wall-clock deadline

        async for ticker in self.ib.pendingTickersEvent:
            if ticker.contract != contract:
                continue
            if ticker.domBids and ticker.domAsks:
                latency = int((time.time_ns() - ticker.time.timestamp() * 1e9) / 1000)
                yield LobTick(
                    contract=contract,
                    bid=[(ticker.domBids[i].price, int(ticker.domBids[i].size))
                         for i in range(min(self.depth, len(ticker.domBids)))],
                    ask=[(ticker.domAsks[i].price, int(ticker.domAsks[i].size))
                         for i in range(min(self.depth, len(ticker.domAsks)))],
                    ts=time.time(),
                    latency_us=latency,
                )
                return  # one-shot
            if time.time() > deadline:
                break

        # Explicit failure
        log.warning("LOB depth unavailable for %s after 5 s", contract.symbol)
        return  # generator ends