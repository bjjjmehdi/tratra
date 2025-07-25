# src/data_ingestion/ib_stream.py  (patched lines only)
import asyncio
import random
import time
from typing import AsyncGenerator, Dict, Tuple

from ib_insync import IB, Contract, Forex, Stock, Ticker

from utils.config import load_config
from utils.logger import get_logger
from utils.market_hours import is_market_hours

cfg = load_config()
log = get_logger("IB_STREAM")


class IBStreamer:
    def __init__(self, ib_cfg: Dict) -> None:
        self.ib = IB()
        self.cfg = ib_cfg
        self._subscribed: set[Tuple[str, str, str]] = set()

    # ---------- patched connect ----------
    async def connect(self) -> None:
        """Connect once with short timeout; fail fast and clear."""
        try:
            await self.ib.connectAsync(
                host=self.cfg["host"],
                port=self.cfg["port"],
                clientId=self.cfg["client_id"],
                timeout=5,
            )
            log.info("IB connected")
        except Exception as e:
            log.error("IB connect failed: %s", e)
            raise  # bubble up so supervisor can decide to retry or abort

    # ---------- rest of file untouched ----------
    async def tick_stream(self) -> AsyncGenerator[Ticker, None]:
        contracts = [
            *(Forex(pair) for pair in cfg["symbols"]["forex"]),
            *(Stock(sym, "SMART", "USD") for sym in cfg["symbols"]["stocks"]),
        ]
        for c in contracts:
            key = (c.symbol, c.exchange, c.currency)
            if key not in self._subscribed:
                self.ib.reqMktData(c, "", False, False)
                self._subscribed.add(key)

        async for tickers in self.ib.pendingTickersEvent:
            if not is_market_hours():
                await asyncio.sleep(1)
                continue
            for t in tickers:
                yield t