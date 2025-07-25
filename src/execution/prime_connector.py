"""REST + FIX bridge to Prime-Broker / FCM."""
import asyncio
import os
from typing import Dict

import httpx
from ib_insync import Contract, Order

from utils.logger import get_logger

log = get_logger("PRIME")

class PrimeConnector:
    def __init__(self) -> None:
        self.enabled = bool(os.getenv("PRIME_API_KEY"))
        self.base = os.getenv("PRIME_BASE_URL", "https://api.prime.example.com")
        self.key = os.getenv("PRIME_API_KEY")

    async def send_order(self, order: Order, contract: Contract) -> Dict:
        if not self.enabled:
            return {"status": "mock", "order_id": "local"}

        payload = {
            "symbol": contract.symbol,
            "action": order.action,
            "qty": order.totalQuantity,
            "orderType": order.orderType,
            "limitPrice": order.lmtPrice if hasattr(order, "lmtPrice") else None,
            "tif": order.tif,
            "exchange": contract.exchange,
        }
        headers = {"Authorization": f"Bearer {self.key}"}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{self.base}/orders", json=payload, headers=headers)
            r.raise_for_status()
            return r.json()