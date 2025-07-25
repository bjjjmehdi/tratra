"""Real-time Reg-T / SMA buffer tracking."""
import datetime as dt
from decimal import Decimal
from typing import Dict

from ib_insync import IB
from utils.config import load_config

cfg = load_config()

class RegTGuard:
    def __init__(self, ib: IB) -> None:
        self.ib = ib
        self.min_sma_ratio = cfg["reg_t"]["min_sma_ratio"]

    def snapshot(self) -> Dict[str, float]:
        vals = {v.key: float(v.value) for v in self.ib.accountValues()}
        sma = vals.get("SMA", 0.0)
        equity = vals.get("NetLiquidation", 0.0)
        buying_power = vals.get("BuyingPower", 0.0)
        return {
            "sma_usd": sma,
            "sma_ratio": sma / max(equity, 1e-9),
            "buying_power": buying_power,
            "breach": sma / max(equity, 1e-9) < self.min_sma_ratio,
        }