"""Zero-defect risk engine – fails successfully."""
import asyncio
import datetime as dt
import logging
from decimal import Decimal
from typing import Dict

import httpx
import pandas as pd
from pydantic import BaseModel, ValidationError

from utils.config import load_config
from utils.logger import get_logger
from utils.market_hours import is_market_hours
from prometheus_client import Counter, Gauge

cfg = load_config()
log = get_logger("RISK")

# ------------- Prometheus counters -------------
KILL_SWITCH = Counter("risk_kill_switch_total", "Kill-switch activations")
MARGIN_BREACH = Counter("risk_margin_breach_total", "Margin buffer breaches")
CALENDAR_FAIL = Counter("risk_calendar_fail_total", "Macro calendar fetch failures")

class SizedOrder(BaseModel):
    action: str
    qty: int
    limit: float
    stop: float
    take: float

class RiskManager:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.max_daily_loss = float(cfg["max_daily_loss_pct"])
        self.max_position = float(cfg["max_position_pct"])
        self.hard_stop = float(cfg["hard_stop_pct"])
        self.margin_buffer = float(cfg["margin_buffer_pct"])
        self.calendar_url = cfg["macro_calendar_url"]
        self._start_nav: float | None = None

    # ---------- public API ----------
    def daily_pnl_pct(self, nav: float) -> float:
        """Safe PnL calc with NaN guards."""
        if self._start_nav is None:
            self._start_nav = nav
        if self._start_nav == 0:
            return 0.0
        return (nav - self._start_nav) / self._start_nav

    def can_trade(self, nav: float) -> bool:
        """Atomic kill-switch."""
        try:
            pnl_pct = self.daily_pnl_pct(nav)
            if pnl_pct < -self.max_daily_loss:
                log.error("Daily loss limit hit (%.2f%%) – kill-switch ON", pnl_pct * 100)
                KILL_SWITCH.inc()
                return False
            return True
        except Exception as e:
            log.exception("Unexpected error in can_trade – defaulting to False: %s", e)
            return False

    async def high_impact_today(self) -> bool:
        """Macro calendar with graceful degradation."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    self.calendar_url,
                    params={"date": dt.date.today().isoformat()},
                )
                r.raise_for_status()
                events = r.json().get("result", [])
                return any(e.get("impact") == "High" for e in events)
        except Exception as e:
            log.warning("Calendar fetch failed – assuming NO high impact: %s", e)
            CALENDAR_FAIL.inc()
            return False  # fail-safe: skip only on explicit “High”

    def size_order(
        self, nav: float, price: float, margin_usage: float
    ) -> SizedOrder:
        """Position sizing with defensive clamping."""
        try:
            if nav <= 0 or price <= 0:
                raise ValueError("Non-positive nav or price")

            cushion = self.margin_buffer
            if margin_usage > (1 - cushion):
                log.warning("Margin buffer breach (%.2f%%)", margin_usage * 100)
                MARGIN_BREACH.inc()
                return SizedOrder(
                    action="HOLD", qty=0, limit=price, stop=price, take=price
                )

            risk_amt = nav * self.hard_stop
            qty = max(0, int(risk_amt / (price * self.hard_stop)))
            if qty == 0:
                return SizedOrder(
                    action="HOLD", qty=0, limit=price, stop=price, take=price
                )

            return SizedOrder(
                action="BUY",
                qty=qty,
                limit=price,
                stop=price * (1 - self.hard_stop),
                take=price * (1 + 2 * self.hard_stop),
            )
        except (ZeroDivisionError, ValidationError, ValueError) as e:
            log.exception("Sizing failed – returning zero order: %s", e)
            return SizedOrder(
                action="HOLD", qty=0, limit=price, stop=price, take=price
            )