"""Zero-defect broker with micro-structure, smart routing, and fill prediction."""
import asyncio
import time
from decimal import Decimal
from typing import Dict

from ib_insync import (
    IB,
    BracketOrder,
    Contract,
    LimitOrder,
    MarketOrder,
    Order,
    Trade,
)
from prometheus_client import Counter, Gauge

from data_ingestion.lob_stream import LobStream, LobTick
from execution.fill_model import FillModel
from execution.risk import RiskManager, SizedOrder
from execution.smart_router import SmartRouter, Route
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("BROKER")

# ---------------- Prometheus ----------------
ORDERS_SENT = Counter("orders_sent_total", "Orders sent")
DAILY_PNL = Gauge("daily_pnl_pct", "Daily P&L %")
SLIPPAGE_BPS = Gauge("slippage_bps", "Observed slippage bps")
FILL_PROB = Gauge("fill_probability", "Predicted passive-fill probability")


class Broker:
    def __init__(self, risk_cfg: Dict) -> None:
        self.ib = IB()
        self.risk = RiskManager(risk_cfg)
        self.start_nav = 0.0  # set on first nav read

        # NEW: micro-structure stack
        self.lob = LobStream(self.ib)
        symbol = risk_cfg.get("symbol", "UNDEF")  # or pull from contract later
        self.router = SmartRouter(risk_cfg.get("venue_fees", {"SMART": 0.3}))
        self.fill_model = FillModel(symbol)

    # ---------------- helpers ----------------
    def _get_nav(self) -> float:
        try:
            return float(self.ib.accountValues(account="")[0].netLiquidation)
        except Exception as e:
            log.exception("Could not read NAV – returning 0: %s", e)
            return 0.0

    def _margin_usage(self) -> float:
        try:
            account = self.ib.accountValues(account="")[0]
            excess = float(account.excessLiquidity)
            gross = float(account.grossPositionValue)
            return gross / (excess + gross) if excess + gross else 0.0
        except Exception as e:
            log.exception("Margin calc failed – returning 0: %s", e)
            return 0.0

    def position_snapshot(self, contract) -> Dict[str, float]:
        """Return live position dict."""
        try:
            matches = [p for p in self.ib.positions() if p.contract == contract]
            if not matches:
                return {"qty": 0, "avg_price": 0.0, "unreal_pnl": 0.0, "real_pnl": 0.0}
            p = matches[0]
            return {
                "qty": int(p.position),
                "avg_price": float(p.averageCost),
                "unreal_pnl": float(p.unrealPNL or 0),
                "real_pnl": float(p.realPNL or 0),
            }
        except Exception as e:
            log.exception("Position snapshot failed: %s", e)
            return {"qty": 0, "avg_price": 0.0, "unreal_pnl": 0.0, "real_pnl": 0.0}

    # ---------------- main entry ----------------
    async def execute(self, decision, contract) -> None:
        nav = self._get_nav()
        if self.risk.start_nav is None:
            self.risk.start_nav = nav

        margin = self._margin_usage()
        DAILY_PNL.set(self.risk.daily_pnl_pct(nav))

        # 1. kill-switch
        if not self.risk.can_trade(nav):
            log.warning("Risk kill-switch active – skipping order")
            return

        # 2. macro filter
        if await self.risk.high_impact_today():
            log.warning("High-impact macro event – skipping order")
            return

        # 3. sizing
        price = float(self.ib.reqMktData(contract, "", False, False).last or 1)
        sized = self.risk.size_order(nav, price, margin)
        if sized.qty == 0:
            return

        # 4. micro-structure snapshot
        lob_gen = self.lob.stream(contract)
        lob = await lob_gen.__anext__()  # latest order-book

        # 5. smart route
        route = self.router.route(lob, sized.action, sized.qty)
        fill_prob = self.fill_model.predict(
            sized.qty,
            route.queue_ahead,
            lob.latency_us,
        )
        FILL_PROB.set(fill_prob)
        log.info("Routing %s %s fill_prob=%.2f route=%s",
                 sized.action, sized.qty, fill_prob, route)

        # 6. build order
        order: Order
        if route.action == "PASSIVE":
            order = LimitOrder(sized.action, sized.qty, route.limit_price)
        else:
            order = MarketOrder(sized.action, sized.qty)

        # 7. place & observe
        trade: Trade = self.ib.placeOrder(contract, order)
        ORDERS_SENT.inc(sized.qty)
        log.info("Order placed: %s", trade)

        # 8. post-trade learning (async callback)
        def on_fill(tr: Trade, fill):
            filled = tr.filled == sized.qty
            self.fill_model.update(lob, sized.qty, filled)

        trade.filledEvent += on_fill

    # ---------------- emergency ----------------
    async def flatten_all(self) -> None:
        try:
            for pos in self.ib.positions():
                qty = int(pos.position)
                if qty == 0:
                    continue
                action = "SELL" if qty > 0 else "BUY"
                order = MarketOrder(action, abs(qty))
                self.ib.placeOrder(pos.contract, order)
        except Exception as e:
            log.exception("Flatten failed: %s", e)