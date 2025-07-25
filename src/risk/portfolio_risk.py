"""Portfolio-level risk: net exposure, sector Greeks, VAR."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from ib_insync import PortfolioItem
from prometheus_client import Gauge
from scipy.stats import norm

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("PORTFOLIO_RISK")

# Prometheus gauges
NET_EXPOSURE = Gauge("net_exposure_usd", "Net dollar exposure")
SECTOR_DELTA = Gauge("sector_delta_greeks", "Sector delta", ["sector"])
PORTFOLIO_VAR = Gauge("portfolio_var_usd", "95 % 1-day VAR in USD")

SECTOR_MAP = {
    "AAPL": "TECH",
    "MSFT": "TECH",
    "JPM": "FIN",
    "XOM": "ENERGY",
    # extend per universe
}

@dataclass
class RiskSnapshot:
    net_exposure: float
    sector_deltas: Dict[str, float]
    var_95: float
    timestamp: dt.datetime


class PortfolioRisk:
    def __init__(self, ib) -> None:
        self.ib = ib
        self.lookback_days = cfg["risk"]["var_lookback_days"]
        self.confidence = cfg["risk"]["var_confidence"]

    def snapshot(self) -> RiskSnapshot:
        try:
            positions = self.ib.positions()
            if not positions:
                return RiskSnapshot(0.0, {}, 0.0, dt.datetime.utcnow())

            # Build DF of positions
            rows = []
            for p in positions:
                px = float(self.ib.reqMktData(p.contract, "", False, False).last or 0)
                dollar = float(p.position) * px
                sector = SECTOR_MAP.get(p.contract.symbol, "OTHER")
                rows.append(
                    {
                        "symbol": p.contract.symbol,
                        "qty": int(p.position),
                        "price": px,
                        "dollar": dollar,
                        "sector": sector,
                    }
                )
            df = pd.DataFrame(rows)

            # Net exposure
            net = df["dollar"].sum()

            # Sector delta (simple dollar per sector)
            sector_deltas = df.groupby("sector")["dollar"].sum().to_dict()

            # 1-day VAR via normal approximation
            # crude: net PnL ~ N(0, σ²); σ from historical net returns
            hist = self._historical_net_returns(net, days=self.lookback_days)
            sigma = hist.std() if len(hist) > 5 else 0.01
            var_95 = abs(norm.ppf(1 - self.confidence) * sigma)

            # Push to Prometheus
            NET_EXPOSURE.set(net)
            for sec, delta in sector_deltas.items():
                SECTOR_DELTA.labels(sector=sec).set(delta)
            PORTFOLIO_VAR.set(var_95)

            return RiskSnapshot(net, sector_deltas, var_95, dt.datetime.utcnow())

        except Exception as e:
            log.exception("Portfolio risk calculation failed: %s", e)
            return RiskSnapshot(0.0, {}, 0.0, dt.datetime.utcnow())

    # ---------- helpers ----------
    def _historical_net_returns(self, current_net: float, days: int) -> pd.Series:
        # stub: load portfolio NAV history
        # for MVP we fake 1 % daily vol
        np.random.seed(42)
        return pd.Series(np.random.normal(0, 0.01, days))