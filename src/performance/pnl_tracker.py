# src/performance/pnl_tracker.py
"""Real-time PnL + equity curve writer – lazy init until IB is ready."""
import datetime as dt
import json
from pathlib import Path
from typing import Dict

import pandas as pd
from prometheus_client import Gauge, CollectorRegistry
from ib_insync import IB

from src.utils.config import load_config
from src.utils.logger import get_logger

cfg = load_config()
log = get_logger("PNL_TRACKER")

# -------------- ONE registry for the whole app --------------
REGISTRY = CollectorRegistry(auto_describe=False)

EQUITY_CURVE_FILE = Path("data/equity_curve.jsonl")
EQUITY_CURVE_FILE.parent.mkdir(parents=True, exist_ok=True)

NAV_GAUGE = Gauge(
    "net_liquidation",
    "Net liquidation value in base currency",
    registry=REGISTRY,
)


class PnLTracker:
    def __init__(self, ib: IB) -> None:
        self.ib = ib
        self.start_nav = self._wait_for_nav()

    def _wait_for_nav(self) -> float:
        while True:
            try:
                values = self.ib.accountValues(account="")
                for av in values:
                    if av.key == "NetLiquidation":
                        return float(av.value)
            except (IndexError, ValueError):
                pass
            import time
            time.sleep(1)

    def _append(self, row: Dict[str, object]) -> None:
        with EQUITY_CURVE_FILE.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def tick(self) -> None:
        nav = float(self.ib.accountValues(account="")[0].netLiquidation)
        NAV_GAUGE.set(nav)
        self._append({"ts": dt.datetime.utcnow().isoformat(), "nav": nav})

    def equity_df(self) -> pd.DataFrame:
        if not EQUITY_CURVE_FILE.exists():
            return pd.DataFrame(columns=["ts", "nav"])
        df = pd.read_json(EQUITY_CURVE_FILE, lines=True)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.sort_values("ts").reset_index(drop=True)

    def live_sharpe(self, days: int = 30) -> float:
        df = self.equity_df()
        if len(df) < 2:
            return 0.0
        df = df.tail(days * 24 * 60)
        returns = df["nav"].pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * (252 * 24 * 60) ** 0.5